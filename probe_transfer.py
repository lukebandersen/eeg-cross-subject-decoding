import glob, os, sys, csv
sys.path.insert(0, os.path.join(os.getcwd(), "Retrieval"))
import numpy as np
import torch
from torch.utils.data import DataLoader
from eegdatasets import EEGDataset

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.environ.get("THINGS_DATA")
IMG_TRAIN = os.environ.get("THINGS_IMG_TRAIN")
IMG_TEST = os.environ.get("THINGS_IMG_TEST")
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]   # FULL RUN
N_CONCEPTS = 100
CHANCE = 1.0 / N_CONCEPTS
DUMMY_N = 20000
OUT_CSV = "probe_transfer_results.csv"

def find_intra_ckpt(encoder, subject):
    import re
    for csv_ in glob.glob(f"outputs/retrieval/{encoder}/{subject}/**/*.csv", recursive=True):
        if re.match(rf"{encoder}_intra_{subject}\.csv", os.path.basename(csv_)):
            ts = os.path.basename(os.path.dirname(csv_))
            if ts.startswith("07-13"):
                continue
            ckpt = f"models/contrast/{encoder}/{subject}/{ts}/best.pth"
            if os.path.exists(ckpt):
                return ckpt
    return None

def build_model(encoder):
    if encoder == "ATMS":
        from models.atms import ATMS
        return ATMS(joint_train=False)
    if encoder == "LaBraM_ATMS":
        from labram_encoder import LaBraM_ATMS
        return LaBraM_ATMS()
    raise ValueError(encoder)

@torch.no_grad()
def extract(model, subject):
    dummy = {'text_features': torch.zeros(DUMMY_N, 1024),
             'img_features': torch.zeros(DUMMY_N, 1024)}
    ds = EEGDataset(DATA_PATH, subjects=[subject], train=True,
                    img_dir_training=IMG_TRAIN, img_dir_test=IMG_TEST,
                    preloaded_features=dummy)
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    embs, labels = [], []
    for batch in loader:
        x = batch[0].to(DEVICE).float()
        s = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)
        try: out = model(x, s)
        except TypeError: out = model(x)
        embs.append(out.cpu().numpy()); labels.append(np.asarray(batch[1]))
    X = np.concatenate(embs); Y = np.concatenate(labels)
    keep = np.unique(Y)[:N_CONCEPTS]
    m = np.isin(Y, keep)
    return X[m], Y[m]

def fit_probe(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf

def per_concept_acc(sc, clf, X, Y):
    pred = clf.predict(sc.transform(X))
    accs = {}
    for c in np.unique(Y):
        m = Y == c
        accs[c] = float((pred[m] == c).mean())
    overall = float((pred == Y).mean())
    return overall, accs

def run_encoder(encoder, writer):
    print(f"\n===== {encoder} =====")
    rows = []
    for S in SUBJECTS:
        ckpt = find_intra_ckpt(encoder, S)
        if ckpt is None:
            print(f"  {S}: no ckpt"); continue
        model = build_model(encoder).to(DEVICE).eval()
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False), strict=False)
        emb = {B: extract(model, B) for B in SUBJECTS}
        Xs, ys = emb[S]
        # within: hold out 20% of examples per concept
        rng = np.random.RandomState(42); tr, te = [], []
        for c in np.unique(ys):
            idx = np.where(ys == c)[0]; rng.shuffle(idx); cut = max(1, int(0.8*len(idx)))
            tr += list(idx[:cut]); te += list(idx[cut:])
        tr, te = np.array(tr), np.array(te)
        sc, clf = fit_probe(Xs[tr], ys[tr])
        within, within_pc = per_concept_acc(sc, clf, Xs[te], ys[te])
        # cross: refit on ALL of S, test on each other subject
        sc2, clf2 = fit_probe(Xs, ys)
        cross_overall, cross_ranks = [], []
        for B in SUBJECTS:
            if B == S: continue
            Xb, yb = emb[B]
            ov, pc = per_concept_acc(sc2, clf2, Xb, yb)
            cross_overall.append(ov)
            # rank corr between within per-concept acc and this cross per-concept acc
            common = sorted(set(within_pc) & set(pc))
            if len(common) > 2:
                a = np.array([within_pc[c] for c in common])
                b = np.array([pc[c] for c in common])
                if a.std() > 0 and b.std() > 0:
                    cross_ranks.append(np.corrcoef(a, b)[0, 1])
        cross = float(np.mean(cross_overall))
        rankcorr = float(np.mean(cross_ranks)) if cross_ranks else float('nan')
        # metrics
        cc_ret = (cross - CHANCE) / (within - CHANCE) if within > CHANCE else float('nan')
        norm_drop = (within - cross) / (within - CHANCE) if within > CHANCE else float('nan')
        print(f"  {S}: within={within:.3f} cross={cross:.3f} | "
              f"cc_retention={cc_ret:.3f} norm_drop={norm_drop:.3f} rank_corr={rankcorr:.3f}")
        row = dict(encoder=encoder, subject=S, within=within, cross=cross,
                   cc_retention=cc_ret, norm_drop=norm_drop, rank_corr=rankcorr)
        writer.writerow(row); rows.append(row)
    if rows:
        import statistics as st
        def mean(k): 
            vals = [r[k] for r in rows if not np.isnan(r[k])]
            return float(np.mean(vals)) if vals else float('nan')
        print(f"  MEAN within={mean('within'):.3f} cross={mean('cross'):.3f} | "
              f"cc_retention={mean('cc_retention'):.3f} "
              f"norm_drop={mean('norm_drop'):.3f} rank_corr={mean('rank_corr'):.3f}")
        return {k: mean(k) for k in ['within','cross','cc_retention','norm_drop','rank_corr']}
    return None

if __name__ == "__main__":
    if not all([DATA_PATH, IMG_TRAIN, IMG_TEST]):
        raise SystemExit("Set THINGS_DATA / THINGS_IMG_TRAIN / THINGS_IMG_TEST first.")
    print(f"Transfer probe: {N_CONCEPTS}-way, chance={CHANCE:.3f}. Three metrics, all reported.")
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["encoder","subject","within","cross",
                                               "cc_retention","norm_drop","rank_corr"])
        writer.writeheader()
        a = run_encoder("ATMS", writer)
        l = run_encoder("LaBraM_ATMS", writer)
    print("\n===== SUMMARY (all three metrics, reported regardless) =====")
    if a and l:
        print(f"{'metric':16s} {'ATMS':>8s} {'LaBraM':>8s}")
        for k in ['within','cross','cc_retention','norm_drop','rank_corr']:
            print(f"{k:16s} {a[k]:8.3f} {l[k]:8.3f}")
        print(f"\nSaved per-subject results to {OUT_CSV}")
        print("Interpretation guide (committed before seeing results):")
        print("  cc_retention: higher = more above-chance signal survives transfer")
        print("  norm_drop:    lower  = less learnable signal lost crossing subjects")
        print("  rank_corr:    higher = concept structure preserved across subjects")
