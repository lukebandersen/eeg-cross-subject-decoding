import glob, os, sys, csv
import sys
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

def _pick_run(paths):
    """Deterministic choice among several runs of one fold.

    Several runs means several seeds and/or epoch budgets. Taking whichever
    glob returned first is arbitrary, and with 35 ATMS checkpoints on disk
    across three seeds that arbitrariness is not small.
    """
    return sorted(paths)[-1] if paths else None


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
    # The two originally-probed encoders keep explicit construction so the
    # published probe numbers reproduce exactly.
    if encoder == "ATMS":
        from models.atms import ATMS
        return ATMS(joint_train=False)
    if encoder == "LaBraM_ATMS":
        from labram_encoder import LaBraM_ATMS
        return LaBraM_ATMS()
    # Everything else comes from the real registry, so the probe covers whatever
    # the panel covers without a per-encoder branch here.
    try:
        from eeg_encoders import build_encoder
    except ImportError as _e:
        raise ValueError(
            encoder + " is not special-cased and eeg_encoders could not be "
            "imported (" + str(_e) + "). Run from the repo root.")
    return build_encoder(encoder)

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
        # Encoders from the command line; no args reproduces the original run.
        encoders = sys.argv[1:] or ["ATMS", "LaBraM_ATMS"]
        results = {}
        for _enc in encoders:
            try:
                results[_enc] = run_encoder(_enc, writer)
            except Exception as _exc:
                # One bad encoder must not lose the others: the CSV is written
                # incrementally, so keep going and report at the end.
                print("  !! " + _enc + " FAILED: " +
                      type(_exc).__name__ + ": " + str(_exc))
                results[_enc] = None
        a = results.get("ATMS")
        l = results.get("LaBraM_ATMS")
    print("\n===== SUMMARY (all three metrics, reported regardless) =====")
    ok = {e: r for e, r in results.items() if r}
    if ok:
        w = max(12, max(len(e) for e in ok) + 1)
        print(f"{'metric':16s}" + "".join(f"{e:>{w}s}" for e in ok))
        for k in ['within','cross','cc_retention','norm_drop','rank_corr']:
            print(f"{k:16s}" + "".join(f"{ok[e][k]:{w}.3f}" for e in ok))
        print(f"\nchance = {CHANCE:.3f}  ({N_CONCEPTS}-way)")
        print("Read cc_retention: raw within-subject accuracy pointed the wrong "
              "way in both retrieval and classification.")
    _failed = [e for e, r in results.items() if not r]
    if _failed:
        print("\nno rows produced for: " + ", ".join(_failed))
        print(f"\nSaved per-subject results to {OUT_CSV}")
        print("Interpretation guide (committed before seeing results):")
        print("  cc_retention: higher = more above-chance signal survives transfer")
        print("  norm_drop:    lower  = less learnable signal lost crossing subjects")
        print("  rank_corr:    higher = concept structure preserved across subjects")
