#!/usr/bin/env python
"""
probe_loso_classify.py -- LOSO-trained classification readout (matrix rows D02/D03).

HOW THIS DIFFERS FROM probe_transfer.py (D01, already run):

  probe_transfer.py (D01):  freeze ONE subject's intra-encoder, fit a probe on
                            THAT subject, test the same probe on other subjects.
                            Measures: does one subject's representation transfer?
                            Result: cc_retention 0.154 (ATMS) vs 0.234 (LaBraM),
                            p=0.012; rank_corr null (0.083 vs 0.084).

  THIS script (D02/D03):    for each held-out subject H, embed the OTHER 9
                            subjects with a frozen encoder, train ONE classifier
                            on all 9 pooled, and test on H.
                            Measures: does a classifier trained ACROSS subjects
                            generalize to an unseen subject?

The second mirrors the retrieval LOSO protocol far more directly, which makes it
the cleaner cross-task comparison for the paper.

METRICS: identical to D01 so the two readouts are directly comparable:
    chance-corrected retention  R = (cross - chance) / (within - chance)
    normalized drop             D = 1 - R
    rank transfer             rho = corr(per-concept within, per-concept cross)

DESIGN NOTES (inherited deliberately from the corrected D01 design):
  - Uses the TRAINING split. The THINGS test split has ONE image per concept and
    cannot support concept classification. This was a real bug caught in D01.
  - Caps at N_CONCEPTS (default 100); chance = 1/N_CONCEPTS.
  - Linear probe only: we measure the representation, not classifier capacity.

RUN (Git Bash, from repo root, same env vars probe_transfer.py uses):
    export THINGS_DATA="C:/.../Preprocessed_data_250Hz"
    export THINGS_IMG_TRAIN="C:/.../training_images"
    export THINGS_IMG_TEST="C:/.../test_images"
    python probe_loso_classify.py --encoder ATMS        --out probe_loso_ATMS.csv
    python probe_loso_classify.py --encoder LaBraM_ATMS --out probe_loso_LaBraM.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Reuse the VERIFIED loading path from probe_transfer.py. One embedding path,
# not two: if that loader is correct this one is too, and any future fix is
# inherited by both probes.
# ---------------------------------------------------------------------------
try:
    from probe_transfer import find_intra_ckpt, build_model, DEVICE
except ImportError as e:
    raise ImportError(
        "probe_loso_classify.py must sit NEXT TO probe_transfer.py (repo root) so "
        f"it can reuse the verified checkpoint/encoder loader. Original error: {e}"
    ) from e

from eegdatasets import EEGDataset

N_CONCEPTS = 100
CHANCE = 1.0 / N_CONCEPTS

DATA_PATH = os.environ.get("THINGS_DATA")
IMG_TRAIN = os.environ.get("THINGS_IMG_TRAIN")
IMG_TEST = os.environ.get("THINGS_IMG_TEST")


def load_checkpoint_verified(model, ckpt_path, min_match=0.80, verbose=True):
    """
    Load a checkpoint AND PROVE IT LOADED.

    probe_transfer.py uses load_state_dict(sd, strict=False), which silently
    ignores keys that do not match. If a layer is ever renamed, the encoder runs
    on RANDOM weights and still produces plausible-looking numbers. That is the
    exact failure this project has been bitten by (the commented-out encoder
    load). We count matched keys and abort rather than score on random weights.
    """
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model_keys = set(model.state_dict().keys())
    matched = model_keys & set(sd.keys())
    frac = len(matched) / max(len(model_keys), 1)

    result = model.load_state_dict(sd, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))

    if verbose:
        print(f"    ckpt: matched {len(matched)}/{len(model_keys)} keys ({frac:.0%}); "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    if frac < min_match:
        raise RuntimeError(
            f"CHECKPOINT FAILED VERIFICATION: only {frac:.0%} of model keys matched in "
            f"{ckpt_path} (need >= {min_match:.0%}). Refusing to run on partially-random "
            f"weights. Missing (first 5): {missing[:5]}"
        )
    return model


@torch.no_grad()
def embed_subject(model, subject, n_concepts=N_CONCEPTS):
    """
    Frozen-encoder embeddings for one subject's TRAINING trials.
    Returns (X, y): X (N, 1024) float32; y (N,) concept ids in [0, n_concepts).
    """
    ds = EEGDataset(DATA_PATH, subjects=[subject], train=True,
                    img_dir_training=IMG_TRAIN, img_dir_test=IMG_TEST)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    embs, labs = [], []
    for batch in loader:
        x = batch[0].to(DEVICE).float()
        y = np.asarray(batch[1])
        s = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)
        try:
            out = model(x, s)          # encoders that take subject ids
        except TypeError:
            out = model(x)             # encoders that do not
        embs.append(out.cpu().numpy())
        labs.append(y)
    X = np.concatenate(embs)
    y = np.concatenate(labs)
    keep = y < n_concepts
    if keep.sum() == 0:
        raise RuntimeError(
            f"{subject}: no trials with concept id < {n_concepts}. Label space "
            f"looks unexpected (min={y.min()}, max={y.max()})."
        )
    return X[keep], y[keep]


def fit_probe(train_X, train_y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(train_X)
    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
    clf.fit(sc.transform(train_X), train_y)
    return sc, clf


def score(sc, clf, X, y):
    return float(clf.score(sc.transform(X), y))


def per_concept_acc(sc, clf, X, y):
    pred = clf.predict(sc.transform(X))
    return {int(c): float((pred[y == c] == c).mean()) for c in np.unique(y)}


def rank_transfer(within_map, cross_map):
    from scipy.stats import pearsonr
    common = sorted(set(within_map) & set(cross_map))
    if len(common) < 3:
        return float("nan")
    a = [within_map[c] for c in common]
    b = [cross_map[c] for c in common]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    try:
        return float(pearsonr(a, b)[0])
    except Exception:
        return float("nan")


def run(encoder, subjects, seed, out_csv):
    print(f"\n===== LOSO-CLASSIFY: {encoder} =====")
    rows = []
    for held in subjects:
        ckpt = find_intra_ckpt(encoder, held)
        if ckpt is None:
            print(f"  {held}: no intra checkpoint -> skip")
            continue
        print(f"  {held}: {ckpt}")
        model = build_model(encoder).to(DEVICE).eval()
        load_checkpoint_verified(model, ckpt)

        emb = {s: embed_subject(model, s) for s in subjects}

        # cross: train on the OTHER 9 pooled, test on held-out
        Xo = np.concatenate([emb[s][0] for s in subjects if s != held])
        yo = np.concatenate([emb[s][1] for s in subjects if s != held])
        sc_c, clf_c = fit_probe(Xo, yo)
        Xh, yh = emb[held]
        cross = score(sc_c, clf_c, Xh, yh)

        # within: split the held-out subject's own trials
        n = len(yh)
        idx = np.random.RandomState(seed).permutation(n)
        cut = int(0.8 * n)
        tr, te = idx[:cut], idx[cut:]
        sc_w, clf_w = fit_probe(Xh[tr], yh[tr])
        within = score(sc_w, clf_w, Xh[te], yh[te])

        R = (cross - CHANCE) / (within - CHANCE) if within > CHANCE else float("nan")
        D = 1.0 - R if R == R else float("nan")
        rho = rank_transfer(per_concept_acc(sc_w, clf_w, Xh[te], yh[te]),
                            per_concept_acc(sc_c, clf_c, Xh, yh))

        print(f"    within={within:.4f} cross={cross:.4f} | "
              f"cc_retention={R:.3f} norm_drop={D:.3f} rank_corr={rho:.3f}")
        rows.append((encoder, held, within, cross, R, D, rho))

        del emb, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        print("  no rows produced (no checkpoints found).")
        return

    d = os.path.dirname(os.path.abspath(out_csv))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("encoder,held_subject,within,cross,cc_retention,norm_drop,rank_corr\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    arr = np.array([[r[2], r[3], r[4], r[5], r[6]] for r in rows], float)
    print(f"\n  MEAN within={np.nanmean(arr[:, 0]):.4f} cross={np.nanmean(arr[:, 1]):.4f} | "
          f"cc_retention={np.nanmean(arr[:, 2]):.3f} "
          f"norm_drop={np.nanmean(arr[:, 3]):.3f} rank_corr={np.nanmean(arr[:, 4]):.3f}")
    print(f"  saved -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="ATMS | LaBraM_ATMS | CBraMod_Encoder")
    ap.add_argument("--subjects", default="1-10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="probe_loso_results.csv")
    args = ap.parse_args()

    if not all([DATA_PATH, IMG_TRAIN, IMG_TEST]):
        print("Set THINGS_DATA, THINGS_IMG_TRAIN, THINGS_IMG_TEST first.")
        print('  export THINGS_DATA="C:/.../Preprocessed_data_250Hz"')
        sys.exit(1)

    subs = []
    for part in args.subjects.split(","):
        if "-" in part:
            a, b = part.split("-")
            subs.extend(range(int(a), int(b) + 1))
        else:
            subs.append(int(part))
    subjects = [f"sub-{i:02d}" for i in subs]

    print(f"LOSO classification probe | encoder={args.encoder} | {len(subjects)} subjects | "
          f"{N_CONCEPTS} concepts (chance={CHANCE:.3f})")
    run(args.encoder, subjects, args.seed, args.out)


if __name__ == "__main__":
    main()
