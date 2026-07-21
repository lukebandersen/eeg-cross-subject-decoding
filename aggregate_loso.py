#!/usr/bin/env python
"""Aggregate LOSO sweep results: per encoder, mean best-val metrics across folds."""
import glob, os, csv
import numpy as np

BASE = os.path.expanduser("~/Desktop/EEG_Image_decode-develop/outputs/retrieval")
METRICS = ["v2_acc","v4_acc","v10_acc","v50_acc","v100_acc","test_accuracy"]
NICE    = {"v2_acc":"v2","v4_acc":"v4","v10_acc":"v10","v50_acc":"v50",
           "v100_acc":"v100","test_accuracy":"v200-t1"}

def best_row(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    best = min(rows, key=lambda r: float(r["val_loss"]))
    return {m: float(best[m]) for m in METRICS}

for enc in ["ATMS", "LaBraM_ATMS"]:
    paths = glob.glob(f"{BASE}/{enc}/sub-*/*/{enc}_loso_sub-*.csv")
    if not paths:
        print(f"\n{enc}: no fold CSVs found"); continue
    per_fold = {}
    for p in paths:
        sub = [x for x in p.split(os.sep) if x.startswith("sub-")][0]
        per_fold[sub] = best_row(p)
    subs = sorted(per_fold)
    print(f"\n{'='*70}\n{enc}  —  {len(subs)} folds\n{'='*70}")
    print(f"{'fold':10s} " + "  ".join(f"{NICE[m]:>8s}" for m in METRICS))
    for sub in subs:
        print(f"{sub:10s} " + "  ".join(f"{per_fold[sub][m]:8.3f}" for m in METRICS))
    print("-"*70)
    means = {m: np.mean([per_fold[s][m] for s in subs]) for m in METRICS}
    stds  = {m: np.std ([per_fold[s][m] for s in subs]) for m in METRICS}
    print(f"{'MEAN':10s} " + "  ".join(f"{means[m]:8.3f}" for m in METRICS))
    print(f"{'STD':10s} " + "  ".join(f"{stds[m]:8.3f}" for m in METRICS))
