#!/usr/bin/env python
"""Aggregate SEED 43 only (07-06/07-07 timestamps); compare degradation to seed 42."""
import glob, os, csv, re
import numpy as np

BASE = os.path.expanduser("~/Desktop/EEG_Image_decode-develop/outputs/retrieval")
ENCS = ["ATMS","LaBraM_ATMS"]

def best_v200(p):
    rows=list(csv.DictReader(open(p)))
    b=min(rows,key=lambda r:float(r["val_loss"]))
    return float(b["test_accuracy"])  # v200 top-1

def collect(mode):
    out={e:{} for e in ENCS}
    for enc in ENCS:
        # seed-43 runs have 07-06 or 07-07 timestamp folders
        paths=[p for p in glob.glob(f"{BASE}/{enc}/sub-*/*/{enc}_{mode}_sub-*.csv")
               if ("07-06" in p or "07-07" in p)]
        for p in paths:
            sub=re.search(r"sub-\d+",p).group()
            ts=os.path.basename(os.path.dirname(p))
            out[enc].setdefault(sub,[]).append((ts,p))
        for sub,lst in out[enc].items():
            lst.sort(); out[enc][sub]=best_v200(lst[-1][1])
    return out

intra=collect("intra"); loso=collect("loso")

print("="*60)
print("SEED 43 — v200 top-1")
print("="*60)
for enc in ENCS:
    name="LaBraM" if enc=="LaBraM_ATMS" else enc
    subs=sorted(set(intra[enc])&set(loso[enc]))
    ai=np.mean([intra[enc][s] for s in subs])
    al=np.mean([loso[enc][s] for s in subs])
    print(f"{name:8s}: intra={ai:.3f}  loso={al:.3f}  retention={al/ai:.0%}  drop={ai-al:.3f}  (n={len(subs)})")

# paired degradation test
common=sorted(set(intra['ATMS'])&set(loso['ATMS'])&set(intra['LaBraM_ATMS'])&set(loso['LaBraM_ATMS']))
a_drop=[intra['ATMS'][s]-loso['ATMS'][s] for s in common]
l_drop=[intra['LaBraM_ATMS'][s]-loso['LaBraM_ATMS'][s] for s in common]
print("-"*60)
print(f"ATMS mean drop = {np.mean(a_drop):.3f} | LaBraM mean drop = {np.mean(l_drop):.3f}")
try:
    from scipy import stats
    t,p=stats.ttest_rel(a_drop,l_drop)
    w,pw=stats.wilcoxon(a_drop,l_drop)
    n_atms_worse=sum(1 for a,l in zip(a_drop,l_drop) if a>l)
    print(f"paired t={t:.2f}, p={p:.4f} | Wilcoxon W={w:.1f}, p={pw:.4f} | n={len(common)}")
    print(f"ATMS degrades more in {n_atms_worse}/{len(common)} subjects")
except ImportError:
    print("(pip install scipy for the paired test)")

print("="*60)
print("SEED 42 (for comparison): ATMS drop 0.207 ret 37% | LaBraM drop 0.014 ret 94% | t=11.0 p<0.0001")
