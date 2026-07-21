#!/usr/bin/env python
import glob, os, csv, re
import numpy as np
BASE = os.path.expanduser("~/Desktop/EEG_Image_decode-develop/outputs/retrieval")
ENCS = ["ATMS","LaBraM_ATMS"]
def best_v200(p):
    rows=list(csv.DictReader(open(p)))
    b=min(rows,key=lambda r:float(r["val_loss"]))
    return float(b["test_accuracy"])
def collect(mode):
    out={e:{} for e in ENCS}
    for enc in ENCS:
        paths=[p for p in glob.glob(f"{BASE}/{enc}/sub-*/*/{enc}_{mode}_sub-*.csv")
               if ("07-08" in p or "07-09" in p)]
        for p in paths:
            sub=re.search(r"sub-\d+",p).group()
            ts=os.path.basename(os.path.dirname(p))
            out[enc].setdefault(sub,[]).append((ts,p))
        for sub,lst in out[enc].items():
            lst.sort(); out[enc][sub]=best_v200(lst[-1][1])
    return out
intra=collect("intra"); loso=collect("loso")
print("="*56); print("SEED 44 — v200 top-1"); print("="*56)
for enc in ENCS:
    name="LaBraM" if enc=="LaBraM_ATMS" else enc
    subs=sorted(set(intra[enc])&set(loso[enc]))
    if not subs: print(f"{name}: no data"); continue
    ai=np.mean([intra[enc][s] for s in subs]); al=np.mean([loso[enc][s] for s in subs])
    print(f"{name:8s}: intra={ai:.3f} loso={al:.3f} retention={al/ai:.0%} drop={ai-al:.3f} (n={len(subs)})")
common=sorted(set(intra['ATMS'])&set(loso['ATMS'])&set(intra['LaBraM_ATMS'])&set(loso['LaBraM_ATMS']))
if common:
    a_drop=[intra['ATMS'][s]-loso['ATMS'][s] for s in common]
    l_drop=[intra['LaBraM_ATMS'][s]-loso['LaBraM_ATMS'][s] for s in common]
    print("-"*56); print(f"ATMS drop={np.mean(a_drop):.3f} | LaBraM drop={np.mean(l_drop):.3f}")
    try:
        from scipy import stats
        t,p=stats.ttest_rel(a_drop,l_drop); w,pw=stats.wilcoxon(a_drop,l_drop)
        nw=sum(1 for a,l in zip(a_drop,l_drop) if a>l)
        print(f"paired t={t:.2f}, p={p:.4f} | Wilcoxon W={w:.1f} | ATMS worse in {nw}/{len(common)}")
    except ImportError: print("(scipy needed for test)")
print("="*56)
print("SEED 42: ATMS 37% ret, drop .207, t=11.0 | LaBraM 94% ret, drop .014")
print("SEED 43: ATMS 33% ret, drop .221, t=13.67 | LaBraM 92% ret, drop .009")
