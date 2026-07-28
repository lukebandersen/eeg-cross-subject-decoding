#!/usr/bin/env python
"""Aggregate intra + LOSO for both encoders; per-subject degradation + paired test."""
import glob, os, csv, re
import numpy as np

BASE = os.path.expanduser("~/Desktop/EEG_Image_decode-develop/outputs/retrieval")
METRICS = ["v2_acc","v4_acc","v10_acc","v50_acc","v100_acc","test_accuracy"]
NICE = {"v2_acc":"v2","v4_acc":"v4","v10_acc":"v10","v50_acc":"v50","v100_acc":"v100","test_accuracy":"v200t1"}
ENCS = ["ATMS","LaBraM_ATMS"]

def best_row(p):
    rows=list(csv.DictReader(open(p)))
    b=min(rows,key=lambda r:float(r["val_loss"]))
    return {m:float(b[m]) for m in METRICS}

def collect(mode):
    out={e:{} for e in ENCS}
    for enc in ENCS:
        paths=glob.glob(f"{BASE}/{enc}/sub-*/*/{enc}_{mode}_sub-*.csv")
        by_sub={}
        for p in paths:
            sub=re.search(r"sub-\d+",p).group()
            ts=os.path.basename(os.path.dirname(p))
            by_sub.setdefault(sub,[]).append((ts,p))
        for sub,lst in by_sub.items():
            lst.sort()
            out[enc][sub]=best_row(lst[-1][1])
    return out

intra=collect("intra"); loso=collect("loso")

def show(title,data):
    for enc in ENCS:
        name="LaBraM" if enc=="LaBraM_ATMS" else enc
        subs=sorted(data[enc])
        print(f"\n{'='*72}\n{title} — {name} — {len(subs)} folds\n{'='*72}")
        print(f"{'fold':10s} "+"  ".join(f"{NICE[m]:>8s}" for m in METRICS))
        for s in subs:
            print(f"{s:10s} "+"  ".join(f"{data[enc][s][m]:8.3f}" for m in METRICS))
        print("-"*72)
        mean={m:np.mean([data[enc][s][m] for s in subs]) for m in METRICS}
        std ={m:np.std ([data[enc][s][m] for s in subs]) for m in METRICS}
        print(f"{'MEAN':10s} "+"  ".join(f"{mean[m]:8.3f}" for m in METRICS))
        print(f"{'STD':10s} "+"  ".join(f"{std[m]:8.3f}" for m in METRICS))

show("INTRA",intra); show("LOSO",loso)

print(f"\n{'='*72}\nPER-SUBJECT DEGRADATION (v200-top1)\n{'='*72}")
print(f"{'subj':8s} {'ATMS_in':>8s} {'ATMS_lo':>8s} {'ATMS_drop':>10s} {'ATMS_ret':>8s} | "
      f"{'LaB_in':>7s} {'LaB_lo':>7s} {'LaB_drop':>9s} {'LaB_ret':>7s}")
common=sorted(set(intra['ATMS'])&set(loso['ATMS'])&set(intra['LaBraM_ATMS'])&set(loso['LaBraM_ATMS']))
a_drop=[]; l_drop=[]; a_ret=[]; l_ret=[]
for s in common:
    ai=intra['ATMS'][s]['test_accuracy']; al=loso['ATMS'][s]['test_accuracy']
    li=intra['LaBraM_ATMS'][s]['test_accuracy']; ll=loso['LaBraM_ATMS'][s]['test_accuracy']
    ad=ai-al; ld=li-ll; ar=al/ai if ai else np.nan; lr=ll/li if li else np.nan
    a_drop.append(ad); l_drop.append(ld); a_ret.append(ar); l_ret.append(lr)
    print(f"{s:8s} {ai:8.3f} {al:8.3f} {ad:10.3f} {ar:8.0%} | {li:7.3f} {ll:7.3f} {ld:9.3f} {lr:7.0%}")
print("-"*72)
print(f"{'MEAN':8s} {np.mean([intra['ATMS'][s]['test_accuracy'] for s in common]):8.3f} "
      f"{np.mean([loso['ATMS'][s]['test_accuracy'] for s in common]):8.3f} "
      f"{np.mean(a_drop):10.3f} {np.nanmean(a_ret):8.0%} | "
      f"{np.mean([intra['LaBraM_ATMS'][s]['test_accuracy'] for s in common]):7.3f} "
      f"{np.mean([loso['LaBraM_ATMS'][s]['test_accuracy'] for s in common]):7.3f} "
      f"{np.mean(l_drop):9.3f} {np.nanmean(l_ret):7.0%}")

# paired test: is ATMS degradation > LaBraM degradation, per subject?
try:
    from scipy import stats
    t,p_t = stats.ttest_rel(a_drop, l_drop)
    w,p_w = stats.wilcoxon(a_drop, l_drop)
    print(f"\n{'='*72}\nPAIRED TEST — does ATMS degrade MORE than LaBraM? (per-subject drop)\n{'='*72}")
    print(f"  mean ATMS drop = {np.mean(a_drop):.3f} | mean LaBraM drop = {np.mean(l_drop):.3f}")
    print(f"  paired t-test:  t={t:.3f}, p={p_t:.4f}")
    print(f"  Wilcoxon:       W={w:.1f}, p={p_w:.4f}")
    print(f"  n = {len(common)} subjects")
    print(f"  => {'SIGNIFICANT (p<0.05): ATMS degrades more' if p_t<0.05 else 'NOT significant at p<0.05'}")
except ImportError:
    print("\n(scipy not installed — run: pip install scipy — for the paired significance test)")
