#!/usr/bin/env python
"""
peek_panel.py -- preliminary panel retention table from whatever has finished.

Scans outputs/retrieval/**/*.csv, pulls the FINAL-row v200 top-1 for each
(encoder, mode, subject) fold, averages folds per encoder, and reports

    retention = mean(LOSO) / mean(intra)

Reports partial sweeps honestly: every row states how many folds it is based on,
and retention is only computed when BOTH modes have at least one fold. Numbers
from incomplete sweeps are marked so they are not mistaken for final results.

Usage (from repo root):
    python peek_panel.py
    python peek_panel.py --min-folds 5      # only show encoders with >=5 folds
    python peek_panel.py --csv panel.csv    # also write a tidy CSV
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
import sys

try:
    from run_config import load_run_config
except ImportError:          # run_config.py not alongside: legacy mode
    def load_run_config(_d):
        return None

# Which column holds v200 top-1. The results CSV header is:
#   epoch,train_loss,val_loss,test_loss,test_accuracy,v2_acc,...
# test_accuracy IS the 200-way top-1 (chance = 1/200 = 0.005).
ACC_COL = "test_accuracy"

CLASS = {
    "ATMS": "specialized",
    "EEGNetv4_Encoder": "specialized",
    "EEGConformer_Encoder": "specialized",
    "ShallowFBCSPNet_Encoder": "specialized",
    "LaBraM_ATMS": "foundation",
    "CBraMod_Encoder": "foundation",
}
PARAMS = {
    "ATMS": "3.20M",
    "EEGNetv4_Encoder": "1.19M",
    "EEGConformer_Encoder": "0.64M",
    "ShallowFBCSPNet_Encoder": "0.89M",
    "LaBraM_ATMS": "6.45M",
    "CBraMod_Encoder": "6.97M",
}

# Fields that must AGREE across encoders for the table to mean anything.
# Anything here that differs makes two runs incomparable, and the script says so
# rather than averaging them. Extend this list when a new axis appears; the
# point of dumping the whole config is that you can, without re-running anything.
COMPARABILITY_FIELDS = ["seed", "epochs", "lr", "batch_size", "n_chans",
                        "n_times", "val_ratio", "dataset"]

# filename pattern: {encoder}_{mode}_{subject}.csv
FNAME = re.compile(r"^(?P<enc>.+)_(?P<mode>intra|loso|joint)_(?P<sub>sub-\d+)\.csv$")


def best_val_acc(path):
    """v200 top-1 at the BEST-VALIDATION-LOSS epoch -- matching what
    train_unified.py reports ("Loading best model [EMA] ... best val loss").

    NOT max(test_accuracy): selecting the epoch by the test metric is test-set
    leakage and would inflate every number. We select on val_loss and then read
    off that epoch's accuracy, which is the same model the pipeline saves.

    Returns None for runs poisoned by the val_loss==0.0 bug (they early-stop on a
    fake zero and sit at chance).
    """
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return None
    if not rows:
        return None

    best_loss, best_acc, saw_valid_loss = None, None, False
    for row in rows:
        acc_raw = row.get(ACC_COL)
        if acc_raw in (None, ""):
            continue
        try:
            acc = float(acc_raw)
        except ValueError:
            continue
        try:
            vloss = float(row.get("val_loss", "") or "nan")
        except ValueError:
            vloss = float("nan")
        if vloss == 0.0:
            # the val-ratio bug: whole run is untrustworthy
            return None
        if vloss == vloss:  # not NaN
            saw_valid_loss = True
            if best_loss is None or vloss < best_loss:
                best_loss, best_acc = vloss, acc

    if saw_valid_loss and best_acc is not None:
        return best_acc

    # no usable val_loss column: fall back to the final parseable row
    for row in reversed(rows):
        try:
            return float(row[ACC_COL])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def collect(roots, timestamps=None):
    """-> (data, ambiguous)

    data      : {(encoder, mode): {subject: (acc, timestamp)}}
    ambiguous : {(encoder, mode, subject): [timestamps...]} for folds with >1 run

    Deliberately does NOT resolve duplicates. Several runs of one fold means
    several seeds and/or epoch budgets on disk, and picking by file mtime
    silently mixes them: it produced a plausible-looking table in which the
    locked pair came from seed 44 while the panel encoders came from seed 42.
    A wrong number that looks right is worse than a crash, so the caller must
    resolve the ambiguity explicitly.
    """
    data = {}
    runs = {}
    dupes = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
            m = FNAME.match(os.path.basename(path))
            if not m:
                continue
            enc, mode, sub = m["enc"], m["mode"], m["sub"]
            ts = os.path.basename(os.path.dirname(path))
            if timestamps and not any(t in ts for t in timestamps):
                continue
            acc = best_val_acc(path)
            if acc is None:
                continue
            key = (enc, mode)
            cfg = load_run_config(os.path.dirname(path))
            runs.setdefault((enc, mode, sub), []).append((ts, acc, cfg))
    ambiguous = {}
    for (enc, mode, sub), lst in runs.items():
        if len(lst) > 1:
            ambiguous[(enc, mode, sub)] = sorted(t for t, _a, _c in lst)
        else:
            ts, acc, cfg = lst[0]
            data.setdefault((enc, mode), {})[sub] = (acc, ts, cfg)
    return data, ambiguous


def _collect_latest(roots, timestamps=None):
    """Old mtime-wins behaviour, reachable only via --allow-latest."""
    data, latest = {}, {}
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
            m = FNAME.match(os.path.basename(path))
            if not m:
                continue
            enc, mode, sub = m["enc"], m["mode"], m["sub"]
            ts = os.path.basename(os.path.dirname(path))
            if timestamps and not any(t in ts for t in timestamps):
                continue
            acc = best_val_acc(path)
            if acc is None:
                continue
            mt = os.path.getmtime(path)
            if latest.get((enc, mode, sub), -1) > mt:
                continue
            latest[(enc, mode, sub)] = mt
            data.setdefault((enc, mode), {})[sub] = (
                acc, ts, load_run_config(os.path.dirname(path)))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["outputs/retrieval", "results/retrieval"],
                    help="where the per-run CSVs live")
    ap.add_argument("--min-folds", type=int, default=1)
    ap.add_argument("--csv", default=None, help="also write a tidy CSV here")
    ap.add_argument("--timestamps", nargs="+", default=None, metavar="SUBSTR",
                    help="only use run directories whose timestamp contains one "
                         "of these substrings, e.g. --timestamps 07-02 07-06 "
                         "07-08. This is how you pin a seed: the seed is NOT "
                         "recorded in the CSV or the path, so the run directory "
                         "is the only handle on which sweep a fold came from.")
    ap.add_argument("--no-config-check", action="store_true",
                    help="skip the run_config.json comparability gate. Only with "
                         "a stated reason; the gate exists because seed and "
                         "epoch-budget mismatches were both caught by memory.")
    ap.add_argument("--allow-latest", action="store_true",
                    help="UNSAFE. Resolve duplicate folds by taking the most "
                         "recent run. Mixes seeds and epoch budgets across "
                         "encoders without saying so. Never use for a table "
                         "that will be compared against published numbers.")
    args = ap.parse_args()

    data, ambiguous = collect(args.roots, args.timestamps)

    if ambiguous and not args.allow_latest:
        print("=" * 78)
        print(" REFUSING TO REPORT: some folds have more than one run on disk.")
        print("=" * 78)
        print(" Several runs of one fold means several seeds and/or epoch")
        print(" budgets. Choosing among them by file mtime does not fail, it")
        print(" produces a plausible table that silently mixes them, which is")
        print(" how the locked pair once came from seed 44 while the panel")
        print(" encoders came from seed 42.\n")
        by_enc = {}
        for (enc, mode, sub), tss in ambiguous.items():
            by_enc.setdefault(enc, set()).update(tss)
        for enc in sorted(by_enc):
            n = sum(1 for k in ambiguous if k[0] == enc)
            print(f"   {enc:<26} {n:>3} ambiguous fold(s)")
            print(f"   {'':<26} runs: {', '.join(sorted(by_enc[enc]))}")
        print("\n Resolve it explicitly, e.g.:")
        print("   python peek_panel.py --timestamps " +
              " ".join(sorted(list(by_enc[sorted(by_enc)[0]]))[:3]))
        print("\n (--allow-latest reinstates the old mtime behaviour, but the"
              "\n  resulting table must not be compared against the paper.)")
        sys.exit(2)

    if ambiguous and args.allow_latest:
        print(f" WARNING: {len(ambiguous)} ambiguous fold(s) resolved by mtime."
              f" Seeds/budgets may be mixed. Not comparable to published numbers.\n")
        # re-collect the old way, explicitly opted into
        data = _collect_latest(args.roots, args.timestamps)

    # ---- comparability gate --------------------------------------------
    # Duplicate files were only ONE way two runs can be incomparable. Seed was
    # another. Epoch budget was a third, caught by memory alone. Rather than
    # adding a check per axis as each bug surfaces, compare the recorded config
    # across every fold in the table and refuse when a field that matters differs.
    if data and not args.no_config_check:
        seen_fields = {}
        unconfigured = set()
        for (enc, _mode), folds in data.items():
            for _sub, (_acc, _ts, cfg) in folds.items():
                if cfg is None:
                    unconfigured.add(enc)
                    continue
                for fld in COMPARABILITY_FIELDS:
                    if fld in cfg:
                        seen_fields.setdefault(fld, {}).setdefault(
                            str(cfg[fld]), set()).add(enc)

        conflicts = {f: v for f, v in seen_fields.items() if len(v) > 1}
        if conflicts:
            print("=" * 78)
            print(" REFUSING TO REPORT: runs in this table were not produced")
            print(" under the same configuration.")
            print("=" * 78)
            for fld, vals in sorted(conflicts.items()):
                print(f"\n   {fld}:")
                for val, encs in sorted(vals.items()):
                    print(f"     {val:<14} <- {', '.join(sorted(encs))}")
            print("\n Averaging across these produces a table whose rows are not")
            print(" comparable to each other. Restrict the set (--timestamps),")
            print(" re-run the odd ones out, or pass --no-config-check if you")
            print(" have a specific reason and will state it wherever the")
            print(" numbers appear.")
            sys.exit(3)

        if unconfigured:
            print(f" NOTE: no run_config.json for: {', '.join(sorted(unconfigured))}")
            print(" These predate config recording, so their seed and epoch budget")
            print(" cannot be verified from the run. Treat comparisons against them")
            print(" as unchecked. (See MANIFEST.md if you have backfilled them.)\n")

    if not data:
        print("No result CSVs found under: " + ", ".join(args.roots))
        print("Run from the repo root, or pass --roots.")
        return

    encoders = sorted({e for (e, _m) in data}, key=lambda e: (CLASS.get(e, "z"), e))

    print("=" * 86)
    print(" PRELIMINARY PANEL RETENTION  (v200 top-1 at best-val-loss epoch; chance = 0.005)")
    print(" retention = mean(LOSO) / mean(intra).  PARTIAL sweeps are marked *.")
    print("=" * 86)
    print(f" {'ENCODER':<26}{'CLASS':<13}{'INTRA':>14}{'LOSO':>14}{'RETENTION':>13}")
    print("-" * 86)

    rows_out = []
    provenance = {}
    for enc in encoders:
        intra = data.get((enc, "intra"), {})
        loso = data.get((enc, "loso"), {})
        if max(len(intra), len(loso)) < args.min_folds:
            continue

        def fmt(d):
            if not d:
                return "--", None
            mean = st.mean(a for a, _t, _c in d.values())
            mark = "" if len(d) == 10 else "*"
            return f"{mean:.4f} ({len(d)}){mark}", mean

        i_txt, i_mean = fmt(intra)
        l_txt, l_mean = fmt(loso)

        if intra and loso and i_mean:
            ret = l_mean / i_mean
            partial = "*" if (len(intra) < 10 or len(loso) < 10) else ""
            r_txt = f"{ret*100:.1f}%{partial}"
        else:
            ret, r_txt = None, "--"

        cls = CLASS.get(enc, "?")
        print(f" {enc:<26}{cls:<13}{i_txt:>14}{l_txt:>14}{r_txt:>13}")
        used = sorted({t for _a, t, _c in list(intra.values()) + list(loso.values())})
        provenance[enc] = used
        rows_out.append({
            "encoder": enc, "class": cls, "params": PARAMS.get(enc, ""),
            "intra_mean": f"{i_mean:.6f}" if intra else "",
            "intra_folds": len(intra),
            "loso_mean": f"{l_mean:.6f}" if loso else "",
            "loso_folds": len(loso),
            "retention": f"{ret:.4f}" if ret else "",
            "complete": "yes" if (len(intra) == 10 and len(loso) == 10) else "no",
        })

    print("-" * 86)
    print(" (n) = folds averaged.  * = fewer than 10 folds, treat as preliminary.")
    print()
    print(" RUN DIRECTORIES USED (check these match across encoders):")
    for enc in sorted(provenance):
        runs = provenance[enc]
        shown = ", ".join(runs[:4]) + (f", +{len(runs)-4} more" if len(runs) > 4 else "")
        print(f"   {enc:<26} {shown}")

    # Two-cluster read, only using encoders with a computable retention
    done = [r for r in rows_out if r["retention"]]
    spec = [float(r["retention"]) for r in done if r["class"] == "specialized"]
    found = [float(r["retention"]) for r in done if r["class"] == "foundation"]
    if spec and found:
        print()
        print(" TWO-CLUSTER READ (preliminary)")
        print(f"   specialized: {', '.join(f'{v*100:.1f}%' for v in sorted(spec))}"
              f"   (mean {st.mean(spec)*100:.1f}%)")
        print(f"   foundation : {', '.join(f'{v*100:.1f}%' for v in sorted(found))}"
              f"   (mean {st.mean(found)*100:.1f}%)")
        gap = min(found) - max(spec)
        if gap > 0:
            print(f"   -> clean separation so far: gap of {gap*100:.1f} points "
                  f"between the worst foundation and the best specialist.")
        else:
            print(f"   -> clusters OVERLAP by {abs(gap)*100:.1f} points. "
                  f"Report honestly; the split is not clean (yet).")
    elif spec or found:
        print("\n Only one class has a computable retention so far; "
              "no cluster comparison yet.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n wrote {args.csv}")


if __name__ == "__main__":
    main()
