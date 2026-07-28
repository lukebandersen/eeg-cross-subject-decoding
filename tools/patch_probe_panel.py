#!/usr/bin/env python
"""
patch_probe_panel.py -- extend probe_transfer.py from 2 encoders to the panel.

Additive only: the ATMS / LaBraM_ATMS path is untouched, so the probe numbers
already reported (cc_retention 0.154 vs 0.234, p=0.012) stay reproducible.

  1. build_model()    falls back to the real registry (build_encoder, the same
                      one the verification gates exercise) for any encoder it
                      does not special-case.
  2. __main__         reads encoder names from argv. No args = original two.
  3. find_intra_ckpt  deterministic (newest timestamp) and prints its choice.
                      It previously took whatever glob returned first; ATMS has
                      35 checkpoints across three seeds, so "first" was arbitrary.
  4. summary          prints N columns instead of two hard-coded ones.

Idempotent. Writes a .bak. Run from the repo root:  python patch_probe_panel.py
"""
import os
import shutil
import sys

TARGET = "probe_transfer.py"

BUILD_OLD = r'''def build_model(encoder):
    if encoder == "ATMS":
        from models.atms import ATMS
        return ATMS(joint_train=False)
    if encoder == "LaBraM_ATMS":
        from labram_encoder import LaBraM_ATMS
        return LaBraM_ATMS()
    raise ValueError(encoder)'''

BUILD_NEW = r'''def build_model(encoder):
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
    return build_encoder(encoder)'''

CKPT_OLD = r'''def find_intra_ckpt(encoder, subject):'''

CKPT_NEW = r'''def _pick_run(paths):
    """Deterministic choice among several runs of one fold.

    Several runs means several seeds and/or epoch budgets. Taking whichever
    glob returned first is arbitrary, and with 35 ATMS checkpoints on disk
    across three seeds that arbitrariness is not small.
    """
    return sorted(paths)[-1] if paths else None


def find_intra_ckpt(encoder, subject):'''

MAIN_OLD = r'''        a = run_encoder("ATMS", writer)
        l = run_encoder("LaBraM_ATMS", writer)'''

MAIN_NEW = r'''        # Encoders from the command line; no args reproduces the original run.
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
        l = results.get("LaBraM_ATMS")'''

SUMMARY_OLD = r'''    if a and l:
        print(f"{'metric':16s} {'ATMS':>8s} {'LaBraM':>8s}")
        for k in ['within','cross','cc_retention','norm_drop','rank_corr']:
            print(f"{k:16s} {a[k]:8.3f} {l[k]:8.3f}")'''

SUMMARY_NEW = r'''    ok = {e: r for e, r in results.items() if r}
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
        print("\nno rows produced for: " + ", ".join(_failed))'''


def main():
    if not os.path.exists(TARGET):
        sys.exit("ERROR: " + TARGET + " not found. Run from the repo root.")

    s = open(TARGET, encoding="utf-8").read()

    if "build_encoder" in s and "sys.argv[1:]" in s:
        print("already patched -- no change.")
        return

    missing = []
    for name, anchor in (("build_model", BUILD_OLD),
                         ("find_intra_ckpt", CKPT_OLD),
                         ("__main__ calls", MAIN_OLD),
                         ("summary block", SUMMARY_OLD)):
        if anchor not in s:
            missing.append(name)
    if missing:
        sys.exit("ERROR: could not find: " + ", ".join(missing) +
                 "\nThe file differs from what this patch expects; not touching it.")

    # Insert `import sys` after the first import line, whatever form the file
    # uses. Anchoring on a specific module name fails when imports are combined
    # on one line ("import csv, glob, os, re" contains no "import re").
    if not any(ln.strip() in ("import sys",) or ln.strip().startswith("import sys")
               for ln in s.splitlines()):
        _lines = s.splitlines(keepends=True)
        for _i, _ln in enumerate(_lines):
            if _ln.startswith("import ") or _ln.startswith("from "):
                _lines.insert(_i + 1, "import sys\n")
                break
        else:
            _lines.insert(0, "import sys\n")
        s = "".join(_lines)

    s = s.replace(BUILD_OLD, BUILD_NEW, 1)
    s = s.replace(CKPT_OLD, CKPT_NEW, 1)
    s = s.replace(MAIN_OLD, MAIN_NEW, 1)
    s = s.replace(SUMMARY_OLD, SUMMARY_NEW, 1)

    shutil.copy(TARGET, TARGET + ".bak_panel")
    open(TARGET, "w", encoding="utf-8").write(s)

    print("patched " + TARGET + "  (backup: " + TARGET + ".bak_panel)")
    print("")
    print("Original behaviour (unchanged):")
    print("    python probe_transfer.py")
    print("")
    print("Panel:")
    print("    python probe_transfer.py EEGNetv4_Encoder EEGConformer_Encoder CBraMod_Encoder")
    print("")
    print("NOTE: OUT_CSV is overwritten each run. Copy the existing")
    print("      probe_transfer_results.csv aside first to keep it.")


if __name__ == "__main__":
    main()
