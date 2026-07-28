#!/bin/bash
# =============================================================================
# declutter.sh -- move clutter into _archive/ WITHOUT touching anything the
# pipeline uses. Run from the repo root.
#
#   bash declutter.sh              # DRY RUN (lists moves, changes nothing)
#   bash declutter.sh --go         # actually move
#
# PHILOSOPHY: conservative. It only archives files that are provably safe:
#   - *.bak* / *.bak_repopath  (backups)
#   - stray *.log              (run logs; results are in CSVs, not logs)
#   - dead-end patch_*.py from the CONCLUDED EEG-ImageNet / Alljoined negatives
#   - obvious junk (environment.ym typo)
# It moves nothing that the sweep or core pipeline imports. A guard list of
# protected files makes damn sure of it: if a target is protected, it is skipped
# and reported, never moved.
#
# NOTHING IS DELETED. Everything goes to _archive/<category>/ and can be moved
# back. _archive/ should be added to .gitignore.
# =============================================================================
set -uo pipefail

GO=0
[ "${1:-}" = "--go" ] && GO=1

ARCHIVE="_archive"

# ---- PROTECTED: never archive these, whatever the patterns say --------------
# Everything the running sweep, the gates, the probes, or the core pipeline
# touches. If you are unsure whether something belongs here, it does.
PROTECTED=(
  "local_sweep.sh" "run_all_gates.sh" "retrieval_sweep.sbatch"
  "setup_cbramod.sh" "fix_cbramod_import.py"
  "patch_braindecode_compat.py" "patch_register_cbramod.py"
  "eegdatasets.py" "encoder_utils.py" "probe_transfer.py"
  "probe_loso_classify.py" "probe_transfer_results.csv"
  "aggregate_all.py" "aggregate_loso.py" "aggregate_seed43.py" "aggregate_seed44.py"
  "requirements.txt" "requirements_arcc.txt" "environment.yml"
  "alljoined_loader.py" "alljoined_dataset.py" "extract_alljoined_clip.py"
  "eegimagenet_dataset.py" "extract_eegimagenet_clip.py" "wnids.txt"
  "setup.sh" "README.md" "LICENSE" "CONTRIBUTING.md" "pretrained_paths.py"
  "github_upload.sh" "scan_paths.py" "fix_sweep_paths.py"
)

is_protected() {
  local f="$1"
  for p in "${PROTECTED[@]}"; do [ "$f" = "$p" ] && return 0; done
  return 1
}

moved=0; skipped=0
do_move() {
  local f="$1" cat="$2"
  [ -e "$f" ] || return 0
  if is_protected "$(basename "$f")"; then
    echo "  [PROTECTED — skip] $f"; skipped=$((skipped+1)); return 0
  fi
  if [ $GO -eq 1 ]; then
    mkdir -p "$ARCHIVE/$cat"
    mv "$f" "$ARCHIVE/$cat/"
    echo "  moved   $f -> $ARCHIVE/$cat/"
  else
    echo "  would move  $f -> $ARCHIVE/$cat/"
  fi
  moved=$((moved+1))
}

echo "=============================================================="
echo " declutter  $([ $GO -eq 1 ] && echo '(LIVE)' || echo '(DRY RUN — nothing changes)')"
echo "=============================================================="

echo ""
echo "[backups]"
for f in *.bak *.bak_* *.bak_repopath; do do_move "$f" backups; done

echo ""
echo "[logs]"
for f in *.log; do do_move "$f" logs; done

echo ""
echo "[dead-end patches: EEG-ImageNet / Alljoined (concluded negatives)]"
for f in patch_alljoined_wiring.py alljoined_pooled_patch.py apply_wiring.py \
         patch_build_montage.py patch_evalk.py patch_labram_eegimagenet.py \
         patch_loader_dropch.py patch_loader_pad.py patch_wiring_labram.py \
         patch_wiring_pad.py patch_znorm.py; do
  do_move "$f" dead_end_patches
done

echo ""
echo "[junk]"
do_move "environment.ym" junk        # empty typo'd file (real one is environment.yml)

echo ""
echo "=============================================================="
echo " $([ $GO -eq 1 ] && echo 'MOVED' || echo 'WOULD MOVE') $moved file(s); skipped $skipped protected."
if [ $GO -eq 0 ]; then
  echo " This was a DRY RUN. Re-run with --go to apply."
else
  echo " All in $ARCHIVE/. Nothing deleted. Add _archive/ to .gitignore."
  echo " Sanity check before resuming the sweep:"
  echo "   bash run_all_gates.sh --quick"
fi
echo "=============================================================="
