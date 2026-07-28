#!/bin/bash
# =============================================================================
# local_sweep.sh -- run the experiment matrix on YOUR 4080. Resumable.
#
# Same science as the ARCC array jobs: same 8 sweeps, same 10 subjects, same
# 300-epoch + early-stopping protocol, same metrics. The ONLY difference is
# that ARCC runs the 10 folds in parallel and this runs them one at a time.
# Identical numbers, more wall clock.
#
# RESUMABLE BY DESIGN: every (encoder, mode, subject) that already produced a
# results CSV is SKIPPED. Ctrl-C any time. Re-run to continue. Hand the
# remainder to ARCC whenever the account lands -- just run submit_all.sh there
# and it will do whatever is still missing.
#
# RUN (Git Bash, repo root):
#     bash local_sweep.sh                 # everything still outstanding
#     bash local_sweep.sh --dry-run       # list what WOULD run, do nothing
#     bash local_sweep.sh --only B01      # a single matrix row
#     bash local_sweep.sh --status        # progress table, run nothing
#
# EPOCH PROTOCOL (measured locally, see the convergence notes below):
#     ATMS          ~40 epochs   LaBraM ~22 epochs   EEGConformer ~196 epochs
#   A fixed 40-epoch cap converged ATMS/LaBraM but UNDERTRAINED EEGConformer
#   7.5x (v200 0.0100 -> 0.0750). So: generous cap + early stopping, and every
#   encoder trains to its own convergence. Fixed protocol, no per-encoder
#   tuning, no hidden budget variable.
# =============================================================================
set -uo pipefail

# ---------------------------- paths (edit if needed) ------------------------
DATA_PATH="./EEG_Image_decode/Preprocessed_data_250Hz"
IMG_TRAIN="./image_set/training_images"
IMG_TEST="./image_set/test_images"
FEATS="./emb_eeg"
OUT_ROOT="./outputs/retrieval"
LOG_DIR="./sweep_logs"
mkdir -p "$LOG_DIR"

EPOCHS=300
PATIENCE=10
BATCH=128
LR=2e-4
SEED=42
SUBJECTS=$(seq -f "sub-%02g" 1 10)

# ---------------------------- the sweeps (= matrix READY rows) --------------
# id|encoder|mode
SWEEPS=(
  "B01|EEGNetv4_Encoder|intra"
  "B02|EEGNetv4_Encoder|loso"
  "B03|EEGConformer_Encoder|intra"
  "B04|EEGConformer_Encoder|loso"
  "B05|ShallowFBCSPNet_Encoder|intra"
  "B06|ShallowFBCSPNet_Encoder|loso"
  "C01|CBraMod_Encoder|intra"
  "C02|CBraMod_Encoder|loso"
)

DRY=0; ONLY=""; STATUS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --status)  STATUS=1 ;;
    --only)    shift; ONLY="$1" ;;
    *) echo "unknown flag: $1"; exit 1 ;;
  esac
  shift
done

# A run is DONE if its results CSV exists (that is the last thing written).
is_done() {
  local enc="$1" mode="$2" sub="$3"
  compgen -G "${OUT_ROOT}/${enc}/${sub}/*/${enc}_${mode}_${sub}.csv" > /dev/null 2>&1
}

# ---------------------------- status table ----------------------------------
if [ $STATUS -eq 1 ]; then
  echo "======================================================================"
  printf " %-4s %-26s %-6s %s\n" "ID" "ENCODER" "MODE" "PROGRESS"
  echo "======================================================================"
  total=0; done_total=0
  for row in "${SWEEPS[@]}"; do
    IFS='|' read -r id enc mode <<< "$row"
    d=0
    for sub in $SUBJECTS; do is_done "$enc" "$mode" "$sub" && d=$((d+1)); done
    total=$((total+10)); done_total=$((done_total+d))
    bar=""; for i in $(seq 1 10); do [ $i -le $d ] && bar="${bar}#" || bar="${bar}."; done
    printf " %-4s %-26s %-6s [%s] %d/10\n" "$id" "$enc" "$mode" "$bar" "$d"
  done
  echo "======================================================================"
  echo " TOTAL: ${done_total}/${total} runs complete ($(( done_total * 100 / total ))%)"
  exit 0
fi

# ---------------------------- the sweep -------------------------------------
started=$(date +%s)
todo=0
for row in "${SWEEPS[@]}"; do
  IFS='|' read -r id enc mode <<< "$row"
  for sub in $SUBJECTS; do is_done "$enc" "$mode" "$sub" || todo=$((todo+1)); done
done
echo "======================================================================"
echo " LOCAL SWEEP  |  $todo runs outstanding  |  epochs=$EPOCHS patience=$PATIENCE"
echo " Resumable: Ctrl-C any time, re-run to continue, or ship the rest to ARCC."
echo "======================================================================"

n=0; failed=0
for row in "${SWEEPS[@]}"; do
  IFS='|' read -r id enc mode <<< "$row"
  [ -n "$ONLY" ] && [ "$ONLY" != "$id" ] && continue

  # -------------------------------------------------------------------------
  # INVOCATION DIFFERS BY MODE. This bit was wrong once; do not "simplify" it.
  #
  #   intra: --subjects sub-XX          -> one invocation per subject.
  #   loso : --subjects <ALL TEN>       -> ONE invocation; train_unified loops
  #          `for sub in args.subjects`, training on args.subjects MINUS sub and
  #          testing on sub. Passing a single subject makes the training set
  #          empty (subjects_train=[sub-01], exclude_subject=sub-01) and dies
  #          with "torch.cat(): expected a non-empty list of Tensors".
  #
  # Consequence: LOSO resume granularity is PER SWEEP, not per fold.
  # -------------------------------------------------------------------------
  # -------------------------------------------------------------------------
  # LOSO needs the FULL pool in --subjects (train = pool minus held-out) plus
  # --test_subjects to isolate one fold. Uses train_unified.py's EXISTING arg
  # (for sub in (args.test_subjects or args.subjects)); NO patch required.
  # Passing a single subject to --mode loso trains on nothing and dies with
  # "torch.cat(): expected a non-empty list of Tensors". Do not "simplify".
  # NOTE: LOSO trains on 9 subjects = ~9x the data per epoch, so expect
  #       roughly 9x this encoder's intra time PER FOLD.
  # -------------------------------------------------------------------------
  if [ "$mode" = "loso" ]; then
    for sub in $SUBJECTS; do
      if is_done "$enc" "$mode" "$sub"; then
        echo "  [skip] $id $enc loso $sub (already done)"
        continue
      fi
      n=$((n+1))
      LOG="${LOG_DIR}/${id}_${enc}_loso_${sub}.log"
      echo ""
      echo "----------------------------------------------------------------------"
      echo " [$n] $id  $enc  loso  hold-out=$sub  (trains on other 9, ~9x data)"
      echo " started $(date +%H:%M:%S)   log: $LOG"
      echo "----------------------------------------------------------------------"
      if [ $DRY -eq 1 ]; then echo "  (dry run)"; continue; fi

      t0=$(date +%s)
      python Retrieval/train_unified.py \
        --encoder_type "$enc" --mode loso --dataset things \
        --data_path "$DATA_PATH" --img_dir_training "$IMG_TRAIN" \
        --img_dir_test "$IMG_TEST" --clip_features_dir "$FEATS" \
        --n_chans 63 --n_times 250 \
        --subjects $SUBJECTS --test_subjects "$sub" \
        --epochs $EPOCHS --batch_size $BATCH --lr $LR --seed $SEED \
        --val_ratio 0.1 > "$LOG" 2>&1
      rc=$?
      mins=$(( ($(date +%s)-t0)/60 ))
      if [ $rc -ne 0 ]; then
        echo "  FAILED (exit $rc, ${mins}m). tail:"; tail -5 "$LOG"; failed=$((failed+1))
      else
        top1=$(grep -oE "Top-1 \(v200\): [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$")
        ep=$(grep -oE "^Epoch [0-9]+/" "$LOG" | tail -1 | grep -oE "[0-9]+")
        echo "  done in ${mins}m | stopped ~ep ${ep:-?} | v200 top-1 = ${top1:-NA}"
      fi
    done
    continue
  fi

  # ---- intra: one invocation per subject (resumable per fold) ----
  for sub in $SUBJECTS; do
    if is_done "$enc" "$mode" "$sub"; then
      echo "  [skip] $id $enc $mode $sub (already done)"
      continue
    fi
    n=$((n+1))
    LOG="${LOG_DIR}/${id}_${enc}_${mode}_${sub}.log"
    echo ""
    echo "----------------------------------------------------------------------"
    echo " [$n] $id  $enc  $mode  $sub"
    echo " started $(date +%H:%M:%S)   log: $LOG"
    echo "----------------------------------------------------------------------"

    if [ $DRY -eq 1 ]; then echo "  (dry run)"; continue; fi

    t0=$(date +%s)
    python Retrieval/train_unified.py \
      --encoder_type "$enc" --mode "$mode" --dataset things \
      --data_path "$DATA_PATH" --img_dir_training "$IMG_TRAIN" \
      --img_dir_test "$IMG_TEST" --clip_features_dir "$FEATS" \
      --n_chans 63 --n_times 250 --subjects "$sub" \
      --epochs $EPOCHS --batch_size $BATCH --lr $LR --seed $SEED \
      --val_ratio 0.1 > "$LOG" 2>&1
    rc=$?
    t1=$(date +%s); mins=$(( (t1-t0)/60 ))

    if [ $rc -ne 0 ]; then
      echo "  FAILED (exit $rc, ${mins}m). tail:"; tail -5 "$LOG"; failed=$((failed+1))
    else
      top1=$(grep -oE "Top-1 \(v200\): [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$")
      ep=$(grep -oE "^Epoch [0-9]+/" "$LOG" | tail -1 | grep -oE "[0-9]+")
      echo "  done in ${mins}m | stopped ~ep ${ep:-?} | v200 top-1 = ${top1:-NA}"
    fi
  done
done

el=$(( ($(date +%s) - started) / 60 ))
echo ""
echo "======================================================================"
echo " SWEEP PAUSED/FINISHED after ${el}m.  ran=$n  failed=$failed"
echo " Progress:  bash local_sweep.sh --status"
echo " Continue:  bash local_sweep.sh"
echo " To ARCC:   stage, then bash scripts/submit_all.sh READY --go"
echo "            (it does whatever is still missing -- nothing is wasted)"
echo "======================================================================"
