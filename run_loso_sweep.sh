#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
ALL_SUBJECTS="sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10"
LOG_DIR="./loso_sweep_logs"
mkdir -p "$LOG_DIR"

run_fold () {
  local ENC="$1" HELDOUT="$2" BATCH="$3" LR="$4"
  local LOG="$LOG_DIR/${ENC}_${HELDOUT}.log"
  echo "=== $(date '+%H:%M:%S')  START ${ENC} holdout=${HELDOUT} ==="
  DATA_PATH="./eeg_dataset/preprocessed_data" \
  IMG_DIR_TRAINING="./image_set/training_images" \
  IMG_DIR_TEST="./image_set/test_images" \
  FEATURES_DIR="./emb_eeg" \
  SUBJECTS="$ALL_SUBJECTS" \
  TEST_SUBJECTS="$HELDOUT" \
  ENCODER="$ENC" MODE=loso FEATURE_TYPE=ViT-H-14 \
  EPOCHS=40 BATCH_SIZE="$BATCH" LR="$LR" EARLY_STOPPING_PATIENCE=10 \
  EMA_DECAY=0.999 LOGIT_SCALE_TYPE=exp VAL_RATIO=0.1 GPU=cuda:0 WANDB_LOGGING=false \
  bash Retrieval/run.sh > "$LOG" 2>&1 || echo "!!! FOLD FAILED: ${ENC} ${HELDOUT} (see $LOG)"
  echo "=== $(date '+%H:%M:%S')  DONE  ${ENC} holdout=${HELDOUT} ==="
}

# ATMS folds (sub-08 already done) — batch 128, LR 2e-4
for S in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-09 sub-10; do
  run_fold ATMS "$S" 128 2e-4
done
# LaBraM folds (sub-08 already done) — batch 64, LR 1e-4, full fine-tune
for S in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-09 sub-10; do
  run_fold LaBraM_ATMS "$S" 64 1e-4
done
echo "=== SWEEP COMPLETE $(date) ==="
