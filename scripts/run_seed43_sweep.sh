#!/bin/bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
ALL="sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10"
LOG_DIR="./seed43_logs"; mkdir -p "$LOG_DIR"
SEED=43

run () {
  local ENC="$1" MODE="$2" S="$3" BATCH="$4" LR="$5"
  local LOG="$LOG_DIR/${ENC}_${MODE}_${S}.log"
  echo "=== $(date '+%H:%M:%S') START ${ENC} ${MODE} ${S} seed=${SEED} ==="
  if [ "$MODE" = "loso" ]; then
    SUBJECTS="$ALL" TEST_SUBJECTS="$S" ENCODER="$ENC" MODE=loso \
    DATA_PATH="./eeg_dataset/preprocessed_data" IMG_DIR_TRAINING="./image_set/training_images" \
    IMG_DIR_TEST="./image_set/test_images" FEATURES_DIR="./emb_eeg" FEATURE_TYPE=ViT-H-14 \
    EPOCHS=40 BATCH_SIZE="$BATCH" LR="$LR" EARLY_STOPPING_PATIENCE=10 EMA_DECAY=0.999 \
    LOGIT_SCALE_TYPE=exp VAL_RATIO=0.1 GPU=cuda:0 WANDB_LOGGING=false SEED="$SEED" \
    bash Retrieval/run.sh > "$LOG" 2>&1 || echo "!!! FAILED ${ENC} ${MODE} ${S}"
  else
    SUBJECTS="$S" ENCODER="$ENC" MODE=intra \
    DATA_PATH="./eeg_dataset/preprocessed_data" IMG_DIR_TRAINING="./image_set/training_images" \
    IMG_DIR_TEST="./image_set/test_images" FEATURES_DIR="./emb_eeg" FEATURE_TYPE=ViT-H-14 \
    EPOCHS=40 BATCH_SIZE="$BATCH" LR="$LR" EARLY_STOPPING_PATIENCE=10 EMA_DECAY=0.999 \
    LOGIT_SCALE_TYPE=exp VAL_RATIO=0.1 GPU=cuda:0 WANDB_LOGGING=false SEED="$SEED" \
    bash Retrieval/run.sh > "$LOG" 2>&1 || echo "!!! FAILED ${ENC} ${MODE} ${S}"
  fi
}

for S in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10; do
  run ATMS        loso  "$S" 128 2e-4
  run LaBraM_ATMS loso  "$S" 64  1e-4
  run ATMS        intra "$S" 128 2e-4
  run LaBraM_ATMS intra "$S" 64  1e-4
done
echo "=== SEED 43 COMPLETE $(date) ==="
