#!/bin/bash
# =============================================================================
# retrieval_sweep.sbatch  --  one encoder x one mode, all 10 subject-folds,
# run as a Slurm ARRAY so the folds execute in PARALLEL across GPUs.
#
# SUBMIT LIKE:
#   sbatch --export=ALL,ENCODER=EEGNetv4_Encoder,MODE=loso,SEED=42,DATASET=things \
#          slurm/retrieval_sweep.sbatch
#
# The array index (0-9) selects the held-out / target subject. On a cluster with
# free GPUs this finishes ~10x faster than the serial nohup sweep you ran locally.
# =============================================================================
#SBATCH --job-name=eeg_retr
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x_%A_%a.log        # %x=jobname %A=arrayjobid %a=arraytask; redirected below
#SBATCH --mail-type=END,FAIL

# --- load central config (account, partition, paths) ---
# SLURM copies the submitting environment; we re-source to be safe/reproducible.
source "$(dirname "$(scontrol show job $SLURM_JOB_ID | grep -oP 'Command=\K\S+' | head -1)")/../config/arcc_env.sh" 2>/dev/null \
  || source "${SLURM_SUBMIT_DIR}/config/arcc_env.sh"

# Apply account/partition/resources from config (as sbatch directives can't read vars,
# we re-exec requeue-safe by having submit_all.sh pass them; here we just use them for paths).
: "${ENCODER:?set ENCODER}"; : "${MODE:?set MODE}"; : "${SEED:=42}"; : "${DATASET:=things}"

SUBJECT_NUM=$(( SLURM_ARRAY_TASK_ID + 1 ))
SUBJECT=$(printf "sub-%02d" "$SUBJECT_NUM")

echo "=============================================================="
echo "Job $SLURM_JOB_ID  array task $SLURM_ARRAY_TASK_ID"
echo "Encoder=$ENCODER  Mode=$MODE  Seed=$SEED  Dataset=$DATASET  Target=$SUBJECT"
echo "Node: $(hostname)   Started: $(date)"
echo "=============================================================="

# --- environment (ARCC best practice: purge, then load exactly what you need) ---
module purge
module load "${ARCC_CONDA_MODULE}"
# `conda activate` is a shell FUNCTION, not a binary. Interactive logins get it
# from ~/.bashrc; a batch shell does not, so without sourcing conda.sh the
# activate silently no-ops and the job runs on the BASE python -- which has no
# torch. Source it explicitly, then activate by full path.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ARCC_ENV_NAME}" || { echo "FATAL: could not activate ${ARCC_ENV_NAME}"; exit 3; }
# fail loudly if we are somehow still on the wrong interpreter
python -c "import torch" 2>/dev/null || { echo "FATAL: torch missing -- wrong env: $(which python)"; exit 4; }
echo "Python: $(which python)"; python --version
echo "GPU allocation:"; nvidia-smi -L

cd "${REPO_DIR}/Retrieval" || { echo "REPO_DIR/Retrieval not found: ${REPO_DIR}"; exit 2; }

# --- build the command for the requested dataset ---
# EPOCH BUDGET: 300 cap + early stopping, NOT a fixed 40.
# Measured convergence (sub-01, verified locally before staging):
#     ATMS           ~40 epochs   (val loss flat 3.098->3.097 by ep 35-40)
#     LaBraM_ATMS    ~22 epochs   (ES fired; val loss RISING after ~ep 13)
#     EEGConformer  ~196 epochs   (ES first incremented at ep 85!)
# A fixed 40-epoch cap CONVERGED ATMS/LaBraM but badly UNDERTRAINED
# EEGConformer: v200 top-1 was 0.0100 at ep40 vs 0.0750 at ep200 (7.5x).
# Shipping at 40 would have reported undertraining as "specialist collapse"
# and invalidated the panel. A generous cap + early stopping means every
# encoder trains to ITS OWN convergence: fixed protocol, no per-encoder
# tuning, no hidden budget variable. Fast encoders stop early and cost nothing.
: "${EPOCHS:=300}"
: "${PATIENCE:=10}"
COMMON=(--encoder_type "$ENCODER" --mode "$MODE" --seed "$SEED"
        --epochs "$EPOCHS" --batch_size 128 --lr 2e-4 --val_ratio 0.1)

if [[ "$DATASET" == "things" ]]; then
  ALL_SUBJECTS=$(seq -f "sub-%02g" 1 10)
  if [[ "$MODE" == "loso" ]]; then
    # ONE FOLD PER ARRAY TASK. The training pool must be all ten; --test_subjects
    # picks the held-out subject (train_unified.py's existing arg; no patch).
    # Passing only "$SUBJECT" here trains on [sub] minus [sub] = nothing and
    # dies with "torch.cat(): expected a non-empty list of Tensors".
    CMD=(python train_unified.py "${COMMON[@]}"
         --dataset things
         --data_path "$DATA_PATH"
         --img_dir_training "$IMG_TRAIN" --img_dir_test "$IMG_TEST"
         --clip_features_dir "$FEATURES_DIR"
         --n_chans 63 --n_times 250
         --subjects $ALL_SUBJECTS --test_subjects "$SUBJECT")
  else
    CMD=(python train_unified.py "${COMMON[@]}"
         --dataset things
         --data_path "$DATA_PATH"
         --img_dir_training "$IMG_TRAIN" --img_dir_test "$IMG_TEST"
         --clip_features_dir "$FEATURES_DIR"
         --n_chans 63 --n_times 250
         --subjects "$SUBJECT")
  fi
elif [[ "$DATASET" == "things_meg" ]]; then
  echo "things_meg path not yet wired -- placeholder (Block E, parked)."; exit 3
else
  echo "unknown DATASET=$DATASET"; exit 3
fi

echo "RUN: ${CMD[*]}"
srun "${CMD[@]}"
status=$?
echo "Finished: $(date)   exit=$status"
exit $status
