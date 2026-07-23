#!/bin/bash
# =============================================================================
# arcc_env.sh  --  SINGLE SOURCE OF TRUTH for all ARCC job scripts.
#
# Edit the values in the "EDIT THESE" block ONCE. Every sbatch script and helper
# sources this file, so you never hardcode your account or partition anywhere
# else. This is deliberate: it means one correct edit, not fifteen.
#
# HOW TO FIND THE RIGHT VALUES (run these on an ARCC login node first):
#   - Your account(s):     sacctmgr show assoc user=$USER format=account,qos
#   - Available partitions: sinfo -s              (look for a GPU partition)
#   - GPU partitions/types: scontrol show partition | grep -i -A2 gpu
#   - Conda module name:    module spider miniconda   (or: module avail conda)
#   - Your project storage: ls /pfs/tc1/project/      (find your project dir)
#
# ARCC's current GPU system is MedicineBow; common GPU partitions look like
# 'mb-a30', 'mb-l40s', or an H100 partition. CONFIRM with sinfo -- do not trust
# this comment, hardware changes. Set PARTITION and GRES to what sinfo shows.
# =============================================================================

# ----------------------------- EDIT THESE ------------------------------------
export ARCC_ACCOUNT="mayocancerai"        # confirmed: sacctmgr show assoc user=mwolff3
export ARCC_PARTITION="mb-h100"           # gpu:h100:8 x6 nodes. Fallback: beartooth-gpu (gpu:a30:2, idle nodes)
export ARCC_GRES="gpu:1"                  # e.g. gpu:1 ; or gpu:l40s:1 to pin a type
export ARCC_EMAIL="lukebandersen@gmail.com"

# Where the project lives on ARCC (project storage, NOT your home dir --
# home is small; put code + data under /pfs/tc1/project/<project>/ ).
# MedicineBow filesystem (verified via arccquota 23 Jul 2026):
#   /home/mwolff3          50 GB   <- TOO SMALL. Preprocessed EEG alone is 97 GB.
#   /gscratch/mwolff3       5 TB   <- fast scratch, but subject to purge policy
#   /project/mayocancerai   5 TB   <- shared group space, persistent
# Code + data live in /project (persistent); job scratch/outputs go to /gscratch.
export ARCC_PROJECT_ROOT="/project/mayocancerai/eeg_decode"
export ARCC_SCRATCH="/gscratch/mwolff3/eeg_decode"

# Conda: either a module name to `module load`, or a path to `conda activate`.
export ARCC_CONDA_MODULE="miniconda3/24.3.0"   # exact version from `module avail conda` on MedicineBow
# FULL PATH, not a name: the env lives in /project (5 TB), not $HOME (50 GB cap).
# Python 3.12 deliberately -- braindecode 0.8 installs incompletely on 3.11 here
# (ModuleNotFoundError: braindecode.training), and 3.12 is what the verified
# local environment uses.
export ARCC_ENV_NAME="/project/mayocancerai/eeg_decode/envs/eeg312"

# Per-job resource defaults (override per-script if a job is heavier/lighter).
export ARCC_CPUS="8"                       # cpus-per-task
# A 9-subject LOSO fold is the heavy case: ~4.2 GB per training subject, so
# peak RSS during dataset assembly is ~38 GB. 32G was too tight and would OOM
# during loading, before a single epoch ran.
export ARCC_MEM="64G"                      # RAM per job
# WALL CLOCK -- sized to the SLOWEST fold, not the average.
# Measured/estimated per-LOSO-fold on one GPU:
#   EEGNet      ~4.7 hr (measured)      CBraMod  ~1.5 hr (measured)
#   ShallowFBCSP ~5 hr (est)            EEGConformer ~20 hr (est: 255 ep x 9x data)
# At 12:00:00 every EEGConformer task would be killed at hour 12 of ~20, losing
# the fold with nothing written. Partition limit is 7 days; 2 days gives margin
# on an H100 (faster than the 4080 these estimates came from) without asking for
# a week. Override per-sweep with ARCC_TIME=... if a sweep is known to be short.
export ARCC_TIME="2-00:00:00"              # 2 days
# -----------------------------------------------------------------------------

# --------------------------- DERIVED (do not edit) ---------------------------
export REPO_DIR="${ARCC_PROJECT_ROOT}/EEG_Image_decode-develop"
export DATA_PATH="${REPO_DIR}/eeg_dataset/preprocessed_data"
export IMG_TRAIN="${REPO_DIR}/image_set/training_images"
export IMG_TEST="${REPO_DIR}/image_set/test_images"
export FEATURES_DIR="${REPO_DIR}/emb_eeg"
export OUTPUT_DIR="${REPO_DIR}/outputs"
export LOG_DIR="${ARCC_PROJECT_ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}" 2>/dev/null

# Sanity guard: refuse to submit with placeholder values still in place.
arcc_preflight() {
  local bad=0
  for v in ARCC_ACCOUNT ARCC_PARTITION ARCC_PROJECT_ROOT; do
    if [[ "${!v}" == *CHANGE_ME* ]]; then
      echo "[arcc_env] ERROR: $v is still 'CHANGE_ME'. Edit config/arcc_env.sh first." >&2
      bad=1
    fi
  done
  return $bad
}
