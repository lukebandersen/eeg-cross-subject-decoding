#!/bin/bash
# =============================================================================
# submit_all.sh -- read config/experiment_matrix.tsv and submit every job whose
# status is READY (or a status you choose) to Slurm, injecting account /
# partition / gres from config/arcc_env.sh via sbatch COMMAND-LINE flags.
#
# Command-line sbatch flags OVERRIDE #SBATCH directives in the script, which is
# how we keep account/partition in ONE place (the config) instead of hardcoded
# in every .sbatch file.
#
# USAGE:
#   bash scripts/submit_all.sh READY          # submit all READY rows (dry-run first!)
#   bash scripts/submit_all.sh READY --go      # actually submit
#   bash scripts/submit_all.sh B01 --go        # submit a single exp_id
#
# Without --go it prints what it WOULD submit (dry run). Always dry-run first.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${HERE}/config/arcc_env.sh"

FILTER="${1:-READY}"
GO="${2:-}"

arcc_preflight || { echo "Fix config/arcc_env.sh before submitting."; exit 1; }

echo "Matrix: ${HERE}/config/experiment_matrix.tsv"
echo "Filter: ${FILTER}   Mode: $([[ "$GO" == "--go" ]] && echo SUBMIT || echo DRY-RUN)"
echo "Account=${ARCC_ACCOUNT}  Partition=${ARCC_PARTITION}  Gres=${ARCC_GRES}"
echo "-----------------------------------------------------------------------"

submit_one() {
  local exp_id="$1" encoder="$2" mode="$3" dataset="$4" seeds="$5"
  # expand comma-separated seeds into one array job per seed
  IFS=',' read -ra SEED_ARR <<< "$seeds"
  for seed in "${SEED_ARR[@]}"; do
    [[ "$seed" == "-" ]] && seed=42
    local jobname="${exp_id}_${encoder}_${mode}_s${seed}"
    local sbatch_script="${HERE}/slurm/retrieval_sweep.sbatch"
    # classification-transfer uses a different script
    if [[ "$mode" == "xfer_frozen" || "$mode" == "loso_classify" ]]; then
      sbatch_script="${HERE}/slurm/probe_transfer.sbatch"
    fi
    local cmd=(sbatch
      --account="${ARCC_ACCOUNT}"
      --partition="${ARCC_PARTITION}"
      --gres="${ARCC_GRES}"
      --cpus-per-task="${ARCC_CPUS}"
      --mem="${ARCC_MEM}"
      --time="${ARCC_TIME}"
      --mail-user="${ARCC_EMAIL}"
      --job-name="${jobname}"
      --output="${LOG_DIR}/${jobname}_%A_%a.log"
      --export="ALL,ENCODER=${encoder},MODE=${mode},SEED=${seed},DATASET=${dataset}"
      "${sbatch_script}")
    if [[ "$GO" == "--go" ]]; then
      echo "SUBMIT: ${jobname}"
      "${cmd[@]}"
    else
      echo "WOULD SUBMIT: ${jobname}"
      echo "   ${cmd[*]}"
    fi
  done
}

# parse the matrix
while IFS=$'\t' read -r exp_id encoder mode dataset seeds status notes; do
  [[ "$exp_id" =~ ^#.*$ || -z "$exp_id" ]] && continue
  # filter: either exact exp_id, or a status keyword
  if [[ "$FILTER" == "$exp_id" || "$FILTER" == "$status" ]]; then
    submit_one "$exp_id" "$encoder" "$mode" "$dataset" "$seeds"
  fi
done < "${HERE}/config/experiment_matrix.tsv"

echo "-----------------------------------------------------------------------"
[[ "$GO" == "--go" ]] || echo "DRY RUN complete. Re-run with '--go' as the 2nd arg to actually submit."
