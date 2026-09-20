#!/bin/bash -l
#SBATCH --job-name=score-ceil
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# CLIPScore for the text-only ceiling run. Runs on the cheap partition; it needs
# no A100 and should not compete with generation for one.

unset SLURM_EXPORT_ENV
set -uo pipefail

module load python/3.12-conda
conda activate "$WORK/envs/metrics"

export HF_HOME="$WORK/hf"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1

python tools/score_ceiling.py \
  --images "$PRL/results/images/ceiling" \
  --out    "$PRL/results/scores/ceiling_prompts.parquet"
rc=$?
echo "scorer exit: $rc"
exit $rc
