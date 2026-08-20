#!/bin/bash -l
#SBATCH --job-name=smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Phase 1 timing calibration. Generates 20 images under one memory profile and
# reports median seconds per image. Run once per profile, then fix the choice.

set -euo pipefail
module load python/3.12-conda
conda activate "$WORK/envs/infu"
export HF_HOME="$WORK/hf"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

srun python src/gen_infu.py \
  --manifest benchmark/manifest_repro.parquet \
  --out "$PRL/results/images/smoke_${PROFILE}" \
  --infu-repo "$PRL/InfiniteYou" \
  --insightface-root "$INSIGHTFACE_HOME/models" \
  --limit 20 ${EXTRA_FLAGS}
