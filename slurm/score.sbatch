#!/bin/bash -l
#SBATCH --job-name=score
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Scoring runs on a cheap GPU while the A100 keeps generating.
# METHOD and MANIFEST are passed through:
#   sbatch.tinygpu --export=ALL,METHOD=infu_aes2,MANIFEST=manifest_repro slurm/score.sbatch

set -euo pipefail
module load python/3.11-anaconda
conda activate "$WORK/envs/metrics"

export HF_HOME="$WORK/hf"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

srun python src/score_all.py \
  --manifest "benchmark/${MANIFEST}.parquet" \
  --images   "$PRL/results/images/${METHOD}" \
  --method   "${METHOD}" \
  --id-root  "$INSIGHTFACE_HOME/models" \
  --out      "$PRL/results/scores/${METHOD}_${MANIFEST}.parquet"
