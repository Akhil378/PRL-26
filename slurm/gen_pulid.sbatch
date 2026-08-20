#!/bin/bash -l
#SBATCH --job-name=pulid-gen
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:19:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
module load python/3.12-conda
conda activate "$WORK/envs/pulid"

export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

srun python src/gen_pulid.py \
  --manifest benchmark/manifest_repro.parquet \
  --out      "$PRL/results/images/pulid" \
  --pulid-repo "$PRL/PuLID" \
  --offload \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards 8
