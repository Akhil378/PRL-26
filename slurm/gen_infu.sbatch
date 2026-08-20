#!/bin/bash -l
#SBATCH --job-name=infu-gen
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Part 1 generation, InfU. Four shards of ~375 images.
# Resubmitting after a timeout is safe: finished cells are skipped on startup.

set -euo pipefail
module load python/3.11-anaconda
conda activate "$WORK/envs/infu"

export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

srun python src/gen_infu.py \
  --manifest benchmark/manifest_repro.parquet \
  --out      "$PRL/results/images/infu_aes2" \
  --infu-repo "$PRL/InfiniteYou" \
  --insightface-root "$INSIGHTFACE_HOME/models" \
  --model-version aes_stage2 \
  --quantize-8bit \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards 4
