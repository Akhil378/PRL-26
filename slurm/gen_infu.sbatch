#!/bin/bash -l
#SBATCH --job-name=infu-gen
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:19:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Part 1 generation, InfU.
#
# Shard count and walltime come from tools/plan_shards.py, run with the
# s/image MEASURED by slurm/smoke.sbatch. Do not request 24:00:00 out of
# caution: SLURM backfill only slots a job into a gap at least as long as
# the request, so a 24 h ask waits for a 24 h gap while a ~2 h ask drops
# into the short gaps that actually exist on a busy cluster.
#
# Resubmitting after a timeout is safe: finished cells are skipped.

set -euo pipefail
module load python/3.12-conda
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
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards 8
