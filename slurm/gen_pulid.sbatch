#!/bin/bash -l
#SBATCH --job-name=pulid-gen
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:19:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=NONE

# Part 1 generation, PuLID-FLUX baseline.
# Submit with:  sbatch.tinygpu slurm/gen_pulid.sbatch
# Verify the FluxGenerator signature with --limit 2 before launching the array.

unset SLURM_EXPORT_ENV

module load python/3.12-conda
conda activate "$WORK/envs/pulid"

export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1

python src/gen_pulid.py \
  --manifest benchmark/manifest_repro.parquet \
  --out      "$PRL/results/images/pulid" \
  --pulid-repo "$PRL/PuLID" \
  --offload \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards 8
