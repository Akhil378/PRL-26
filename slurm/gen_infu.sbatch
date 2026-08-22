#!/bin/bash -l
#SBATCH --job-name=infu-gen
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:19:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=NONE

# Part 1 generation, InfU.
#
# Submit with:  sbatch.tinygpu slurm/gen_infu.sbatch
# (the .tinygpu suffix is required on the shared tinyx frontend; without it the
#  command targets TinyFat instead.)
#
# Shard count and --time come from tools/plan_shards.py using the s/image
# MEASURED by slurm/smoke.sbatch. Do not pad --time to 24:00:00 out of caution:
# SLURM backfill only fills a gap at least as long as the request, so a 24 h ask
# waits for a 24 h gap while a ~2 h ask drops into the short gaps that actually
# exist. Resubmitting after a timeout is safe -- finished cells are skipped.

unset SLURM_EXPORT_ENV      # required by NHR: without it srun/modules misbehave

module load python/3.12-conda
conda activate "$WORK/envs/infu"

export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

# Compute nodes have NO outbound internet. Any HuggingFace call that tries the
# network hangs on connect retries instead of failing: a smoke job sat for 25
# minutes inside snapshot_download() for a repo already fully cached on $WORK.
# Offline mode makes the hub resolve from cache and fail fast if something is
# genuinely missing, which is the error you want.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/gen_infu.py \
  --manifest benchmark/manifest_repro.parquet \
  --out      "$PRL/results/images/infu_aes2" \
  --infu-repo "$PRL/InfiniteYou" \
  --insightface-root "$INSIGHTFACE_HOME" \
  --model-version aes_stage2 \
  --quantize-8bit \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards 8
