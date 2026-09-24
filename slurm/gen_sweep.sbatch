#!/bin/bash -l
#SBATCH --job-name=sweep-gen
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=NONE

# Part 2 sweep: the identity-strength arms of the Pareto frontier.
#
#   sbatch.tinygpu slurm/gen_sweep.sbatch infu  0.6 "--cpu-offload"
#   sbatch.tinygpu slurm/gen_sweep.sbatch infu  0.8 "--cpu-offload"
#   sbatch.tinygpu slurm/gen_sweep.sbatch pulid 0.6 "--offload --id-start 1"
#   sbatch.tinygpu slurm/gen_sweep.sbatch pulid 0.8 "--offload --id-start 1"
#
# DO NOT run a 1.0 arm. That point of every curve is the main grid, which uses
# the same cells and the same seeds because manifest_sweep.parquet is a strict
# subset of manifest_style.parquet. Regenerating it would spend ~3 GPU-h to
# produce images that already exist, and if any setting had drifted meanwhile it
# would produce a 1.0 point inconsistent with the grid it is supposed to anchor.
#
# The knob differs by method but means the same thing -- how hard identity is
# pushed into the generation:
#   InfU   --cond-scale   (infusenet_conditioning_scale)
#   PuLID  --id-weight
# These are NOT calibrated against each other and the report must not treat 0.6
# on one as equivalent to 0.6 on the other. That is exactly why the result is a
# curve: each method is compared against its own trade-off, and the frontiers
# are then compared as shapes rather than pointwise.
#
# 192 cells over 2 shards is 96 per task: ~40 min for InfU, ~56 min for PuLID.

unset SLURM_EXPORT_ENV

METHOD="${1:?usage: gen_sweep.sbatch <infu|pulid> <scale> \"<flags>\"}"
SCALE="${2:?scale is required, e.g. 0.6 or 0.8}"
FLAGS="${3:-}"
SHARDS="${4:-2}"

case "$SCALE" in
  1.0|1|1.00)
    echo "ERROR: the 1.0 arm is the main grid -- see the comment in this file." >&2
    echo "       Score style_${METHOD} against manifest_sweep instead." >&2
    exit 2 ;;
esac

# Same guard as gen_style.sbatch: PuLID's default start_step is its photoreal
# setting and would handicap it across a stylisation experiment.
if [ "$METHOD" = "pulid" ] && [[ "$FLAGS" != *"--id-start"* ]]; then
  echo "ERROR: PuLID requires an explicit --id-start for Part 2." >&2
  echo "       Use the SAME value as the main grid, or the frontier's 1.0" >&2
  echo "       point will not belong to the same curve as 0.6 and 0.8." >&2
  exit 2
fi

module load python/3.12-conda

export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Output leaf encodes the knob and its value, so the scoring step and the
# frontier plot can both recover which arm an image belongs to from its path.
TAG=$(echo "$SCALE" | tr -d '.')

# shellcheck disable=SC2086
if [ "$METHOD" = "infu" ]; then
  conda activate "$WORK/envs/infu"
  python src/gen_infu.py \
    --manifest benchmark/manifest_sweep.parquet \
    --out      "$PRL/results/images/sweep_infu_cs${TAG}" \
    --infu-repo "$PRL/InfiniteYou" \
    --insightface-root "$INSIGHTFACE_HOME" \
    --model-version aes_stage2 \
    --cond-scale "$SCALE" \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SHARDS" $FLAGS
else
  conda activate "$WORK/envs/pulid"
  python src/gen_pulid.py \
    --manifest benchmark/manifest_sweep.parquet \
    --out      "$PRL/results/images/sweep_pulid_iw${TAG}" \
    --pulid-repo "$PRL/PuLID" \
    --id-weight "$SCALE" \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SHARDS" $FLAGS
fi
rc=$?

echo "--- sweep $METHOD scale=$SCALE shard ${SLURM_ARRAY_TASK_ID:-0} done, rc=$rc ---"
exit $rc
