#!/bin/bash -l
#SBATCH --job-name=style-gen
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=NONE

# Part 2 generation: the out-of-distribution stylization stress test.
#
#   sbatch.tinygpu slurm/gen_style.sbatch infu  "--cpu-offload"
#   sbatch.tinygpu slurm/gen_style.sbatch pulid "--offload --id-start 1"
#
# Sweep arms (the Pareto frontier) reuse this script with a scale knob and their
# own output directory, so they never collide with the main grid:
#
#   sbatch.tinygpu slurm/gen_style.sbatch infu  "--cpu-offload --cond-scale 0.6" 6 style_infu_cs06
#   sbatch.tinygpu slurm/gen_style.sbatch pulid "--offload --id-start 1 --id-weight 0.6" 6 style_pulid_iw06
#
# Positional arguments, because --export=NONE discards the submitting shell's
# environment (SUBMITTING.md): <method> "<flags>" [shards] [out-dir-leaf].
#
# InfU runs at bf16 (--cpu-offload) here too, matching Part 1 (CONTEXT 6.2); the
# run_config.json beside every Part 2 image directory records quantize_8bit: false.
# 576 cells over 6 shards is 96 per task: ~55 min for InfU at 33.8 s/img and
# ~56 min for PuLID at 35.1 s/img, both inside the 1:30 request.
#
# ---------------------------------------------------------------------------
# WHY --id-start IS MANDATORY FOR PuLID AND MUST BE THE SAME FOR EVERY CELL
# ---------------------------------------------------------------------------
# PuLID's docs recommend start_step=4 for photorealistic output and 0-1 for
# stylised output. Part 2's manifest holds BOTH: every base prompt appears once
# as a photoreal control and three times stylised, and all four share a seed so
# the within-cell difference isolates the style effect (CONTEXT decision 3).
#
# Varying start_step by style would destroy exactly that. The photoreal control
# and its stylised siblings would differ in two ways at once -- the style AND the
# identity-injection schedule -- and the difference would no longer be
# attributable to style. A control has to run under the same conditioning as the
# thing it controls for.
#
# So one value covers all 576 cells, and it is passed explicitly rather than
# defaulted: this is the central fairness parameter of the experiment and it
# belongs in a command that can be read back afterwards. It is recorded in
# run_config.json regardless.
#
# Consequence worth a line in the report: Part 2's photoreal cells are NOT
# comparable to Part 1's, which run PuLID's photoreal value. They are not meant
# to be -- Part 2's photoreal condition is an internal control for Part 2, not a
# re-run of Part 1. InfU has no equivalent knob, so the asymmetry is PuLID-side
# only.

unset SLURM_EXPORT_ENV

METHOD="${1:?usage: gen_style.sbatch <infu|pulid> \"<flags>\" [shards] [out-leaf]}"
FLAGS="${2:-}"
SHARDS="${3:-6}"
OUTNAME="${4:-}"

# Fail before the queue rather than after a pipeline load: an unset id-start
# would silently run PuLID at its photoreal default across a stylisation
# experiment, which is the single most expensive wrong setting available here.
if [ "$METHOD" = "pulid" ] && [[ "$FLAGS" != *"--id-start"* ]]; then
  echo "ERROR: PuLID requires an explicit --id-start for Part 2." >&2
  echo "       Its default (4) is PuLID's photorealistic setting and would" >&2
  echo "       handicap it on precisely the question Part 2 asks." >&2
  echo "       See the comment block in this script." >&2
  exit 2
fi

module load python/3.12-conda

export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

# Compute nodes have NO outbound internet; offline mode resolves from cache and
# fails fast instead of hanging on connect retries.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# shellcheck disable=SC2086
if [ "$METHOD" = "infu" ]; then
  conda activate "$WORK/envs/infu"
  python src/gen_infu.py \
    --manifest benchmark/manifest_style.parquet \
    --out      "$PRL/results/images/${OUTNAME:-style_infu}" \
    --infu-repo "$PRL/InfiniteYou" \
    --insightface-root "$INSIGHTFACE_HOME" \
    --model-version aes_stage2 \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SHARDS" $FLAGS
else
  conda activate "$WORK/envs/pulid"
  python src/gen_pulid.py \
    --manifest benchmark/manifest_style.parquet \
    --out      "$PRL/results/images/${OUTNAME:-style_pulid}" \
    --pulid-repo "$PRL/PuLID" \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$SHARDS" $FLAGS
fi
rc=$?

# Exit with the GENERATOR's status, not the last echo's. Job 1789452 crashed in
# python and still reported COMPLETED 0:0 because a trailing command succeeded.
echo "--- shard ${SLURM_ARRAY_TASK_ID:-0} done, rc=$rc ---"
exit $rc
