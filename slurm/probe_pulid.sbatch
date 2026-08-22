#!/bin/bash -l
#SBATCH --job-name=probe-pulid
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# Step 3 of the verification ladder: prove PuLID-FLUX can produce ONE image
# before spending anything on the pilot or the array.
#
# This is deliberately throwaway -- unlike pilot.sbatch it writes to its own
# directory, not the real Part 1 output. The point is to shake out the plumbing
# (weights staged, API signature right, id_image type right), and settings may
# still change afterwards; images generated under settings we then revise would
# be indistinguishable from good ones in the real output dir.
#
# It exercises everything the InfU smoke test could not:
#   - PuLID's own T5 (xlabs-ai), which is what killed job 1789452
#   - EVA-CLIP, facexlib bisenet parsing, and PuLID's antelopev2 copy
#   - the FluxGenerator signature, including pretrained_model/version
#   - the id_image contract (numpy RGB, not PIL)
#
#   sbatch.tinygpu slurm/probe_pulid.sbatch
#   sbatch.tinygpu slurm/probe_pulid.sbatch 4        # 4 images instead of 2
#
# 45 minutes is generous for 2 images, but PuLID's pipeline load is unmeasured
# and a walltime kill mid-load would teach us nothing. The pilot will size the
# array properly.

unset SLURM_EXPORT_ENV

N="${1:-2}"

module load python/3.12-conda
conda activate "$WORK/envs/pulid"

export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

# Compute nodes have NO outbound internet; offline mode makes the hub resolve
# from cache and fail fast rather than hanging on connect retries.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export NO_ALBUMENTATIONS_UPDATE=1

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# PuLID addresses its weights relative to the cwd, so this symlink is load-bearing.
echo "--- models/ resolves to: $(readlink -f models 2>/dev/null || echo 'MISSING') ---"

OUT="$PRL/results/images/probe_pulid"
python src/gen_pulid.py \
  --manifest benchmark/manifest_repro.parquet \
  --out "$OUT" \
  --pulid-repo "$PRL/PuLID" \
  --offload \
  --limit "$N"
rc=$?

echo "--- probe result ---"
ls -la "$OUT"/*.png 2>/dev/null || echo "NO IMAGES PRODUCED"
if [ -f "$OUT/log_shard0.csv" ]; then cat "$OUT/log_shard0.csv"; fi

# Exit with the GENERATOR's status, not the last echo's. Job 1789452 crashed in
# python and still reported State=COMPLETED ExitCode=0:0 to sacct, because a
# trailing reporting command succeeded after it. A failure has to look like one.
exit $rc
