#!/bin/bash -l
#SBATCH --job-name=calib
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# Part 2 calibration control: the ArcFace domain-shift floor.
#
#   sbatch.tinygpu slurm/gen_calib.sbatch          # all 96 cells
#   sbatch.tinygpu slurm/gen_calib.sbatch 4        # 4 cells, plumbing probe
#
# No array: 96 cells at roughly 18 s each is about 30 minutes. img2img runs only
# strength x num_steps of the schedule, so the 0.3 arm costs about 9 steps
# rather than 30 and the set is cheaper than its image count suggests.
#
# Runs in the `infu` env: diffusers 0.31.0 ships FluxImg2ImgPipeline and the
# FLUX snapshot is already cached, so no new environment was created for this.

unset SLURM_EXPORT_ENV

N="${1:-0}"

module load python/3.12-conda
conda activate "$WORK/envs/infu"

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

OUT="$PRL/results/images/calib"
python src/gen_calib.py \
  --manifest benchmark/manifest_calib.parquet \
  --out "$OUT" \
  --cpu-offload \
  --limit "$N"
rc=$?

echo "--- calibration result ---"
ls "$OUT"/*.png 2>/dev/null | wc -l
if [ -f "$OUT/log_shard0.csv" ]; then tail -5 "$OUT/log_shard0.csv"; fi

# Exit with the GENERATOR's status. Job 1789452 crashed in python and still
# reported COMPLETED 0:0 because a trailing reporting command succeeded after it.
exit $rc
