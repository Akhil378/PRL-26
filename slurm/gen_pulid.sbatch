#!/bin/bash -l
#SBATCH --job-name=pulid-gen
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:50:00
#SBATCH --array=0-11
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --export=NONE

# Part 1 generation, PuLID-FLUX baseline.
# Submit with:  sbatch.tinygpu slurm/gen_pulid.sbatch [ID_START] [OUT_NAME]
#   no args        start_step 4 into results/images/pulid (Part 1 as run)
#   0 pulid_s0     start_step 0, PuLID's app default, into pulid_s0
# start_step is PuLID's identity-versus-editability knob: its docs suggest 4
# for photoreal scenes, its app defaults to 0, and the report (Section 5.1) measures
# that the choice moves both ID Loss and CLIPScore by more than the effects
# Part 1 reports.
#
# The generator signature is verified (probe 1790289) and the pilot ran 97 cells
# clean at 35.1 s/img, which is what sizes this array: 12 shards of 125 cells at
# ~1:50, matching gen_infu.sbatch so both methods trickle through the shared GPU
# pool at the same granularity.
#
# bf16 + offload is PuLID's only option -- it has no 8-bit path -- which is what
# forced the decision to run InfU at bf16 too (report, Appendix C).

unset SLURM_EXPORT_ENV

module load python/3.12-conda
conda activate "$WORK/envs/pulid"

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

ID_START="${1:-4}"
OUT_NAME="${2:-pulid}"

python src/gen_pulid.py \
  --manifest benchmark/manifest_repro.parquet \
  --out      "$PRL/results/images/$OUT_NAME" \
  --id-start "$ID_START" \
  --pulid-repo "$PRL/PuLID" \
  --offload \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards 12
