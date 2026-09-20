#!/bin/bash -l
#SBATCH --job-name=extras
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# Two late controls that the discussion needs:
#   paperid  InfU on the authors' own example portraits, same 200 prompts.
#            Separates "our benchmark is harder" from "our setup differs".
#   ceiling  Text-only FLUX. The upper bound the paper's CLIPScore claim is
#            stated against, measured on this benchmark.
#
# Both run at bf16 with offload, matching the Part 1 arms exactly: a control
# measured at a different precision than the thing it controls is not a control.
#
#   sbatch.tinygpu slurm/gen_extras.sbatch paperid
#   sbatch.tinygpu slurm/gen_extras.sbatch ceiling

unset SLURM_EXPORT_ENV
set -uo pipefail

WHICH="${1:?usage: gen_extras.sbatch <paperid|ceiling>}"

module load python/3.12-conda
conda activate "$WORK/envs/infu"

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

if [ "$WHICH" = "paperid" ]; then
  python src/gen_infu.py \
    --manifest benchmark/manifest_paperid.parquet \
    --out "$PRL/results/images/paperid_infu" \
    --infu-repo "$PRL/InfiniteYou" \
    --insightface-root "$INSIGHTFACE_HOME" \
    --model-version aes_stage2 \
    --cpu-offload
  rc=$?
else
  python src/gen_flux.py \
    --prompts benchmark/prompts_200.json \
    --out "$PRL/results/images/ceiling"
  rc=$?
fi

# sacct reports the LAST command's status, so a trailing echo would mask a
# failure here. Exit on the generator's own code.
echo "generator exit: $rc"
exit $rc
