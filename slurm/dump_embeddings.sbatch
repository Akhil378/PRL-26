#!/bin/bash -l
#SBATCH --job-name=dump-emb
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# Face and CLIP embeddings of every image, kept so that analysis survives the
# end of cluster access (1 Oct 2026). See tools/dump_embeddings.py.
#
#   sbatch.tinygpu slurm/dump_embeddings.sbatch "infu_aes2:manifest_repro floor ceiling" --refs

unset SLURM_EXPORT_ENV
set -uo pipefail
SETS="${1:?usage: dump_embeddings.sbatch \"<set[:manifest]> ...\" [--refs]}"
shift

module load python/3.12-conda
conda activate "$WORK/envs/infu"

export HF_HOME="$WORK/hf"
export INSIGHTFACE_HOME="$WORK/insightface"
export TORCH_HOME="$WORK/torch"
export PRL="$WORK/prl26"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1
python tools/dump_embeddings.py --sets $SETS --images "$PRL/results/images" \
  --out "$PRL/results/embeddings" --insightface-root "$INSIGHTFACE_HOME" "$@"
echo "exit: $?"
