#!/bin/bash -l
#SBATCH --job-name=score
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# Scoring runs on a cheap RTX card while the A100 keeps generating.
#
# Arguments are positional, NOT --export: with --export=NONE the environment of
# the submitting shell is deliberately discarded, so exported variables would
# not survive.
#
#   sbatch.tinygpu slurm/score.sbatch infu_aes2 manifest_repro
#   sbatch.tinygpu slurm/score.sbatch pulid     manifest_style

unset SLURM_EXPORT_ENV

METHOD="${1:?usage: score.sbatch <method> <manifest-stem>}"
MANIFEST="${2:?usage: score.sbatch <method> <manifest-stem>}"

module load python/3.12-conda
conda activate "$WORK/envs/metrics"

export HF_HOME="$WORK/hf"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1

python src/score_all.py \
  --manifest "benchmark/${MANIFEST}.parquet" \
  --images   "$PRL/results/images/${METHOD}" \
  --method   "${METHOD}" \
  --id-root  "$INSIGHTFACE_HOME" \
  --out      "$PRL/results/scores/${METHOD}_${MANIFEST}.parquet"
