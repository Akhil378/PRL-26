#!/bin/bash -l
#SBATCH --job-name=panel
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# One more identity judge from outside the ArcFace lineage, over every image set
# tools/dump_embeddings.py saved. See tools/score_panel.py.
#
#   sbatch.tinygpu slurm/score_panel.sbatch sface      # infu env
#   sbatch.tinygpu slurm/score_panel.sbatch vggface    # infu env, GPU
#   sbatch.tinygpu slurm/score_panel.sbatch dlib       # dlib env (conda-forge dlib)

unset SLURM_EXPORT_ENV
set -uo pipefail
JUDGE="${1:?usage: score_panel.sbatch <sface|vggface|dlib>}"

module load python/3.12-conda
if [ "$JUDGE" = "dlib" ]; then conda activate "$WORK/envs/dlib"; else conda activate "$WORK/envs/infu"; fi
export PRL="$WORK/prl26"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1
python tools/score_panel.py --judge "$JUDGE" --emb "$PRL/results/embeddings" \
  --images "$PRL/results/images" --models "$WORK/facejudges" --out "$PRL/results/panel"
echo "exit: $?"
