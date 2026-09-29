#!/bin/bash -l
#SBATCH --job-name=score-extras
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# The last experiments before cluster access ends (1 Oct 2026). Scoring only,
# no generation; every image already exists.
#   facenet  ID Loss under a non-ArcFace judge, every image set and the floor
#            (tools/score_facenet.py)
#   recover  the faces antelopev2 missed at 640, searched again at full size
#            (tools/recover_missed_faces.py)
#   phrases  every Part 2 image against every style phrase
#            (tools/score_style_phrases.py)
#
#   sbatch.tinygpu slurm/score_extras.sbatch                  # all three
#   sbatch.tinygpu slurm/score_extras.sbatch "recover phrases"

unset SLURM_EXPORT_ENV
set -uo pipefail
STEPS="${1:-facenet recover phrases}"

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
S="$PRL/results/scores"
rc=0

for step in $STEPS; do
  echo "=== $step  $(date +%T)"
  case "$step" in
    phrases)
      python tools/score_style_phrases.py --images "$PRL/results/images" \
        --out "$S/style_phrases.parquet" || rc=$? ;;
    recover)
      python tools/recover_missed_faces.py --scores "$S" --images "$PRL/results/images" \
        --out-dir "$S" --insightface-root "$INSIGHTFACE_HOME" || rc=$? ;;
    facenet)
      for spec in infu_aes2:manifest_repro pulid:manifest_repro pulid_s0:manifest_repro \
                  paperid_infu:manifest_paperid \
                  style_infu:manifest_style style_pulid:manifest_style \
                  sweep_infu_cs06:manifest_sweep sweep_infu_cs08:manifest_sweep \
                  sweep_pulid_iw06:manifest_sweep sweep_pulid_iw08:manifest_sweep; do
        set -- ${spec/:/ }
        python tools/score_facenet.py --manifest "benchmark/$2.parquet" \
          --images "$PRL/results/images/$1" --method "$1" \
          --out "$S/facenet_$1_$2.csv" || rc=$?
      done
      python tools/score_facenet.py --floor --images "$PRL/results/images/floor" \
        --out "$S/facenet_floor.csv" || rc=$? ;;
  esac
done
echo "exit: $rc"
exit $rc
