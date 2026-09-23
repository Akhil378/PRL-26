#!/bin/bash -l
#SBATCH --job-name=score-irse
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# ID Loss under InfU's own encoder (facexlib IR-SE50), for Part 1 both methods
# and the authors'-portraits control. See tools/score_infu_encoder.py for why:
# the primary ID Loss uses antelopev2's glintr100, which is PuLID's encoder and
# not InfU's.
#
#   sbatch.tinygpu slurm/score_irse50.sbatch

unset SLURM_EXPORT_ENV
set -uo pipefail

module load python/3.12-conda
conda activate "$WORK/envs/infu"

export HF_HOME="$WORK/hf"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1

rc=0
# SETS and FLOOR select the work; the defaults are the Part 1 run.
SETS="${SETS:-infu_aes2:manifest_repro pulid:manifest_repro paperid_infu:manifest_paperid}"
for spec in $SETS; do
  set -- ${spec/:/ }
  python tools/score_infu_encoder.py \
    --manifest "benchmark/$2.parquet" \
    --images   "$PRL/results/images/$1" \
    --method   "$1" \
    --insightface-root "$INSIGHTFACE_HOME" \
    --out      "$PRL/results/scores/irse50_$1_$2.csv" || rc=$?
done
if [ "${FLOOR:-0}" = "1" ]; then
  python tools/score_infu_encoder.py --floor --manifest none --method floor --images "$PRL/results/images/floor" --insightface-root "$INSIGHTFACE_HOME" --out "$PRL/results/scores/irse50_floor.csv" || rc=$?
fi
echo "scorer exit: $rc"
exit $rc
