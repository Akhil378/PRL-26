#!/bin/bash -l
#SBATCH --job-name=smoke
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# Phase 1 timing calibration -- run this FIRST, before any array job.
# Generates 20 images under one memory profile and reports median s/image.
# Feed that number into tools/plan_shards.py to size the real runs.
#
#   sbatch.tinygpu slurm/smoke.sbatch quantize   "--quantize-8bit"
#   sbatch.tinygpu slurm/smoke.sbatch offload    "--cpu-offload"
#   sbatch.tinygpu slurm/smoke.sbatch both       "--quantize-8bit --cpu-offload"

unset SLURM_EXPORT_ENV

PROFILE="${1:?usage: smoke.sbatch <profile-name> \"<flags>\"}"
FLAGS="${2:-}"

module load python/3.12-conda
conda activate "$WORK/envs/infu"

export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
export PRL="$WORK/prl26"

cd "${SLURM_SUBMIT_DIR:-$HOME/prl26-infiniteyou}" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Uses the two upstream example portraits, NOT the curated identity set, so
# calibration is not blocked on identity sourcing. Idempotent.
python tools/make_fixtures.py

# shellcheck disable=SC2086
python src/gen_infu.py \
  --manifest benchmark/manifest_repro.parquet \
  --out "$PRL/results/images/smoke_${PROFILE}" \
  --infu-repo "$PRL/InfiniteYou" \
  --insightface-root "$INSIGHTFACE_HOME" \
  --limit 20 $FLAGS

echo "--- median s/image for profile ${PROFILE} ---"
python - <<'PY'
import glob, pandas as pd, os
f = sorted(glob.glob(os.path.expandvars("$PRL/results/images/smoke_*/log_shard0.csv")))
for p in f:
    d = pd.read_csv(p); ok = d[d.status == "ok"]
    if len(ok):
        print(f"{p.split(os.sep)[-2]:<24} n={len(ok):<4} median={ok.sec.median():.1f}s  mean={ok.sec.mean():.1f}s")
PY
