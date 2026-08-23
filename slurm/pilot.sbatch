#!/bin/bash -l
#SBATCH --job-name=pilot
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=NONE

# Pilot: one real shard of the real Part 1 run, before committing to the array.
#
# This is deliberately NOT throwaway work. It writes into the SAME output
# directory as the full run using the SAME manifest and seeds, so every image it
# produces counts toward Part 1 -- the subsequent array skips them on resume.
# The cost of de-risking the big run is therefore zero images, only queue time.
#
# It exercises what the 20-image smoke test cannot:
#   - the sharding path with a real --num-shards value
#   - sustained generation over ~50 images (thermal/memory drift, slow leaks)
#   - a per-shard wall-clock number to feed tools/plan_shards.py
#
#   sbatch.tinygpu slurm/pilot.sbatch infu  "--quantize-8bit"
#   sbatch.tinygpu slurm/pilot.sbatch pulid "--offload"
#   sbatch.tinygpu slurm/pilot.sbatch infu  "--cpu-offload" 15 control_infu_bf16
#
# The third form is the 6.2 precision control: the SAME 100 cells as the
# 8-bit arm, hence the same seeds and the same latents, so the quantisation
# effect can be taken as a paired difference instead of argued about.

unset SLURM_EXPORT_ENV

METHOD="${1:?usage: pilot.sbatch <infu|pulid> \"<flags>\"}"
FLAGS="${2:-}"
SHARDS="${3:-30}"        # 1500/30 = 50 cells
# Optional 4th argument: override the output directory leaf. The default
# lands in the REAL Part 1 output, which is the whole point of the pilot --
# its images count. A control arm must NOT do that: regenerating the same
# cells at a different precision would collide with the real run's files and
# be skipped as already-done. Pass a name here for anything that is a
# measurement about the pipeline rather than a contribution to Part 1.
OUTNAME="${4:-}"

module load python/3.12-conda

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

# shellcheck disable=SC2086
if [ "$METHOD" = "infu" ]; then
  conda activate "$WORK/envs/infu"
  python src/gen_infu.py \
    --manifest benchmark/manifest_repro.parquet \
    --out "$PRL/results/images/${OUTNAME:-infu_aes2}" \
    --infu-repo "$PRL/InfiniteYou" \
    --insightface-root "$INSIGHTFACE_HOME" \
    --model-version aes_stage2 \
    --shard 0 --num-shards "$SHARDS" $FLAGS
else
  conda activate "$WORK/envs/pulid"
  python src/gen_pulid.py \
    --manifest benchmark/manifest_repro.parquet \
    --out "$PRL/results/images/${OUTNAME:-pulid}" \
    --pulid-repo "$PRL/PuLID" \
    --shard 0 --num-shards "$SHARDS" $FLAGS
fi
rc=$?

echo "--- pilot timing ---"
python - << 'PY'
import glob, os, pandas as pd
for p in sorted(glob.glob(os.path.expandvars("$PRL/results/images/*/log_shard0.csv"))):
    d = pd.read_csv(p); ok = d[d.status == "ok"]
    if len(ok):
        print(f"{p.split(os.sep)[-2]:<14} n={len(ok):<4} "
              f"median={ok.sec.median():6.1f}s  mean={ok.sec.mean():6.1f}s  "
              f"total={ok.sec.sum()/60:5.1f}min")
    bad = d[d.status != "ok"]
    if len(bad):
        print(f"  {len(bad)} failures:", bad.status.value_counts().to_dict())
PY

# Exit with the GENERATOR's status, not the trailing report's. Job 1789452
# crashed in python yet reported State=COMPLETED ExitCode=0:0 to sacct because
# this reporting block succeeded after it. A failure has to look like one.
exit $rc
