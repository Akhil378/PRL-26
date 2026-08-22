#!/bin/bash -l
# One-time cluster setup. Run on the tinyx FRONTEND, not inside a GPU job:
# downloads are network-bound and paying A100 time to wait on them is waste.
#
#   ssh prl && cd ~/prl26-infiniteyou && bash tools/setup_cluster.sh
#
# Safe to re-run; every step is idempotent.

set -euo pipefail

export PRL="$WORK/prl26"
export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
mkdir -p "$PRL" "$HF_HOME" "$INSIGHTFACE_HOME" "$WORK/envs" logs

echo "=== 0. persist the cache locations ==="
# ~60 GB of weights; $HOME is 100 GB and also holds the conda envs
grep -q 'export HF_HOME' ~/.bashrc || cat >> ~/.bashrc << 'RC'

# --- PRL 2026 project ---
export PRL="$WORK/prl26"
export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
RC

echo "=== 1. upstream repositories ==="
[ -d "$PRL/InfiniteYou" ] || git clone --depth 1 https://github.com/bytedance/InfiniteYou.git "$PRL/InfiniteYou"
[ -d "$PRL/PuLID" ]       || git clone --depth 1 https://github.com/ToTheBeginning/PuLID.git   "$PRL/PuLID"

module load python/3.12-conda

echo "=== 2. environments ==="
if [ ! -d "$WORK/envs/infu" ]; then
  conda create -y -p "$WORK/envs/infu" python=3.11
  conda run -p "$WORK/envs/infu" pip install -r "$PRL/InfiniteYou/requirements.txt"
  conda run -p "$WORK/envs/infu" pip install pandas pyarrow
fi
if [ ! -d "$WORK/envs/metrics" ]; then
  conda create -y -p "$WORK/envs/metrics" python=3.11
  conda run -p "$WORK/envs/metrics" pip install torch torchvision \
      "transformers>=4.48,<5" insightface==0.7.3 onnxruntime-gpu opencv-python \
      pillow pandas pyarrow scipy matplotlib
fi
if [ ! -d "$WORK/envs/pulid" ]; then
  conda create -y -p "$WORK/envs/pulid" python=3.11
  conda run -p "$WORK/envs/pulid" pip install -r "$PRL/PuLID/requirements.txt"
  conda run -p "$WORK/envs/pulid" pip install pandas pyarrow
fi

echo "=== 3. weights ==="
# FLUX.1-dev is gated: accept the licence and `huggingface-cli login` first.
conda run -p "$WORK/envs/infu" huggingface-cli download black-forest-labs/FLUX.1-dev --exclude "*.gguf"
conda run -p "$WORK/envs/infu" huggingface-cli download ByteDance/InfiniteYou
conda run -p "$WORK/envs/metrics" huggingface-cli download openai/clip-vit-large-patch14
conda run -p "$WORK/envs/metrics" huggingface-cli download openai/clip-vit-base-patch32
conda run -p "$WORK/envs/metrics" huggingface-cli download yuvalkirstain/PickScore_v1
conda run -p "$WORK/envs/metrics" huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K

echo "=== 4. insightface packs ==="
conda run -p "$WORK/envs/metrics" python - << 'PY'
from insightface.app import FaceAnalysis
for pack in ("antelopev2", "buffalo_l"):
    FaceAnalysis(name=pack, providers=["CPUExecutionProvider"]).prepare(ctx_id=-1)
    print("ready:", pack)
PY

echo
echo "=== disk used on \$WORK ==="
du -sh "$WORK/hf" "$WORK/envs" "$PRL" 2>/dev/null || true
echo
echo "Next: sbatch.tinygpu slurm/smoke.sbatch quantize \"--quantize-8bit\""
echo "See slurm/SUBMITTING.md for the full order of operations."
