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
# conda's default package cache is /apps/python/3.12-conda/pkgs, which is
# read-only, so every `conda create` dies with NoWritablePkgsDirError.
# Setting CONDA_PKGS_DIRS is NOT enough: the python/3.12-conda modulefile
# PREPENDS the read-only path to it, so whatever is exported before
# `module load` gets clobbered. Configure it in ~/.condarc instead, which no
# modulefile touches.
mkdir -p "$PRL" "$HF_HOME" "$INSIGHTFACE_HOME" "$WORK/envs" "$CONDA_PKGS_DIRS" logs

echo "=== 0. persist the cache locations ==="
# ~60 GB of weights; $HOME is 100 GB and also holds the conda envs
grep -q 'export HF_HOME' ~/.bashrc || cat >> ~/.bashrc << 'RC'

# --- PRL 2026 project ---
export PRL="$WORK/prl26"
export HF_HOME="$WORK/hf"
export HUGGINGFACE_HUB_CACHE="$WORK/hf/hub"
export INSIGHTFACE_HOME="$WORK/insightface"
export PATH="$WORK/envs/tools/bin:$PATH"
RC

# Login shells read .bash_profile, not .bashrc, and "#!/bin/bash -l" batch
# scripts are login shells. Bridge them so both see the same environment.
[ -f ~/.bash_profile ] || cat > ~/.bash_profile << "BP"
[ -f ~/.bashrc ] && . ~/.bashrc
BP

echo "=== 1. upstream repositories ==="
[ -d "$PRL/InfiniteYou" ] || git clone --depth 1 https://github.com/bytedance/InfiniteYou.git "$PRL/InfiniteYou"
[ -d "$PRL/PuLID" ]       || git clone --depth 1 https://github.com/ToTheBeginning/PuLID.git   "$PRL/PuLID"

module load python/3.12-conda
unset CONDA_PKGS_DIRS          # let ~/.condarc win over the modulefile's prepend

if ! grep -q "pkgs_dirs" ~/.condarc 2>/dev/null; then
  conda config --add pkgs_dirs "$WORK/conda/pkgs"
  conda config --add envs_dirs "$WORK/envs"
fi
echo "pkgs_dirs -> $(conda config --show pkgs_dirs | tail -1)"

echo "=== 2. environments ==="
# small frontend env: huggingface-cli for the gated login, available before the
# heavy envs exist
if [ ! -d "$WORK/envs/tools" ]; then
  conda create -y -q -p "$WORK/envs/tools" python=3.12
  conda run -p "$WORK/envs/tools" pip install -q "huggingface_hub==0.28.1"
fi
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
# ~90 GB over a link that drops. A single ChunkedEncodingError once killed this
# whole script via set -e, after 54 GB had already landed. Retry instead, and
# never let one repo abort the rest: huggingface-cli resumes from .incomplete
# blobs, so a retry costs only the bytes actually lost.
FAILED_REPOS=""
hf_get() {
  local env="$1" repo="$2"; shift 2
  for attempt in 1 2 3 4 5; do
    if conda run -p "$WORK/envs/$env" huggingface-cli download "$repo" "$@"; then
      echo "  ok: $repo"
      return 0
    fi
    echo "  attempt $attempt/5 failed for $repo; retrying in 20s"
    sleep 20
  done
  echo "  GIVING UP on $repo"
  FAILED_REPOS="$FAILED_REPOS $repo"
  return 0            # keep going; reported in the summary
}

# FLUX.1-dev is gated: accept the licence and `huggingface-cli login` first.
hf_get infu    black-forest-labs/FLUX.1-dev --exclude "*.gguf"
hf_get infu    ByteDance/InfiniteYou
hf_get metrics openai/clip-vit-large-patch14
hf_get metrics openai/clip-vit-base-patch32
hf_get metrics yuvalkirstain/PickScore_v1
hf_get metrics laion/CLIP-ViT-H-14-laion2B-s32B-b79K

echo "=== 4. insightface packs ==="
conda run -p "$WORK/envs/metrics" python - << 'PY'
from insightface.app import FaceAnalysis
for pack in ("antelopev2", "buffalo_l"):
    FaceAnalysis(name=pack, providers=["CPUExecutionProvider"]).prepare(ctx_id=-1)
    print("ready:", pack)
PY

echo
if [ -n "$FAILED_REPOS" ]; then
  echo "!!! these repos did not complete:$FAILED_REPOS"
  echo "!!! re-run this script; downloads resume from where they stopped."
fi
echo
echo "=== disk used on \$WORK ==="
du -sh "$WORK/hf" "$WORK/envs" "$PRL" 2>/dev/null || true
echo
echo "Next: sbatch.tinygpu slurm/smoke.sbatch quantize \"--quantize-8bit\""
echo "See slurm/SUBMITTING.md for the full order of operations."
