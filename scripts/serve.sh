#!/usr/bin/env bash
# Serves thinkingmachines/Inkling-2-Mini on one GB10 with vLLM + DSpark on 127.0.0.1:$PORT.
# usage: serve.sh [q8|humming|stock]
#   q8 (default): Humming NVFP4 experts + INT8 draft-only weights + wide BF16 GEMV tiles (this recipe)
#   humming:      Humming NVFP4 experts, stock vLLM modules
#   stock:        the image's default expert backend, stock vLLM modules
# Environment overrides: MODEL_DIR, PORT, GPU_UTIL, MAX_LEN, MAX_SEQS, CACHE_DIR, CONTAINER
set -euo pipefail

MODE="${1:-q8}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="vllm/vllm-openai@sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648"
REV="ed40e7da625f594fc9c4990f5f66be24873c22d7"
MODEL="${MODEL_DIR:-$HOME/models/Inkling-2-Mini-${REV:0:7}}"
CACHE="${CACHE_DIR:-$HOME/.cache/inkling2-mini-vllm}"
PORT="${PORT:-8020}"
CONTAINER="${CONTAINER:-inkling2-mini-gb10}"
SERVED="thinkingmachines/Inkling-2-Mini"
SRC="/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2/nvidia"

# Overlay files and their pinned SHA-256 (checked before every launch).
OVERLAY=(
  "dspark.py:6eb1e24f9d305674721440c533ce6730fcedf67f433c7d1458e0a4aab8586464"
  "ops/gemv.py:a95c674196e994da456aec19ef777a25de6ea9206ef3d27b5e3039057ef16b02"
  "ops/gemv_q8.py:b3deb328d4106d5f53a958b13f838a77eb637a949fb77817ab5c25e5820fa2ba"
)

[[ -f "$MODEL/model-00001-of-00001.safetensors" && -f "$MODEL/mtp.safetensors" ]] ||
  { echo "missing checkpoint in $MODEL (run scripts/download.py)" >&2; exit 2; }
mkdir -p "$CACHE"
# The container runs as root with every capability dropped (no DAC override), so the cache must be world-writable.
chmod a+rwx "$CACHE"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

extra=()
args=(
  serve /model
  --served-model-name "$SERVED"
  --tensor-parallel-size 1
  --dtype bfloat16
  --max-model-len "${MAX_LEN:-262144}"
  --gpu-memory-utilization "${GPU_UTIL:-0.61}"
  --max-num-seqs "${MAX_SEQS:-8}"
  --max-num-batched-tokens 4096
  --reasoning-parser inkling
  --enable-auto-tool-choice
  --tool-call-parser inkling
  --limit-mm-per-prompt '{"image":262144,"video":0}'
  --skip-mm-profiling
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
  --host 127.0.0.1
  --port "$PORT"
)
case "$MODE" in
  stock) ;;
  humming) args+=(--moe-backend humming) ;;
  q8)
    args+=(--moe-backend humming)
    extra+=(-e INKLING2_DRAFT_QUANT=int8 -e INKLING2_DRAFT_QUANT_LAYERS=1)
    for entry in "${OVERLAY[@]}"; do
      rel="${entry%%:*}"; want="${entry##*:}"
      file="$HERE/overlay/vllm/models/inkling2/nvidia/$rel"
      got="$(sha256sum "$file" | cut -d' ' -f1)"
      [[ "$got" == "$want" ]] || { echo "overlay $rel changed: $got != $want" >&2; exit 2; }
      extra+=(--mount "type=bind,src=$file,dst=$SRC/$rel,readonly")
    done
    ;;
  *) echo "unknown mode: $MODE (expected q8, humming or stock)" >&2; exit 2 ;;
esac

exec docker run --rm \
  --name "$CONTAINER" \
  --label inkling2-mini.mode="$MODE" \
  --gpus all \
  --network host \
  --ipc host \
  --init \
  --memory 104g \
  --memory-swap 104g \
  --ulimit memlock=-1 \
  --ulimit core=0 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --mount "type=bind,src=$MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$CACHE,dst=/cache" \
  -e MAX_JOBS=4 -e NVCC_THREADS=2 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e XDG_CACHE_HOME=/cache -e HF_HOME=/cache/hf \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e TRITON_CACHE_DIR=/cache/triton \
  -e CUDA_CACHE_PATH=/cache/cuda -e FLASHINFER_WORKSPACE_BASE=/cache -e VLLM_CACHE_ROOT=/cache/vllm \
  "${extra[@]}" \
  --entrypoint vllm \
  "$IMAGE" \
  "${args[@]}"
