#!/usr/bin/env bash
# Serves thinkingmachines/Inkling-2-Mini on one GB10 with vLLM on 127.0.0.1:$PORT.
# usage: serve.sh [dflash2|q8|humming|stock]
#   dflash2 (default): Humming NVFP4 experts + DFlash2 drafter with draft-only INT4 (head) / INT8 (body) weights
#   q8:      Humming NVFP4 experts + bundled DSpark drafter with INT8 draft-only weights + wide BF16 GEMV tiles
#   humming: Humming NVFP4 experts, bundled DSpark drafter, stock vLLM modules
#   stock:   the image's default expert backend, bundled DSpark drafter, stock vLLM modules
# Environment overrides: MODEL_DIR, DRAFT_DIR, PORT, GPU_UTIL, MAX_LEN, MAX_SEQS, CACHE_DIR, CONTAINER,
#   and for dflash2: DFLASH2_HEAD_QUANT, DFLASH2_BODY_QUANT (off|int8|int4_g64|int4_g128|nvfp4|hybrid)
set -euo pipefail

MODE="${1:-dflash2}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="vllm/vllm-openai@sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648"
REV="ed40e7da625f594fc9c4990f5f66be24873c22d7"
DRAFT_REV="4f9088ea66c21f2b99ed657d1941c306b03e84d2"
MODEL="${MODEL_DIR:-$HOME/models/Inkling-2-Mini-${REV:0:7}}"
DRAFT="${DRAFT_DIR:-$HOME/models/Inkling-2-Mini-DFlash2-${DRAFT_REV:0:7}}"
CACHE="${CACHE_DIR:-$HOME/.cache/inkling2-mini-vllm}"
PORT="${PORT:-8020}"
CONTAINER="${CONTAINER:-inkling2-mini-gb10}"
SERVED="thinkingmachines/Inkling-2-Mini"
PKG="/usr/local/lib/python3.14/dist-packages"

# Overlay files: repo path, path inside the image's site-packages, pinned SHA-256 (checked before every launch).
Q8_OVERLAY=(
  "overlay/vllm/models/inkling2/nvidia/dspark.py:vllm/models/inkling2/nvidia/dspark.py:6eb1e24f9d305674721440c533ce6730fcedf67f433c7d1458e0a4aab8586464"
  "overlay/vllm/models/inkling2/nvidia/ops/gemv.py:vllm/models/inkling2/nvidia/ops/gemv.py:a95c674196e994da456aec19ef777a25de6ea9206ef3d27b5e3039057ef16b02"
  "overlay/vllm/models/inkling2/nvidia/ops/gemv_q8.py:vllm/models/inkling2/nvidia/ops/gemv_q8.py:b3deb328d4106d5f53a958b13f838a77eb637a949fb77817ab5c25e5820fa2ba"
)
DFLASH2_OVERLAY=(
  "overlay/dflash2/vllm/models/inkling2/nvidia/ops/gemv.py:vllm/models/inkling2/nvidia/ops/gemv.py:9fac49b224ff4e2d9593dfe0eba97f4bb288a27de04c4229f53ae23561b45117"
  "overlay/vllm/models/inkling2/nvidia/ops/gemv_q8.py:vllm/models/inkling2/nvidia/ops/gemv_q8.py:b3deb328d4106d5f53a958b13f838a77eb637a949fb77817ab5c25e5820fa2ba"
  "overlay/dflash2/vllm/models/inkling2/nvidia/ops/gemv_q4.py:vllm/models/inkling2/nvidia/ops/gemv_q4.py:682f822a6803ef84379871f4281d24647c0abe177b8220da38e35c20768fafa8"
  "overlay/dflash2/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:2c29e2412419aaaa0572b81a8fc32f7197f08c983861b0b2062508855cdf0060"
  "overlay/dflash2/vllm/v1/worker/gpu/spec_decode/dflash/utils.py:vllm/v1/worker/gpu/spec_decode/dflash/utils.py:3cb949afb98772ed8ff3bb6afe9ffb9896a5e8b65cba3cd322406120af324bc2"
  "overlay/dflash2/vllm/model_executor/models/qwen3_dflash2.py:vllm/model_executor/models/qwen3_dflash2.py:f31c2792c23ba34e126e0855830e570acadfba49634575e56b668180105b51bd"
)

[[ -f "$MODEL/model-00001-of-00001.safetensors" && -f "$MODEL/mtp.safetensors" ]] ||
  { echo "missing checkpoint in $MODEL (run scripts/download.py)" >&2; exit 2; }
mkdir -p "$CACHE"
# The container runs as root with every capability dropped (no DAC override), so the cache must be world-writable.
chmod a+rwx "$CACHE"

extra=()
mount_overlay() {
  local entry rel dst want got
  for entry in "$@"; do
    IFS=: read -r rel dst want <<<"$entry"
    got="$(sha256sum "$HERE/$rel" | cut -d' ' -f1)"
    [[ "$got" == "$want" ]] || { echo "overlay $rel changed: $got != $want" >&2; exit 2; }
    extra+=(--mount "type=bind,src=$HERE/$rel,dst=$PKG/$dst,readonly")
  done
}

args=(
  serve /model
  --served-model-name "$SERVED"
  --tensor-parallel-size 1
  --dtype bfloat16
  --max-model-len "${MAX_LEN:-262144}"
  --max-num-seqs "${MAX_SEQS:-8}"
  --max-num-batched-tokens 4096
  --reasoning-parser inkling
  --enable-auto-tool-choice
  --tool-call-parser inkling
  --limit-mm-per-prompt '{"image":262144,"video":0}'
  --skip-mm-profiling
  --safetensors-load-strategy eager
  --host 127.0.0.1
  --port "$PORT"
)
DSPARK='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
case "$MODE" in
  dflash2)
    [[ -f "$DRAFT/model.safetensors" && -f "$DRAFT/config.json" ]] ||
      { echo "missing DFlash2 draft in $DRAFT (run scripts/download.py --dflash2)" >&2; exit 2; }
    mount_overlay "${DFLASH2_OVERLAY[@]}"
    extra+=(--mount "type=bind,src=$DRAFT,dst=/draft,readonly")
    extra+=(-e "DFLASH2_HEAD_QUANT=${DFLASH2_HEAD_QUANT:-int4_g128}" -e "DFLASH2_BODY_QUANT=${DFLASH2_BODY_QUANT:-int8}")
    extra+=(-e DFLASH2_SKIP_EXPIRED_CTX=1 -e DFLASH2_FUSED_OPS=1)
    args+=(--moe-backend humming --gpu-memory-utilization "${GPU_UTIL:-0.65}")
    args+=(--speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":7,"draft_sample_method":"probabilistic","kv_cache_dtype":"bfloat16"}')
    ;;
  q8)
    mount_overlay "${Q8_OVERLAY[@]}"
    extra+=(-e INKLING2_DRAFT_QUANT=int8 -e INKLING2_DRAFT_QUANT_LAYERS=1)
    args+=(--moe-backend humming --gpu-memory-utilization "${GPU_UTIL:-0.61}" --speculative-config "$DSPARK")
    ;;
  humming) args+=(--moe-backend humming --gpu-memory-utilization "${GPU_UTIL:-0.61}" --speculative-config "$DSPARK") ;;
  stock) args+=(--gpu-memory-utilization "${GPU_UTIL:-0.61}" --speculative-config "$DSPARK") ;;
  *) echo "unknown mode: $MODE (expected dflash2, q8, humming or stock)" >&2; exit 2 ;;
esac

docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

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
