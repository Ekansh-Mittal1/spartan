#!/usr/bin/env sh
# Start the VLM pipeline using SGLang on Jetson.
#
# This starts TWO processes:
#   1. SGLang inference server (inside jetson-containers, GPU)
#   2. VLM service (on host, thin HTTP proxy that the Node backend talks to)
#
# Usage: ./scripts/run-vlm-sglang.sh
#
# Environment variables:
#   SGLANG_MODEL       HuggingFace model (default: Efficient-Large-Model/NVILA-Lite-2B-hf-0626)
#   SGLANG_PORT        SGLang server port (default: 30000)
#   VLM_PORT           VLM service port (default: 5000)
#   VLM_HOST           VLM service bind address (default: 0.0.0.0)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SGLANG_MODEL="${SGLANG_MODEL:-Efficient-Large-Model/NVILA-Lite-2B-hf-0626}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
VLM_PORT="${VLM_PORT:-5000}"
VLM_HOST="${VLM_HOST:-0.0.0.0}"

echo "=== SGLang VLM Pipeline ==="
echo "  Model:       $SGLANG_MODEL"
echo "  SGLang port: $SGLANG_PORT"
echo "  VLM port:    $VLM_PORT"
echo ""

# ── Step 1: Start SGLang inference server (Docker) ──
echo "Starting SGLang server (this may take a minute to load the model)..."
jetson-containers run \
  $(autotag sglang) \
  python3 -m sglang.launch_server \
    --model-path "$SGLANG_MODEL" \
    --device cuda \
    --dtype half \
    --mem-fraction-static 0.8 \
    --context-length 2048 \
    --port "$SGLANG_PORT" \
    --host 0.0.0.0 &
SGLANG_PID=$!
trap 'kill $SGLANG_PID 2>/dev/null' EXIT

# Wait for SGLang to be ready
echo "Waiting for SGLang server on port $SGLANG_PORT..."
MAX_WAIT=180
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
  if curl -s "http://127.0.0.1:$SGLANG_PORT/health" >/dev/null 2>&1; then break; fi
  sleep 3
  elapsed=$((elapsed + 3))
done
if ! curl -s "http://127.0.0.1:$SGLANG_PORT/health" >/dev/null 2>&1; then
  echo "SGLang server did not become ready after ${MAX_WAIT}s"
  exit 1
fi
echo "SGLang server ready."

# ── Step 2: Start VLM service (thin proxy) ──
echo "Starting VLM service on $VLM_HOST:$VLM_PORT..."
export VLM_BACKEND=sglang
export SGLANG_BASE_URL="http://127.0.0.1:$SGLANG_PORT"
export VLM_HOST
export VLM_PORT
exec python3 -m vlm_service
