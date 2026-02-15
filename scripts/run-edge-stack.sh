#!/usr/bin/env sh
# Start VLM service in background, then Node backend (for local edge pipeline).
# Usage: ./scripts/run-edge-stack.sh
#
# Env:
#   VLM_BACKEND   mlx (Mac, default) | nanollm (Jetson NanoLLM) | trt (Jetson TensorRT-LLM)
#   VLM_MODE      local (auto-set by this script)
#   VLM_LOCAL_URL http://127.0.0.1:5000 (default)
#   OPENAI_API_KEY Required for world-context
#
# Jetson NanoLLM-specific:
#   VLM_MODEL_PATH        Model repo or local path
#   NANOLLM_API           mlc (default) or trt
#   NANOLLM_QUANTIZATION  q4f16_ft (default)
#   MAX_FRAME_DIM         768 (default)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VLM_PORT="${VLM_PORT:-5000}"
VLM_LOCAL_URL="${VLM_LOCAL_URL:-http://127.0.0.1:5000}"
export VLM_PORT VLM_LOCAL_URL
export VLM_MODE=local
export VLM_BACKEND="${VLM_BACKEND:-mlx}"

# For Jetson NanoLLM, bind to 0.0.0.0 by default so the HUD can connect from LAN.
if [ "$VLM_BACKEND" = "nanollm" ]; then
  export VLM_HOST="${VLM_HOST:-0.0.0.0}"
fi

echo "Starting VLM service on port $VLM_PORT (backend=$VLM_BACKEND)..."
python -m vlm_service &
VLM_PID=$!
trap 'kill $VLM_PID 2>/dev/null' EXIT

# Wait for service to be up (NanoLLM model loading can take longer; wait up to 120s)
MAX_WAIT=120
if [ "$VLM_BACKEND" = "mlx" ]; then MAX_WAIT=30; fi
echo "Waiting up to ${MAX_WAIT}s for VLM service..."
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
  if curl -s "${VLM_LOCAL_URL}/health" >/dev/null 2>&1; then break; fi
  sleep 2
  elapsed=$((elapsed + 2))
done
if ! curl -s "${VLM_LOCAL_URL}/health" >/dev/null 2>&1; then
  echo "VLM service did not become ready after ${MAX_WAIT}s"
  exit 1
fi
echo "VLM service ready."

cd "$ROOT/helmet-backend"
export VLM_LOCAL_URL
node server.js
