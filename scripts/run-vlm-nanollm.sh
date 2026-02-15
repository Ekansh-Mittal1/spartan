#!/usr/bin/env sh
# Start the local VLM service using NanoLLM on Jetson.
# Usage: ./scripts/run-vlm-nanollm.sh [port]
#
# Environment variables:
#   VLM_PORT              HTTP port (default: 5000)
#   VLM_HOST              Bind address (default: 0.0.0.0 — accessible from LAN)
#   VLM_MODEL_PATH        Model repo or local path (default: Efficient-Large-Model/VILA1.5-3b)
#   NANOLLM_API           NanoLLM runtime: mlc or trt (default: mlc)
#   NANOLLM_QUANTIZATION  Quantization preset (default: q4f16_ft)
#   MAX_FRAME_DIM         Max image dimension sent to model (default: 768)
#   VLM_PROMPT            Default prompt when none supplied per request.

# Run from repo root so python -m vlm_service finds the package
cd "$(dirname "$0")/.." || exit 1

export VLM_BACKEND=nanollm
export VLM_PORT="${1:-${VLM_PORT:-5000}}"
export VLM_HOST="${VLM_HOST:-0.0.0.0}"
export VLM_MODEL_PATH="${VLM_MODEL_PATH:-Efficient-Large-Model/VILA1.5-3b}"
export NANOLLM_API="${NANOLLM_API:-mlc}"
export NANOLLM_QUANTIZATION="${NANOLLM_QUANTIZATION:-q4f16_ft}"
export MAX_FRAME_DIM="${MAX_FRAME_DIM:-768}"

echo "VLM service (NanoLLM): $VLM_HOST:$VLM_PORT"
echo "  model=$VLM_MODEL_PATH  api=$NANOLLM_API  quant=$NANOLLM_QUANTIZATION"
exec python -m vlm_service
