#!/usr/bin/env sh
# Start the local VLM service (MLX on Mac, or set VLM_BACKEND=trt on Jetson).
# Usage: ./scripts/run-vlm-service.sh [port]
# Env: VLM_BACKEND=mlx|trt, VLM_PORT=5000, VLM_HOST=127.0.0.1, TRTLLM_MODEL_PATH (Jetson)

# Run from repo root so python -m vlm_service finds the package
cd "$(dirname "$0")/.." || exit 1
export VLM_PORT="${1:-${VLM_PORT:-5000}}"
export VLM_HOST="${VLM_HOST:-127.0.0.1}"
export VLM_BACKEND="${VLM_BACKEND:-mlx}"
echo "VLM service: $VLM_HOST:$VLM_PORT backend=$VLM_BACKEND"
exec python -m vlm_service
