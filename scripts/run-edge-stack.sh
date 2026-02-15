#!/usr/bin/env sh
# Start VLM service in background, then Node backend (for local edge pipeline).
# Usage: ./scripts/run-edge-stack.sh
# Env: VLM_MODE=local, VLM_LOCAL_URL=http://127.0.0.1:5000, OPENAI_API_KEY (for world)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VLM_PORT="${VLM_PORT:-5000}"
VLM_LOCAL_URL="${VLM_LOCAL_URL:-http://127.0.0.1:5000}"
export VLM_PORT VLM_LOCAL_URL
export VLM_MODE=local
export VLM_BACKEND="${VLM_BACKEND:-mlx}"

echo "Starting VLM service on port $VLM_PORT..."
python -m vlm_service &
VLM_PID=$!
trap 'kill $VLM_PID 2>/dev/null' EXIT

# Wait for service to be up
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -s "${VLM_LOCAL_URL}/health" >/dev/null 2>&1; then break; fi
  if [ "$i" -eq 10 ]; then echo "VLM service did not become ready"; exit 1; fi
  sleep 1
done
echo "VLM service ready."

cd "$ROOT/helmet-backend"
export VLM_LOCAL_URL
node server.js
