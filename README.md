# Spartan

Helmet HUD with real-time vision (VLM) and world-context pipeline. Runs in browser + Node backend; VLM can be cloud (OpenAI) or local (Qwen on Mac/Jetson).

DevPost: https://devpost.com/software/spartan-edge-ai-situational-helmet-for-first-responders

## Project layout

| Directory | Description |
|-----------|-------------|
| **helmet-hud** | Vite + React HUD (2560×1440, camera, corner panels, WebSocket state). |
| **helmet-backend** | Node server: WebSocket `/ws/state`, frame ingestion, VLM + world-context, state broadcast. |
| **vlm_service** | Python VLM service (FastAPI): local Qwen via MLX (Mac) or TensorRT-LLM (Jetson). |
| **docs** | Deployment and hardware docs (e.g. Jetson). |
| **scripts** | Shell scripts for starting the edge stack. |

## Edge AI pipeline

Two modes:

1. **OpenAI only** (default): Backend uses OpenAI Vision (gpt-4o-mini) for VLM and GPT-5 mini for world context. Set `OPENAI_API_KEY` in `helmet-backend/.env`.
2. **Local Qwen**: Run the Python VLM service (MLX on Mac or TensorRT-LLM on Jetson), then the backend with `VLM_MODE=local` and `VLM_LOCAL_URL` pointing at the service. No OpenAI key needed for VLM; world context still uses OpenAI.

### Local Qwen (Mac)

```bash
# Terminal 1: VLM service (MLX)
cd vlm_service && pip install -r requirements.txt && python -m vlm_service

# Terminal 2: Backend
cd helmet-backend && VLM_MODE=local VLM_LOCAL_URL=http://127.0.0.1:5000 npm start

# Terminal 3: HUD
cd helmet-hud && VITE_WS_URL=http://localhost:8765 npm run dev
```

Open `http://localhost:5173/?source=camera`.

### Jetson Orin Nano Super

See **[docs/JETSON_DEPLOY.md](docs/JETSON_DEPLOY.md)** for JetPack, TensorRT-LLM, Qwen2-VL setup, env vars, and run order.

## Quick test (no browser)

From `helmet-backend`: `node test-one-frame.mjs` (sends one frame, prints VLM + world; backend and optionally VLM service must be running).
