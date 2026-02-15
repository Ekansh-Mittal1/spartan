# Jetson Orin Nano Super Deployment

Deploy the edge AI pipeline (local Qwen VLM + Node backend + HUD) on Jetson Orin Nano Super. Camera and HUD run in the browser on the device; VLM runs locally via TensorRT-LLM.

## Prerequisites

- **JetPack 6.x** (recommended for TensorRT-LLM on Jetson).
- **TensorRT-LLM for Jetson**: Use the Jetson-specific branch or wheels (e.g. `v0.12.0-jetson` or later). See [TensorRT-LLM for Jetson](https://forums.developer.nvidia.com/t/tensorrt-llm-for-jetson/313227) and [Jetson AI Lab](https://jetson-ai-lab.com/tensorrt_llm.html).
- **Qwen2-VL / Qwen2.5-VL**: Build or download a TensorRT-LLM engine for a Qwen vision-language model (e.g. Qwen2-VL-2B-Instruct). NVIDIA docs and TensorRT-LLM release notes describe VL model support and conversion from Hugging Face checkpoints.

## Environment

| Variable | Description |
|---------|-------------|
| `VLM_BACKEND` | `trt` on Jetson (use `mlx` only on Mac). |
| `TRTLLM_MODEL_PATH` | Path to the TensorRT-LLM Qwen2-VL engine or checkpoint directory. |
| `VLM_MODEL_PATH` | Optional; used by MLX backend (ignore when `VLM_BACKEND=trt`). |
| `VLM_HOST` / `VLM_PORT` | VLM service bind address and port (default `127.0.0.1:5000`). |

## Run order

1. **VLM service** (Python, local Qwen via TensorRT-LLM):
   ```bash
   export VLM_BACKEND=trt
   export TRTLLM_MODEL_PATH=/path/to/your/qwen2vl/engine
   python -m vlm_service
   ```
   The service listens on `http://127.0.0.1:5000` by default.

2. **Node backend** (orchestrator, world context via OpenAI):
   ```bash
   cd helmet-backend
   export VLM_MODE=local
   export VLM_LOCAL_URL=http://127.0.0.1:5000
   export OPENAI_API_KEY=your_key   # for world-context only
   npm start
   ```

3. **HUD** (build and serve; or dev server):
   ```bash
   cd helmet-hud
   export VITE_WS_URL=http://<jetson-ip>:8765
   npm run build && npx serve -s dist -l 5173
   # or: npm run dev
   ```

4. **Browser on Jetson**: Open `http://<jetson-ip>:5173/?source=camera`, allow camera. Frames are sent to the backend; VLM runs locally, world context from OpenAI; HUD shows `vlm_text` and `world_context`.

## TensorRT-LLM backend implementation

The repo includes a **stub** for the TensorRT-LLM backend in `vlm_service/backends/trt_backend.py`. You must implement `load()` and `infer()` using the TensorRT-LLM Python API for Qwen2-VL on your JetPack/TensorRT-LLM version. Typical steps:

- In `load()`: Create a session or load the engine from `TRTLLM_MODEL_PATH`.
- In `infer()`: Run the vision encoder + LLM with the image (file path or bytes) and prompt; return the generated text.

Refer to NVIDIA TensorRT-LLM and Jetson AI Lab documentation for the exact VL API and model conversion (Hugging Face → TensorRT-LLM checkpoint/engine).

## Optional: one-command start

Use `scripts/run-edge-stack.sh` to start the VLM service and Node backend together (see repo polish section).
