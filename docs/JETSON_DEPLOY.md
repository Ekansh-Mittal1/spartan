# Jetson Orin Nano Super Deployment

Deploy the edge AI pipeline (local VLM + Node backend + HUD) on Jetson Orin Nano Super. Camera and HUD run in the browser; VLM runs locally via **NanoLLM** (recommended) or TensorRT-LLM.

---

## Option A — NanoLLM (recommended)

NanoLLM from the Jetson AI Lab provides optimised inference for vision-language models (VILA, LLaVA, Obsidian, etc.) on Jetson with MLC or TensorRT backends, quantisation, and KV-cache management out of the box.

### Prerequisites

- **JetPack 6.x** (Ubuntu 22.04, CUDA 12.x).
- **NanoLLM** — install via one of:
  - **jetson-containers** (easiest):
    ```bash
    # Pull the pre-built NanoLLM container
    jetson-containers run $(autotag nano_llm)
    ```
  - **From source** (if running outside a container):
    ```bash
    git clone https://github.com/dusty-nv/NanoLLM
    cd NanoLLM && pip install -e .
    ```
- A supported vision model (downloaded automatically on first run):
  - `Efficient-Large-Model/VILA1.5-3b` (default, good speed/quality)
  - `Efficient-Large-Model/VILA-2.5-3b`
  - `liuhaotian/llava-v1.6-vicuna-7b`
  - Any NanoLLM-compatible VLM

### Environment

| Variable | Default | Description |
|---------|---------|-------------|
| `VLM_BACKEND` | — | **Must be `nanollm`** on Jetson with NanoLLM. |
| `VLM_MODEL_PATH` | `Efficient-Large-Model/VILA1.5-3b` | HuggingFace repo ID or local path. |
| `NANOLLM_API` | `mlc` | NanoLLM runtime: `mlc` (MLC-LLM) or `trt` (TensorRT). |
| `NANOLLM_QUANTIZATION` | `q4f16_ft` | Quantisation preset (see NanoLLM docs). |
| `MAX_FRAME_DIM` | `768` | Max pixel dimension; lower = faster inference. |
| `VLM_HOST` / `VLM_PORT` | `0.0.0.0` / `5000` | VLM service bind address and port. |
| `VLM_PROMPT` | `Describe this image briefly.` | Default prompt. |

### Run order

1. **VLM service** (Python, NanoLLM):
   ```bash
   # Quick start with the helper script:
   ./scripts/run-vlm-nanollm.sh

   # Or manually:
   export VLM_BACKEND=nanollm
   export VLM_MODEL_PATH=Efficient-Large-Model/VILA1.5-3b
   python -m vlm_service
   ```
   The service listens on `http://0.0.0.0:5000` by default.

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

4. **Browser**: Open `http://<jetson-ip>:5173/?source=camera`, allow camera. Frames are sent to the backend → NanoLLM VLM runs locally → world context from OpenAI → HUD displays `vlm_text` and `world_context`.

### One-command start (VLM + Node backend)

```bash
export VLM_BACKEND=nanollm
export OPENAI_API_KEY=your_key
./scripts/run-edge-stack.sh
```

This starts the NanoLLM VLM service in the background, waits for it to load, then starts the Node backend.

---

## Option B — TensorRT-LLM

For maximum throughput with pre-built TensorRT engines.

### Prerequisites

- **JetPack 6.x**.
- **TensorRT-LLM for Jetson**: Use the Jetson-specific branch or wheels (e.g. `v0.12.0-jetson` or later). See [TensorRT-LLM for Jetson](https://forums.developer.nvidia.com/t/tensorrt-llm-for-jetson/313227) and [Jetson AI Lab](https://jetson-ai-lab.com/tensorrt_llm.html).
- **Qwen2-VL / Qwen2.5-VL**: Build or download a TensorRT-LLM engine for a Qwen vision-language model.

### Environment

| Variable | Description |
|---------|-------------|
| `VLM_BACKEND` | `trt` on Jetson. |
| `TRTLLM_MODEL_PATH` | Path to the TensorRT-LLM Qwen2-VL engine or checkpoint directory. |
| `VLM_HOST` / `VLM_PORT` | VLM service bind address and port (default `0.0.0.0:5000`). |

### Run order

1. **VLM service** (Python, TensorRT-LLM):
   ```bash
   export VLM_BACKEND=trt
   export TRTLLM_MODEL_PATH=/path/to/your/qwen2vl/engine
   python -m vlm_service
   ```

2–4: Same as Option A (Node backend, HUD, browser).

### TensorRT-LLM backend implementation

The repo includes a **stub** for the TensorRT-LLM backend in `vlm_service/backends/trt_backend.py`. You must implement `load()` and `infer()` using the TensorRT-LLM Python API for Qwen2-VL on your JetPack/TensorRT-LLM version.

---

## Architecture

```
Browser (HUD)          Node Backend          VLM Service (Jetson)
  ┌─────────┐   WS     ┌──────────┐  HTTP   ┌──────────────────┐
  │ React   │◄────────►│ server.js │───────►│ FastAPI + NanoLLM │
  │ helmet- │  frames  │ helmet-  │ /infer │ vlm_service/      │
  │ hud     │  + state │ backend  │        │ backends/         │
  └─────────┘          └──────────┘        │ nanollm_backend.py│
       ▲                    │               └──────────────────┘
       │                    ▼                       │
  Camera                OpenAI API            NanoLLM + VILA
  (getUserMedia)        (world context)       (local GPU inference)
```

The frontend sends camera frames via WebSocket to the Node backend. The Node backend POSTs base64 images to the VLM service's `/infer` endpoint. The VLM service returns `{ "text": "...", "elapsed_s": ... }`. The Node backend then calls OpenAI for world-context and broadcasts the updated HUD state back to the frontend via WebSocket. No frontend changes are needed when switching backends.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'nano_llm'` | Install NanoLLM from source or use the jetson-containers image. |
| Model download hangs | Pre-download: `huggingface-cli download Efficient-Large-Model/VILA1.5-3b` |
| Out of memory | Use a smaller model or lower `MAX_FRAME_DIM` (e.g. 512). |
| VLM service doesn't start | Check `VLM_BACKEND=nanollm` is set. Check NanoLLM install: `python -c "from nano_llm import NanoLLM"`. |
| HUD shows "Disconnected" | Ensure `VITE_WS_URL` points to the Jetson's IP and port 8765. |
| Slow inference | Try `NANOLLM_API=trt` if you have TensorRT engines, or reduce `MAX_FRAME_DIM`. |
