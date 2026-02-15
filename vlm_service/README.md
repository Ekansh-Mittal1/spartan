# VLM Service

Local VLM for the edge pipeline. Runs as an HTTP service; the Node backend POSTs frames to `/infer` and receives `{ "text", "elapsed_s" }`.

## Backends

- **MLX** (Mac / Apple Silicon): `VLM_BACKEND=mlx`. Uses `mlx-vlm`; default model is `Qwen2.5-VL-3B-Instruct-8bit`. Set `VLM_MODEL_PATH` to override.
- **NanoLLM** (Jetson Orin / Nano): `VLM_BACKEND=nanollm`. Uses NanoLLM with vision-language models (VILA, LLaVA, etc.). Default model is `Efficient-Large-Model/VILA1.5-3b`. See [docs/JETSON_DEPLOY.md](../docs/JETSON_DEPLOY.md).
- **TensorRT-LLM** (Jetson): `VLM_BACKEND=trt`. Set `TRTLLM_MODEL_PATH` to the engine directory. Implement `trt_backend.py` per [docs/JETSON_DEPLOY.md](../docs/JETSON_DEPLOY.md).

## Setup

### Mac (MLX)

Use a **dedicated venv** so NumPy/scipy versions don't clash:

```bash
cd /path/to/spartan
python -m venv .venv
source .venv/bin/activate
pip install -r vlm_service/requirements.txt
```

### Jetson (NanoLLM)

Install NanoLLM first (from jetson-containers or source), then:

```bash
cd /path/to/spartan
pip install -r vlm_service/requirements-jetson.txt
```

Or install TensorRT-LLM separately for the `trt` backend; see docs.

## Run

From the **repo root** (so `vlm_service` is on the module path):

```bash
# Mac
export VLM_BACKEND=mlx
python -m vlm_service

# Jetson (NanoLLM)
export VLM_BACKEND=nanollm
python -m vlm_service
# or: ./scripts/run-vlm-nanollm.sh

# Jetson (TensorRT-LLM)
export VLM_BACKEND=trt
export TRTLLM_MODEL_PATH=/path/to/engine
python -m vlm_service
```

Listens on `http://127.0.0.1:5000` by default (Jetson NanoLLM script uses `0.0.0.0`). Override with `VLM_HOST` and `VLM_PORT`.

## Environment Variables

| Variable | Default | Backend | Description |
|---------|---------|---------|-------------|
| `VLM_BACKEND` | `mlx` | All | `mlx`, `nanollm`, or `trt` |
| `VLM_HOST` | `127.0.0.1` | All | Bind address |
| `VLM_PORT` | `5000` | All | HTTP port |
| `VLM_PROMPT` | `Describe this image briefly.` | All | Default prompt |
| `VLM_MODEL_PATH` | *(backend-specific)* | mlx, nanollm | Model repo ID or local path |
| `NANOLLM_API` | `mlc` | nanollm | NanoLLM runtime: `mlc` or `trt` |
| `NANOLLM_QUANTIZATION` | `q4f16_ft` | nanollm | Quantisation preset |
| `MAX_FRAME_DIM` | `768` | nanollm | Max image dimension before inference |
| `TRTLLM_MODEL_PATH` | — | trt | TensorRT-LLM engine directory |

## API

- `POST /infer`: JSON body `{ "image_base64": "<base64>", "prompt": "..." }`. Returns `{ "text": "...", "elapsed_s": ... }`.
- `GET /health`: `{ "status": "ok", "backend": "nanollm" }`.
