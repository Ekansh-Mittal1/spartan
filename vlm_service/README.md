# VLM Service

Local Qwen VLM for the edge pipeline. Runs as an HTTP service; the Node backend POSTs frames to `/infer` and receives `{ "text", "elapsed_s" }`.

## Backends

- **MLX** (Mac): `VLM_BACKEND=mlx`. Uses `mlx-vlm`; default model is `Qwen2.5-VL-3B-Instruct-8bit` (mlx-vlm supports `qwen2_5_vl`; `qwen3_vl` is not supported yet). Set `VLM_MODEL_PATH` to override (e.g. another Qwen2.5-VL or Qwen2-VL model).
- **TensorRT-LLM** (Jetson): `VLM_BACKEND=trt`. Set `TRTLLM_MODEL_PATH` to the engine directory. Implement `trt_backend.py` per [docs/JETSON_DEPLOY.md](../docs/JETSON_DEPLOY.md).

## Setup

Use a **dedicated venv** so NumPy/scipy versions don't clash with other projects (e.g. Anaconda):

```bash
cd /path/to/spartan-sync
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r vlm_service/requirements.txt
```

On Jetson, install TensorRT-LLM separately; see docs.

## Run

From the **repo root** (so `vlm_service` is on the module path):

```bash
cd /path/to/spartan-sync
export VLM_BACKEND=mlx   # or trt on Jetson
python -m vlm_service
```

Listens on `http://127.0.0.1:5000` by default. Override with `VLM_HOST` and `VLM_PORT`.

## API

- `POST /infer`: JSON body `{ "image_base64": "<base64>", "prompt": "..." }`. Returns `{ "text": "...", "elapsed_s": ... }`.
- `GET /health`: `{ "status": "ok", "backend": "mlx" }`.
