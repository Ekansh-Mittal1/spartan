"""FastAPI server for local VLM inference. POST /infer with image + prompt, returns text."""
import base64
import logging
import os
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vlm_service")

from .backends import get_backend  # noqa: E402

app = FastAPI(title="VLM Service", description="Local VLM for edge pipeline (MLX / NanoLLM / TRT)")
_backend = None
_backend_name: str = ""


class InferBody(BaseModel):
    image_base64: str
    prompt: Optional[str] = None


def _get_backend():
    global _backend, _backend_name
    if _backend is None:
        _backend_name = (os.environ.get("VLM_BACKEND") or "mlx").lower()
        _backend = get_backend()
        _backend.load()
    return _backend


@app.post("/infer")
async def infer(body: InferBody):
    """Run VLM on an image. JSON body: { "image_base64": "<base64>", "prompt": "..." (optional) }."""
    prompt_val = body.prompt or os.environ.get("VLM_PROMPT", "Describe this image briefly.")
    try:
        image_bytes = base64.b64decode(body.image_base64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64: {e}") from e
    try:
        backend = _get_backend()
        text, elapsed_s = backend.infer(image_bytes, prompt=prompt_val)
        return JSONResponse(content={"text": text, "elapsed_s": round(elapsed_s, 4)})
    except Exception as e:
        logger.error("VLM inference failed:\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"text": f"[VLM error: {e}]", "elapsed_s": 0.0},
        )


@app.get("/health")
async def health():
    backend_name = (os.environ.get("VLM_BACKEND") or "mlx").lower()
    return {"status": "ok", "backend": backend_name}
