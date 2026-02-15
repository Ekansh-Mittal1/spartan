"""TensorRT-LLM VLM backend (Jetson). Qwen2-VL / Qwen2.5-VL via TensorRT-LLM."""
import os
import tempfile
import time
from pathlib import Path

TRTLLM_MODEL_PATH_ENV = "TRTLLM_MODEL_PATH"
MAX_TOKENS = 150


class TRTBackend:
    """TensorRT-LLM vision-language backend for Jetson Orin.

    Set TRTLLM_MODEL_PATH to the TensorRT-LLM engine or checkpoint directory.
    Requires TensorRT-LLM Jetson build (see docs/JETSON_DEPLOY.md).
    """

    def __init__(self):
        self._session = None
        self._model_path = os.environ.get(TRTLLM_MODEL_PATH_ENV)

    def load(self):
        model_path = self._model_path
        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(
                f"TRTLLM_MODEL_PATH must point to a TensorRT-LLM Qwen2-VL engine dir. "
                f"Got: {model_path!r}. See docs/JETSON_DEPLOY.md."
            )
        # TensorRT-LLM Python API for VL models varies by version; stub for now.
        # Typical: build session from engine dir, bind vision encoder + LLM.
        # self._session = tensorrt_llm.VLSession(model_path=model_path)  # placeholder
        self._session = {"model_path": model_path}

    def infer(self, image_path_or_bytes, prompt=None):
        if self._session is None:
            self.load()
        path, is_temp = self._ensure_path(image_path_or_bytes)
        try:
            start = time.perf_counter()
            # Placeholder: replace with actual TensorRT-LLM VL inference.
            # text = self._session.generate(image_path=path, prompt=prompt or "Describe this image briefly.", max_tokens=MAX_TOKENS)
            text = (
                "[TensorRT-LLM backend not implemented: install TensorRT-LLM on Jetson "
                "and implement infer() in vlm_service/backends/trt_backend.py. See docs/JETSON_DEPLOY.md.]"
            )
            elapsed = time.perf_counter() - start
            return text.strip(), elapsed
        finally:
            if is_temp:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _ensure_path(self, image_path_or_bytes):
        if isinstance(image_path_or_bytes, (str, Path)) and os.path.isfile(image_path_or_bytes):
            return str(image_path_or_bytes), False
        if isinstance(image_path_or_bytes, (bytes, bytearray)):
            f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            f.write(image_path_or_bytes)
            f.close()
            return f.name, True
        raise ValueError("image_path_or_bytes must be a file path or bytes")
