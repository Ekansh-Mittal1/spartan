"""VLM backends: MLX (Mac) and TensorRT-LLM (Jetson)."""
import os

def get_backend():
    backend = (os.environ.get("VLM_BACKEND") or "mlx").lower()
    if backend == "mlx":
        from .mlx_backend import MLXBackend
        return MLXBackend()
    if backend == "trt":
        from .trt_backend import TRTBackend
        return TRTBackend()
    raise ValueError(f"Unknown VLM_BACKEND={backend!r}. Use mlx or trt.")
