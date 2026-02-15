"""VLM backends: MLX (Mac), NanoLLM (Jetson), and TensorRT-LLM (Jetson)."""
import os


def get_backend():
    backend = (os.environ.get("VLM_BACKEND") or "mlx").lower()
    if backend == "mlx":
        from .mlx_backend import MLXBackend
        return MLXBackend()
    if backend == "nanollm":
        from .nanollm_backend import NanoLLMBackend
        return NanoLLMBackend()
    if backend == "trt":
        from .trt_backend import TRTBackend
        return TRTBackend()
    raise ValueError(f"Unknown VLM_BACKEND={backend!r}. Use mlx, nanollm, or trt.")
