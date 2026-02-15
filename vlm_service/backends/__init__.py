"""VLM backends: MLX (Mac), NanoLLM (Jetson), SGLang (Jetson/GPU), TensorRT-LLM (Jetson)."""
import os


def get_backend():
    backend = (os.environ.get("VLM_BACKEND") or "mlx").lower()
    if backend == "mlx":
        from .mlx_backend import MLXBackend
        return MLXBackend()
    if backend == "nanollm":
        from .nanollm_backend import NanoLLMBackend
        return NanoLLMBackend()
    if backend == "sglang":
        from .sglang_backend import SGLangBackend
        return SGLangBackend()
    if backend == "trt":
        from .trt_backend import TRTBackend
        return TRTBackend()
    raise ValueError(f"Unknown VLM_BACKEND={backend!r}. Use mlx, nanollm, sglang, or trt.")
