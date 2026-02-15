"""MLX VLM backend (Mac / Apple Silicon). Uses mlx-vlm with Qwen2.5-VL (qwen2_5_vl)."""
import importlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

# ---------------------------------------------------------------------------
# Patch transformers 5.x bug: video_processor_class_from_name crashes when
# torchvision is absent because some VIDEO_PROCESSOR_MAPPING_NAMES values are
# None, and `class_name in None` raises TypeError.  We patch once at import
# time so the rest of the stack (AutoProcessor.from_pretrained) works.
# ---------------------------------------------------------------------------
import transformers.models.auto.video_processing_auto as _vpa  # noqa: E402

_orig_vp_class_from_name = _vpa.video_processor_class_from_name


def _safe_video_processor_class_from_name(class_name: str):  # type: ignore[override]
    for module_name, extractors in _vpa.VIDEO_PROCESSOR_MAPPING_NAMES.items():
        if extractors is None:
            continue
        if class_name in extractors:
            mod_name = _vpa.model_type_to_module_name(module_name)
            module = importlib.import_module(f".{mod_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue
    for extractor in _vpa.VIDEO_PROCESSOR_MAPPING._extra_content.values():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor
    return None


_vpa.video_processor_class_from_name = _safe_video_processor_class_from_name
# ---------------------------------------------------------------------------

from mlx_vlm import load, generate  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402

MODEL_PATH_ENV = "VLM_MODEL_PATH"
# Qwen2.5-VL is supported by mlx-vlm (qwen2_5_vl). Qwen3-VL (qwen3_vl) is not in mlx-vlm yet.
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-8bit"
MAX_TOKENS = 150

# Keys that mlx_vlm's qwen2_5_vl expects at root so ModelConfig.from_dict can propagate
# them into text_config AND so ModelConfig itself can read eos_token_id, vocab_size, etc.
_ROOT_KEYS_NEEDED = frozenset({
    # TextConfig fields
    "model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
    "num_attention_heads", "rms_norm_eps", "vocab_size", "num_key_value_heads",
    "max_position_embeddings", "rope_theta", "rope_traditional", "rope_scaling",
    "tie_word_embeddings", "sliding_window", "use_sliding_window", "use_cache",
    # ModelConfig fields that may also be popped into nested text_config by HF
    "eos_token_id", "bos_token_id",
})


def _normalize_qwen25vl_config(config: dict[str, Any]) -> dict[str, Any]:
    """Ensure root has all keys mlx_vlm expects for Qwen2.5-VL.

    HuggingFace's AutoConfig.from_pretrained().to_dict() pops text-related keys
    (including hidden_size, vocab_size, eos_token_id, …) from root kwargs and nests
    them under config["text_config"]. mlx_vlm expects those keys at root level:
      - ModelConfig.from_dict() builds text_config from root-level keys
      - ModelConfig.eos_token_id is read from root
    Merge any missing root keys from the nested text_config dict.
    """
    if config.get("model_type") != "qwen2_5_vl":
        return config
    nested = config.get("text_config")
    if not isinstance(nested, dict):
        return config
    for k, v in nested.items():
        if k in _ROOT_KEYS_NEEDED and k not in config:
            config[k] = v
    return config


class MLXBackend:
    def __init__(self):
        self._model = None
        self._processor = None
        self._formatted_prompt = None

    def load(self):
        model_path = os.environ.get(MODEL_PATH_ENV) or DEFAULT_MODEL
        import mlx_vlm.utils as mlx_utils
        _orig_load_config = mlx_utils.load_config
        try:
            def _patched_load_config(path, **kwargs):
                out = _orig_load_config(path, **kwargs)
                out = _normalize_qwen25vl_config(out)
                # Transformers 5.x AutoConfig.to_dict() drops rope_scaling to None;
                # restore from the raw config.json on disk when that happens.
                if out.get("model_type") == "qwen2_5_vl" and out.get("rope_scaling") is None:
                    resolved = path if isinstance(path, Path) else mlx_utils.get_model_path(str(path))
                    raw_path = Path(resolved) / "config.json"
                    if raw_path.exists():
                        with open(raw_path, encoding="utf-8") as f:
                            raw = json.load(f)
                        if raw.get("rope_scaling") is not None:
                            out["rope_scaling"] = raw["rope_scaling"]
                return out
            mlx_utils.load_config = _patched_load_config
            self._model, self._processor = load(model_path, trust_remote_code=True)
        finally:
            mlx_utils.load_config = _orig_load_config
        prompt = os.environ.get("VLM_PROMPT", "Describe this image briefly.")
        self._formatted_prompt = apply_chat_template(
            self._processor, self._model.config, prompt, num_images=1
        )

    def infer(self, image_path_or_bytes: str | Path | bytes | bytearray, prompt: str | None = None):
        if self._model is None or self._processor is None:
            self.load()
        if prompt is not None:
            self._formatted_prompt = apply_chat_template(
                self._processor, self._model.config, prompt, num_images=1
            )
        pil_img = self._to_pil(image_path_or_bytes)
        start = time.perf_counter()
        output = generate(
            self._model,
            self._processor,
            self._formatted_prompt,
            [pil_img],
            verbose=False,
            max_tokens=MAX_TOKENS,
        )
        elapsed = time.perf_counter() - start
        text = output.text if hasattr(output, "text") else str(output)
        return text.strip(), elapsed

    @staticmethod
    def _to_pil(image_path_or_bytes: str | Path | bytes | bytearray) -> Image.Image:
        """Convert file path or raw bytes to an RGB PIL Image."""
        if isinstance(image_path_or_bytes, (str, Path)) and os.path.isfile(image_path_or_bytes):
            return Image.open(image_path_or_bytes).convert("RGB")
        if isinstance(image_path_or_bytes, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_path_or_bytes)).convert("RGB")
        raise ValueError("image_path_or_bytes must be a file path or raw image bytes")
