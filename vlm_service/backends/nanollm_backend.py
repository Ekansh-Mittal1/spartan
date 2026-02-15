"""NanoLLM VLM backend (Jetson Orin / Nano). Uses nano_llm with vision-language models.

NanoLLM supports several VLMs on Jetson (VILA, LLaVA, Obsidian, etc.) and handles
quantized inference via MLC/TensorRT backends.  This backend wraps the same
`/infer` HTTP interface used by the MLX backend so the Node orchestrator and
frontend work unchanged.

Environment variables
---------------------
VLM_MODEL_PATH       Model repo ID or local path (default: Efficient-Large-Model/VILA1.5-3b).
NANOLLM_API          NanoLLM runtime backend: "mlc" or "trt" (default: mlc).
NANOLLM_QUANTIZATION Quantization preset (default: q4f16_ft).
MAX_FRAME_DIM        Max pixel dimension before inference (default: 768). Lower = faster.
VLM_PROMPT           Default prompt when none is supplied per-request.
"""

import io
import os
import time
import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger("vlm_service.nanollm")

MODEL_PATH_ENV = "VLM_MODEL_PATH"
DEFAULT_MODEL = "Efficient-Large-Model/VILA1.5-3b"

NANOLLM_API_ENV = "NANOLLM_API"
DEFAULT_API = "mlc"

NANOLLM_QUANTIZATION_ENV = "NANOLLM_QUANTIZATION"
DEFAULT_QUANTIZATION = "q4f16_ft"

MAX_FRAME_DIM_ENV = "MAX_FRAME_DIM"
DEFAULT_MAX_FRAME_DIM = 768

MAX_TOKENS = 150


class NanoLLMBackend:
    """NanoLLM vision-language backend for Jetson Orin / Nano.

    Follows the same interface as MLXBackend (load / infer) so the FastAPI
    server can swap backends transparently via VLM_BACKEND=nanollm.
    """

    def __init__(self):
        self._model = None
        self._chat_history = None
        self._max_frame_dim: int = int(
            os.environ.get(MAX_FRAME_DIM_ENV, str(DEFAULT_MAX_FRAME_DIM))
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self):
        """Load the NanoLLM model and create a ChatHistory instance."""
        # Import here so the module can be imported on non-Jetson hosts for
        # testing / linting without nano_llm installed.
        from nano_llm import NanoLLM, ChatHistory  # type: ignore[import-untyped]

        model_path = os.environ.get(MODEL_PATH_ENV) or DEFAULT_MODEL
        api = os.environ.get(NANOLLM_API_ENV) or DEFAULT_API
        quantization = os.environ.get(NANOLLM_QUANTIZATION_ENV) or DEFAULT_QUANTIZATION

        logger.info(
            "Loading NanoLLM model=%s api=%s quantization=%s",
            model_path, api, quantization,
        )

        self._model = NanoLLM.from_pretrained(
            model_path,
            api=api,
            quantization=quantization,
        )

        if not getattr(self._model, "has_vision", True):
            logger.warning(
                "Model %s may not have vision capabilities; continuing anyway.",
                model_path,
            )

        self._chat_history = ChatHistory(self._model)
        logger.info("NanoLLM model loaded successfully.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        image_path_or_bytes: Union[str, Path, bytes, bytearray],
        prompt: str | None = None,
    ) -> Tuple[str, float]:
        """Run VLM inference on an image.

        Parameters
        ----------
        image_path_or_bytes : str | Path | bytes | bytearray
            File path to an image or raw image bytes (e.g. JPEG).
        prompt : str, optional
            Text prompt for the model.  Falls back to VLM_PROMPT env or a
            sensible default.

        Returns
        -------
        tuple[str, float]
            (generated_text, elapsed_seconds)
        """
        if self._model is None or self._chat_history is None:
            self.load()

        prompt_text = prompt or os.environ.get(
            "VLM_PROMPT", "Describe this image briefly."
        )

        # Convert input to an RGB numpy array that NanoLLM expects.
        frame_rgb = self._to_rgb_array(image_path_or_bytes)

        # Resize large frames for speed.
        frame_rgb = self._maybe_resize(frame_rgb)

        # Skip mostly-black frames (camera warmup / lens cap).
        if np.mean(frame_rgb) < 15:
            logger.debug("Skipping near-black frame (mean pixel < 15).")
            return "[skipped - black frame]", 0.0

        try:
            start = time.perf_counter()

            # Build chat with image + prompt.
            self._chat_history.append("user", image=frame_rgb)
            self._chat_history.append("user", prompt_text, use_cache=True)
            embedding, _ = self._chat_history.embed_chat()

            # Generate response.
            reply = self._model.generate(
                embedding,
                kv_cache=self._chat_history.kv_cache,
                max_new_tokens=MAX_TOKENS,
                stop_tokens=getattr(self._chat_history.template, "stop", None),
            )

            # NanoLLM may return a streaming object; materialise the text.
            text = getattr(reply, "text", None)
            if not text:
                text = "".join(reply)
            text = (text or "").strip()

            # Reset chat history for the next single-turn inference.
            self._chat_history.reset()

            elapsed = time.perf_counter() - start
            return text, elapsed

        except Exception as exc:
            logger.error("NanoLLM inference failed: %s", exc, exc_info=True)
            # Always reset so the next call starts clean.
            try:
                self._chat_history.reset()
            except Exception:
                pass
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        """Down-scale frame so the longest edge is at most max_frame_dim."""
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim <= self._max_frame_dim:
            return frame
        scale = self._max_frame_dim / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        # Use PIL for high-quality downsampling (LANCZOS).
        pil = Image.fromarray(frame)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        return np.asarray(pil)

    @staticmethod
    def _to_rgb_array(
        image_path_or_bytes: Union[str, Path, bytes, bytearray],
    ) -> np.ndarray:
        """Convert file path or raw bytes to an RGB numpy array."""
        if isinstance(image_path_or_bytes, (str, Path)) and os.path.isfile(
            image_path_or_bytes
        ):
            pil = Image.open(image_path_or_bytes).convert("RGB")
            return np.asarray(pil)
        if isinstance(image_path_or_bytes, (bytes, bytearray)):
            pil = Image.open(io.BytesIO(image_path_or_bytes)).convert("RGB")
            return np.asarray(pil)
        raise ValueError(
            "image_path_or_bytes must be a file path or raw image bytes"
        )
