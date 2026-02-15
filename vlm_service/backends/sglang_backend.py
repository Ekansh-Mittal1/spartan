"""SGLang VLM backend (Jetson / any GPU). Calls a local OpenAI-compatible server.

Works with any inference server that implements the OpenAI chat completions API
with vision support: SGLang, vLLM, Ollama, etc. The server runs as a separate
process and this backend calls it over HTTP.

Environment variables
---------------------
SGLANG_BASE_URL    Base URL of the inference server (default: http://127.0.0.1:30000).
SGLANG_MODEL       Model name to pass in the API request (default: "default").
                   Set to the actual model name if the server requires it.
VLM_PROMPT         Default prompt when none is supplied per-request.
MAX_NEW_TOKENS     Maximum tokens to generate (default: 150).
"""

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Tuple, Union

logger = logging.getLogger("vlm_service.sglang")

SGLANG_BASE_URL_ENV = "SGLANG_BASE_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:30000"

SGLANG_MODEL_ENV = "SGLANG_MODEL"
DEFAULT_MODEL = "default"

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "150"))

# Timeout for inference requests (seconds). VLM inference on edge can take a while.
REQUEST_TIMEOUT = int(os.environ.get("SGLANG_TIMEOUT", "60"))


class SGLangBackend:
    """Vision-language backend that calls a local OpenAI-compatible API server.

    Follows the same interface as the other backends (load / infer) so the
    FastAPI server can swap backends transparently via VLM_BACKEND=sglang.
    """

    def __init__(self):
        self._base_url = (
            os.environ.get(SGLANG_BASE_URL_ENV) or DEFAULT_BASE_URL
        ).rstrip("/")
        self._model = os.environ.get(SGLANG_MODEL_ENV) or DEFAULT_MODEL

    # ------------------------------------------------------------------
    # Model loading (just verifies the server is reachable)
    # ------------------------------------------------------------------

    def load(self):
        """Verify the inference server is reachable."""
        health_urls = [
            f"{self._base_url}/health",
            f"{self._base_url}/v1/models",
        ]
        for url in health_urls:
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        logger.info(
                            "SGLang server reachable at %s (model=%s)",
                            self._base_url,
                            self._model,
                        )
                        return
            except Exception:
                continue

        logger.warning(
            "SGLang server at %s not reachable yet. "
            "Inference will retry on first request.",
            self._base_url,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        image_path_or_bytes: Union[str, Path, bytes, bytearray],
        prompt: str | None = None,
    ) -> Tuple[str, float]:
        """Run VLM inference via the OpenAI-compatible chat completions API.

        Parameters
        ----------
        image_path_or_bytes : str | Path | bytes | bytearray
            File path to an image or raw image bytes (e.g. JPEG).
        prompt : str, optional
            Text prompt for the model.

        Returns
        -------
        tuple[str, float]
            (generated_text, elapsed_seconds)
        """
        prompt_text = prompt or os.environ.get(
            "VLM_PROMPT", "Describe this image briefly."
        )

        # Convert input to base64 JPEG.
        image_b64 = self._to_base64(image_path_or_bytes)
        data_url = f"data:image/jpeg;base64,{image_b64}"

        # Build the OpenAI chat completions request.
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.1,
        }

        body = json.dumps(payload).encode("utf-8")
        url = f"{self._base_url}/v1/chat/completions"

        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            start = time.perf_counter()
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - start

            # Extract text from standard OpenAI response format.
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip()

            return text, elapsed

        except urllib.error.URLError as exc:
            logger.error("SGLang request failed: %s", exc)
            raise RuntimeError(
                f"SGLang server at {self._base_url} unreachable: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("SGLang inference failed: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_base64(image_path_or_bytes: Union[str, Path, bytes, bytearray]) -> str:
        """Convert file path or raw bytes to a base64 string."""
        if isinstance(image_path_or_bytes, (str, Path)) and os.path.isfile(
            image_path_or_bytes
        ):
            with open(image_path_or_bytes, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        if isinstance(image_path_or_bytes, (bytes, bytearray)):
            return base64.b64encode(image_path_or_bytes).decode("ascii")
        raise ValueError(
            "image_path_or_bytes must be a file path or raw image bytes"
        )
