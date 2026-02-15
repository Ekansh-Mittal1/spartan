"""Local VLM backend — calls Ollama, SGLang, or any OpenAI-compatible server.

Auto-detects Ollama and uses its native /api/chat endpoint (which supports
vision via the "images" field). For other servers (SGLang, vLLM) it uses
the standard /v1/chat/completions endpoint with image_url content blocks.

Environment variables
---------------------
SGLANG_BASE_URL    Base URL of the inference server (default: http://127.0.0.1:11434).
SGLANG_MODEL       Model name to pass in the API request (default: "moondream").
VLM_PROMPT         Default prompt when none is supplied per-request.
MAX_NEW_TOKENS     Maximum tokens to generate (default: 150).
SGLANG_TIMEOUT     Request timeout in seconds (default: 60).
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
DEFAULT_BASE_URL = "http://127.0.0.1:11434"

SGLANG_MODEL_ENV = "SGLANG_MODEL"
DEFAULT_MODEL = "moondream"

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "150"))
REQUEST_TIMEOUT = int(os.environ.get("SGLANG_TIMEOUT", "60"))


class SGLangBackend:
    """Vision-language backend that calls a local inference server.

    Auto-detects Ollama (uses /api/chat with native image support) vs
    SGLang/vLLM (uses /v1/chat/completions with OpenAI image_url format).

    Follows the same interface as the other backends (load / infer) so the
    FastAPI server can swap backends transparently via VLM_BACKEND=sglang.
    """

    def __init__(self):
        self._base_url = (
            os.environ.get(SGLANG_BASE_URL_ENV) or DEFAULT_BASE_URL
        ).rstrip("/")
        self._model = os.environ.get(SGLANG_MODEL_ENV) or DEFAULT_MODEL
        self._is_ollama: bool | None = None  # detected on load()

    # ------------------------------------------------------------------
    # Server detection & loading
    # ------------------------------------------------------------------

    def _detect_ollama(self) -> bool:
        """Check if the server is Ollama by probing its root endpoint."""
        try:
            req = urllib.request.Request(self._base_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if "Ollama" in body:
                    return True
        except Exception:
            pass
        # Also check /api/tags (Ollama-specific)
        try:
            req = urllib.request.Request(
                f"{self._base_url}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        return False

    def load(self):
        """Detect server type and verify it is reachable."""
        self._is_ollama = self._detect_ollama()
        if self._is_ollama:
            logger.info(
                "Ollama detected at %s (model=%s). Using /api/chat.",
                self._base_url,
                self._model,
            )
            return

        # Try SGLang/vLLM style endpoints
        for path in ("/health", "/v1/models"):
            try:
                req = urllib.request.Request(
                    f"{self._base_url}{path}", method="GET"
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        logger.info(
                            "OpenAI-compatible server at %s (model=%s). "
                            "Using /v1/chat/completions.",
                            self._base_url,
                            self._model,
                        )
                        return
            except Exception:
                continue

        logger.warning(
            "Server at %s not reachable. Will retry on first request.",
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
        """Run VLM inference on an image.

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
        # Auto-detect on first call if load() wasn't called
        if self._is_ollama is None:
            self._is_ollama = self._detect_ollama()

        if self._is_ollama:
            return self._infer_ollama(image_path_or_bytes, prompt)
        else:
            return self._infer_openai(image_path_or_bytes, prompt)

    def _infer_ollama(
        self,
        image_path_or_bytes: Union[str, Path, bytes, bytearray],
        prompt: str | None,
    ) -> Tuple[str, float]:
        """Ollama native API: POST /api/chat with images array."""
        prompt_text = prompt or os.environ.get(
            "VLM_PROMPT", "Describe this image briefly."
        )
        image_b64 = self._to_base64(image_path_or_bytes)

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {
                "num_predict": MAX_NEW_TOKENS,
                "temperature": 0.1,
            },
        }

        body = json.dumps(payload).encode("utf-8")
        url = f"{self._base_url}/api/chat"

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

            text = result.get("message", {}).get("content", "").strip()
            return text, elapsed

        except urllib.error.HTTPError as exc:
            logger.error("Ollama request failed: %s %s", exc.code, exc.reason)
            raise RuntimeError(
                f"Ollama at {self._base_url} returned {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            logger.error("Ollama unreachable: %s", exc)
            raise RuntimeError(
                f"Ollama at {self._base_url} unreachable: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Ollama inference failed: %s", exc, exc_info=True)
            raise

    def _infer_openai(
        self,
        image_path_or_bytes: Union[str, Path, bytes, bytearray],
        prompt: str | None,
    ) -> Tuple[str, float]:
        """OpenAI-compatible API: POST /v1/chat/completions with image_url."""
        prompt_text = prompt or os.environ.get(
            "VLM_PROMPT", "Describe this image briefly."
        )
        image_b64 = self._to_base64(image_path_or_bytes)
        data_url = f"data:image/jpeg;base64,{image_b64}"

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

            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip()
            return text, elapsed

        except urllib.error.URLError as exc:
            logger.error("OpenAI-compatible request failed: %s", exc)
            raise RuntimeError(
                f"Server at {self._base_url} unreachable: {exc}"
            ) from exc
        except Exception as exc:
            logger.error("Inference failed: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_base64(
        image_path_or_bytes: Union[str, Path, bytes, bytearray],
    ) -> str:
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
