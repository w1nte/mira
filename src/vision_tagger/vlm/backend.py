"""VLM backends: Ollama (default) and a deterministic fake for tests / dry runs."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import zlib
from collections.abc import Callable
from typing import Any, Protocol

import httpx
import numpy as np
from PIL import Image, ImageStat

from ..config import VlmConfig
from .schema import CONFIDENCE_SUFFIX, Vocabulary

log = logging.getLogger(__name__)


class VlmError(RuntimeError):
    pass


class VlmConnectionError(VlmError):
    """The VLM server could not be reached at all."""


class VlmBackend(Protocol):
    model: str

    def analyze(
        self, image: Image.Image, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]: ...

    def chat_json(
        self, system: str, user: str, schema: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]: ...

    def embed(self, texts: list[str], model: str) -> np.ndarray: ...

    def has_model(self, name: str) -> bool: ...

    def health(self) -> dict[str, Any]: ...

    def unload(self) -> None: ...

    def close(self) -> None: ...


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def encode_image(image: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_json_answer(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model answer (tolerates code fences / leading prose)."""
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise VlmError(f"no JSON object in answer: {text[:200]!r}") from None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise VlmError(f"invalid JSON in answer: {exc}") from exc
    if not isinstance(data, dict):
        raise VlmError("answer is not a JSON object")
    return data


class OllamaBackend:
    """Talks to a local Ollama server (``/api/chat`` with JSON-schema structured output)."""

    def __init__(self, cfg: VlmConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self._client = client or httpx.Client(
            base_url=cfg.base_url.rstrip("/"), timeout=httpx.Timeout(cfg.timeout, connect=10.0)
        )

    # ------------------------------------------------------------------ transport
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = self._client.post(path, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise VlmConnectionError(f"cannot reach Ollama at {self.cfg.base_url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise VlmError(f"Ollama request failed: {exc}") from exc
        if resp.status_code != 200:
            raise VlmError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _chat(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        model: str | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "user", "content": user}
        if images:
            message["images"] = images
        payload = {
            "model": model or self.cfg.model,
            "stream": False,
            "format": schema,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature,
                "seed": self.cfg.seed,
                "num_ctx": self.cfg.num_ctx,
            },
            "messages": [{"role": "system", "content": system}, message],
        }
        body = self._post("/api/chat", payload)
        content = (body.get("message") or {}).get("content", "")
        return parse_json_answer(content)

    # ------------------------------------------------------------------ API
    def analyze(
        self, image: Image.Image, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        return self._chat(system, user, schema, images=[encode_image(image)])

    def chat_json(
        self, system: str, user: str, schema: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        """Text-only structured answer (used by the search query interpreter / re-ranker)."""
        return self._chat(system, user, schema, model=model)

    def embed(self, texts: list[str], model: str) -> np.ndarray:
        """L2-normalised embeddings for ``texts`` (``/api/embed``)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        body = self._post(
            "/api/embed", {"model": model, "input": texts, "keep_alive": self.cfg.keep_alive}
        )
        vectors = body.get("embeddings") or []
        if len(vectors) != len(texts):
            raise VlmError(f"embedding model returned {len(vectors)} vectors for {len(texts)}")
        return _normalize_rows(np.asarray(vectors, dtype=np.float32))

    def _installed(self) -> set[str]:
        from .models import installed_models

        return set(installed_models(self._client))

    def has_model(self, name: str) -> bool:
        from .models import normalize

        try:
            installed = self._installed()
        except Exception:
            return False
        return normalize(name) in installed

    def pull(
        self,
        name: str,
        on_progress: Callable[[Any], None] | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Any:
        """Let Ollama download ``name`` (see :func:`vlm.models.pull`)."""
        from .models import pull

        return pull(name, client=self._client, on_progress=on_progress, cancel=cancel)

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": "ollama", "url": self.cfg.base_url, "model": self.model}
        try:
            version = self._client.get("/api/version", timeout=3).json().get("version")
            names = self._installed()
        except httpx.TransportError as exc:
            info.update(
                ok=False,
                error=f"cannot reach Ollama at {self.cfg.base_url} ({type(exc).__name__}: {exc}); "
                "install it from https://ollama.com/download and start it",
            )
            return info
        except Exception as exc:
            info.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            return info
        from .models import normalize

        info.update(ok=normalize(self.model) in names, version=version, installed=sorted(names))
        if not info["ok"]:
            info["error"] = f"model {self.model!r} not pulled (ollama pull {self.model})"
        return info

    def unload(self) -> None:
        try:
            self._client.post(
                "/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=10
            )
        except Exception as exc:  # pragma: no cover - best effort
            log.debug("unload failed: %s", exc)

    def close(self) -> None:
        self._client.close()


class FakeBackend:
    """Deterministic pseudo-analysis derived from image statistics.

    Useful for tests and for trying the UI without a GPU. ``responses`` can force answers.
    """

    model = "fake"

    EMBED_DIM = 256

    def __init__(
        self,
        vocab: Vocabulary,
        responses: list[Any] | None = None,
        chat_responses: list[Any] | None = None,
    ) -> None:
        self.vocab = vocab
        self.responses = list(responses or [])
        self.chat_responses = list(chat_responses or [])
        self.calls = 0
        self.chat_calls: list[tuple[str, str]] = []

    def analyze(
        self, image: Image.Image, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls += 1
        if self.responses:
            r = self.responses.pop(0)
            if isinstance(r, Exception):
                raise r
            if isinstance(r, str):
                return parse_json_answer(r)
            return r
        stat = ImageStat.Stat(image.convert("RGB"))
        r, g, b = stat.mean[:3]
        bright = (r + g + b) / 3
        out: dict[str, Any] = {}
        for f in self.vocab.fields.values():
            base = f.base_type
            if base == "enum":
                out[f.name] = f.values[int(bright + len(f.name)) % len(f.values)]
            elif base == "enum_list":
                dominant = max((r, "red"), (g, "green"), (b, "blue"))[1]
                out[f.name] = [dominant] if dominant in f.values else f.values[:1]
            elif base == "list":
                out[f.name] = ["Trees", "a car"] if g > r else ["Buildings", "people"]
            elif base == "boolean":
                out[f.name] = False if f.name == "nsfw" else bright > 127
            elif base == "integer":
                out[f.name] = int(bright) % 4
            else:
                out[f.name] = f"A {'bright' if bright > 127 else 'dark'} image."
            if f.inferred:
                out[f.name + CONFIDENCE_SUFFIX] = round((bright % 100) / 100, 2)
        return out

    def chat_json(
        self, system: str, user: str, schema: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        """Queued answers only; without one the caller falls back to keyword search."""
        self.chat_calls.append((system, user))
        if not self.chat_responses:
            raise VlmError("the fake backend has no text model")
        r = self.chat_responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return parse_json_answer(r) if isinstance(r, str) else r

    def embed(self, texts: list[str], model: str) -> np.ndarray:
        """Deterministic bag-of-words hashing vectors: shared words -> higher similarity."""
        out = np.zeros((len(texts), self.EMBED_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            text = re.sub(r"^search_\w+:\s*", "", text.lower())
            for token in re.findall(r"[\w]+", text):
                if len(token) > 2:
                    out[row, zlib.crc32(token.encode()) % self.EMBED_DIM] += 1.0
        return _normalize_rows(out)

    def has_model(self, name: str) -> bool:
        return True

    def health(self) -> dict[str, Any]:
        return {"backend": "fake", "ok": True, "model": self.model}

    def unload(self) -> None:
        pass

    def close(self) -> None:
        pass


def make_backend(cfg: VlmConfig, vocab: Vocabulary) -> VlmBackend:
    if cfg.backend == "fake":
        return FakeBackend(vocab)
    return OllamaBackend(cfg)
