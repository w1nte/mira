"""VLM backends: Ollama (default) and a deterministic fake for tests / dry runs."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any, Protocol

import httpx
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

    def health(self) -> dict[str, Any]: ...

    def unload(self) -> None: ...

    def close(self) -> None: ...


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

    def analyze(
        self, image: Image.Image, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        payload = {
            "model": self.cfg.model,
            "stream": False,
            "format": schema,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature,
                "seed": self.cfg.seed,
                "num_ctx": self.cfg.num_ctx,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [encode_image(image)]},
            ],
        }
        try:
            resp = self._client.post("/api/chat", json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise VlmConnectionError(f"cannot reach Ollama at {self.cfg.base_url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise VlmError(f"Ollama request failed: {exc}") from exc
        if resp.status_code != 200:
            raise VlmError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        content = (body.get("message") or {}).get("content", "")
        return parse_json_answer(content)

    def health(self) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": "ollama", "url": self.cfg.base_url, "model": self.model}
        try:
            version = self._client.get("/api/version", timeout=3).json().get("version")
            tags = self._client.get("/api/tags", timeout=3).json().get("models", [])
        except Exception as exc:
            info.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            return info
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        info.update(ok=wanted in names, version=version, installed=sorted(n for n in names if n))
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

    def __init__(self, vocab: Vocabulary, responses: list[Any] | None = None) -> None:
        self.vocab = vocab
        self.responses = list(responses or [])
        self.calls = 0

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
                out[f.name] = bright > 127
            elif base == "integer":
                out[f.name] = int(bright) % 4
            else:
                out[f.name] = f"A {'bright' if bright > 127 else 'dark'} image."
            if f.inferred:
                out[f.name + CONFIDENCE_SUFFIX] = round((bright % 100) / 100, 2)
        return out

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
