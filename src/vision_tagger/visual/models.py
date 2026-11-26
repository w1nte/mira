"""The model files of visual search: pinned downloads, verified before use.

SigLIP 2 base (Google, Apache-2.0) as ONNX from onnx-community, pinned to one commit:

* the image encoder in float32 (372 MB): on the CPU twice as fast as the float16 file, and the
  int8 file loses too much (its vectors differ from float32 ones by a cosine of 0.2, and photos
  are found far less often),
* the text encoder in float16 (565 MB): the same vectors as float32 (cosine 0.99999) at half
  the size; most of it is the vocabulary of more than 100 languages,
* its Gemma tokenizer (34 MB).

The files are downloaded once into ``visual.model_dir`` (default: next to the face models,
``<faces.model_root>/models/visual``), like the animal models: streamed into a temporary file,
checked for size and SHA-256 and only then moved into place. Without internet the error says
where to put the files by hand. The download reports its progress and gives way to a pause or
stop of the pipeline.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..animals.models import ModelFile, ModelUnavailable, _replace, sha256_of

log = logging.getLogger(__name__)

_SIGLIP2_B16 = (
    "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/"
    "ba1f3b0843f24bc5417d38e19c37b287d719b2f4/"
)


@dataclass(frozen=True)
class VisualModel:
    key: str  # the vector space, stored with every vector (``visual_embeddings.model``)
    title: str
    size: int  # the image encoder's input: size x size pixels
    dim: int
    files: dict[str, ModelFile] = field(hash=False)  # "vision", "text", "tokenizer"
    max_tokens: int = 64
    pad_id: int = 0
    pad_token: str = "<pad>"
    #: cosine similarity of a request and a photo: ``low`` is exceeded by only 10 % of the photos
    #: that don't show it, ``high`` is typical of clear matches (the best match of 9 in 10
    #: requests is above it). Measured on 1158 labelled photos, 38 requests in English and
    #: German: matching photos 0.05 (10th percentile) / 0.10 (median), others 0.04 (90th).
    low: float = 0.04
    high: float = 0.12
    #: changes when the preprocessing changes (re-embeds every photo)
    revision: int = 1

    @property
    def download_size(self) -> int:
        return sum(f.size for f in self.files.values())


SIGLIP2_BASE = VisualModel(
    key="siglip2-base-p16-224",
    title="SigLIP 2",
    size=224,
    dim=768,
    files={
        "vision": ModelFile(
            name="siglip2_base_p16_224_vision.onnx",
            url=_SIGLIP2_B16 + "onnx/vision_model.onnx",
            sha256="c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad",
            size=371_807_752,
            license="Apache-2.0",
            title="SigLIP 2 image encoder",
        ),
        "text": ModelFile(
            name="siglip2_base_p16_224_text_fp16.onnx",
            url=_SIGLIP2_B16 + "onnx/text_model_fp16.onnx",
            sha256="711da56ada0a4aa11c7dd3320df741081a3cae4f0ae1b5e5c6d5b294738d0eb0",
            size=564_862_230,
            license="Apache-2.0",
            title="SigLIP 2 text encoder",
        ),
        "tokenizer": ModelFile(
            name="siglip2_tokenizer.json",
            url=_SIGLIP2_B16 + "tokenizer.json",
            sha256="cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
            size=34_363_039,
            license="Apache-2.0",
            title="SigLIP 2 tokenizer",
        ),
    },
)

_SIGLIP2_B32 = (
    "https://huggingface.co/onnx-community/siglip2-base-patch32-256-ONNX/resolve/"
    "7efccb86b5b6601bfe9e14326e1c11b40b68d44c/"
)

#: patches of 32 instead of 16 pixels: three times as fast on the CPU, finds a little less
#: (COCO: the right photo first for 64.5 instead of 67.5 % of the captions)
SIGLIP2_BASE_FAST = VisualModel(
    key="siglip2-base-p32-256",
    title="SigLIP 2 (fast)",
    size=256,
    dim=768,
    files={
        "vision": ModelFile(
            name="siglip2_base_p32_256_vision.onnx",
            url=_SIGLIP2_B32 + "onnx/vision_model.onnx",
            sha256="0ddb7507cfcb54c2cd503029949add9dc6586de67164bb69c2c09588cdcc9719",
            size=378_478_689,
            license="Apache-2.0",
            title="SigLIP 2 (fast) image encoder",
        ),
        "text": ModelFile(
            name="siglip2_base_p32_256_text_fp16.onnx",
            url=_SIGLIP2_B32 + "onnx/text_model_fp16.onnx",
            sha256="3f8149272095410332433ca84f2c99611ed4679429eff43fbf3983ea15664ef3",
            size=564_862_230,
            license="Apache-2.0",
            title="SigLIP 2 (fast) text encoder",
        ),
        "tokenizer": SIGLIP2_BASE.files["tokenizer"],  # the same file
    },
)

#: ``visual.model`` -> model
MODELS: dict[str, VisualModel] = {
    "siglip2-base": SIGLIP2_BASE,
    "siglip2-base-fast": SIGLIP2_BASE_FAST,
}


def current_model(settings) -> VisualModel:
    return MODELS[settings.visual.model]


def visual_version(settings) -> str:
    """The ``visual`` stage version: the model files and the preprocessing."""
    spec = current_model(settings)
    return f"{spec.key}:{spec.files['vision'].sha256[:8]}-v{spec.revision}"


def default_model_dir(settings) -> Path:
    """``visual.model_dir``, else ``models/visual`` under the face model root."""
    if settings.visual.model_dir:
        return Path(settings.visual.model_dir).expanduser()
    faces = settings.faces
    root = Path(faces.model_root).expanduser() if faces.model_root else Path.home() / ".insightface"
    return root / "models" / "visual"


class DownloadInterrupted(RuntimeError):
    """The pipeline was paused or stopped during a download (nothing broken is kept)."""


Report = Callable[[dict[str, Any] | None], None]


def _present(target: Path, spec: ModelFile) -> bool:
    try:
        if target.stat().st_size == spec.size and sha256_of(target) == spec.sha256:
            return True
        log.warning("%s is not the expected file; downloading it again", target)
    except OSError:
        pass
    return False


def ensure_files(
    model_dir: Path,
    model: VisualModel,
    *,
    names: tuple[str, ...] = ("vision", "text", "tokenizer"),
    interrupted: Callable[[], bool] | None = None,
    report: Report | None = None,
    client=None,
) -> dict[str, Path]:
    """The paths of the model's files; missing ones are downloaded first (and verified).

    ``report`` gets the progress (like an Ollama download: ``model``, ``phase``, ``completed``,
    ``total``, ``percent``); ``interrupted`` is asked during the download, which then stops
    with :class:`DownloadInterrupted`. Raises :class:`ModelUnavailable` without internet.
    """
    paths = {name: Path(model_dir) / model.files[name].name for name in names}
    missing = [n for n in names if not _present(paths[n], model.files[n])]
    if not missing:
        return paths
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    total = sum(model.files[n].size for n in missing)
    done = 0
    try:
        for name in missing:
            spec = model.files[name]
            tmp = paths[name].with_name(f"{spec.name}.{uuid.uuid4().hex}.part")
            try:
                _download(
                    spec, tmp, paths[name], client, interrupted, report, model.title, done, total
                )
                _replace(tmp, paths[name])
            finally:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
            done += spec.size
            log.info("saved %s (%d bytes)", paths[name], spec.size)
    finally:
        if report is not None:
            report(None)
    return paths


def _download(
    spec: ModelFile,
    tmp: Path,
    target: Path,
    client,
    interrupted: Callable[[], bool] | None,
    report: Report | None,
    title: str,
    before: int,
    total: int,
) -> None:
    import httpx

    log.info("downloading %s from %s", spec.title, spec.url)
    digest = hashlib.sha256()
    size = 0
    own = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True)

    def progress(phase: str) -> None:
        if report is not None:
            completed = before + size
            report(
                {
                    "stage": "visual",
                    "model": title,
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                    "percent": min(100, 100 * completed // total) if total else None,
                }
            )

    try:
        progress("download")
        with http.stream("GET", spec.url) as resp, tmp.open("wb") as fh:
            resp.raise_for_status()
            for block in resp.iter_bytes(1 << 20):
                fh.write(block)
                digest.update(block)
                size += len(block)
                progress("download")
                if interrupted is not None and interrupted():
                    raise DownloadInterrupted(f"download of {spec.title} interrupted")
    except httpx.HTTPError as exc:
        raise ModelUnavailable(
            f"cannot download the {spec.title} ({exc}). Check the internet connection, "
            f"or download {spec.url} yourself and save it as {target} (SHA-256 {spec.sha256})."
        ) from exc
    finally:
        if own:
            http.close()
    progress("verify")
    if size != spec.size or digest.hexdigest() != spec.sha256:
        raise ModelUnavailable(
            f"the downloaded {spec.title} is not the expected file "
            f"({size} bytes, SHA-256 {digest.hexdigest()}; expected {spec.size} bytes, "
            f"SHA-256 {spec.sha256}). It was not saved."
        )
