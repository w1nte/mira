"""The model files of the animal stage: pinned downloads, verified before use.

Each file is downloaded once into ``animals.model_dir`` (default: next to the face models,
``<faces.model_root>/models/animals``). The download streams into a temporary file, checks size
and SHA-256 and only then moves it into place, so a broken or tampered download never ends up
being loaded. Without internet the error says where to put the file by hand.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelFile:
    name: str  # file name in the model folder
    url: str
    sha256: str
    size: int
    license: str
    title: str


#: YOLOX-s, trained on COCO (80 classes, 10 of them animals); Megvii, Apache-2.0
YOLOX_S = ModelFile(
    name="yolox_s.onnx",
    url="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx",
    sha256="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    size=35_858_002,
    license="Apache-2.0",
    title="YOLOX-s",
)

#: DINOv2 ViT-S/14 (Meta, Apache-2.0) as ONNX, pinned to one commit of onnx-community/dinov2-small
DINOV2_SMALL = ModelFile(
    name="dinov2_small.onnx",
    url=(
        "https://huggingface.co/onnx-community/dinov2-small/resolve/"
        "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d/onnx/model.onnx"
    ),
    sha256="f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a",
    size=88_532_934,
    license="Apache-2.0",
    title="DINOv2-small",
)

MODELS = (YOLOX_S, DINOV2_SMALL)


class ModelUnavailable(RuntimeError):
    """A model file is missing and could not be downloaded (offline, blocked, corrupted)."""


def default_model_dir(settings) -> Path:
    """``animals.model_dir``, else ``models/animals`` under the face model root."""
    if settings.animals.model_dir:
        return Path(settings.animals.model_dir).expanduser()
    faces = settings.faces
    root = Path(faces.model_root).expanduser() if faces.model_root else Path.home() / ".insightface"
    return root / "models" / "animals"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def ensure_model(model_dir: Path, spec: ModelFile, *, client=None) -> Path:
    """The path of ``spec`` in ``model_dir``; downloaded (and verified) if it is not there yet.

    A file that is present is checked too (hashing it takes a fraction of a second); a partial
    or foreign file is downloaded again.
    """
    target = Path(model_dir) / spec.name
    try:
        if target.stat().st_size == spec.size and sha256_of(target) == spec.sha256:
            return target
        log.warning("%s is not the expected file; downloading it again", target)
    except OSError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{uuid.uuid4().hex}.part")
    try:
        _download(spec, tmp, client)
        _replace(tmp, target)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    log.info("saved %s (%d bytes)", target, spec.size)
    return target


def _download(spec: ModelFile, tmp: Path, client=None) -> None:
    import httpx

    log.info("downloading %s from %s", spec.title, spec.url)
    digest = hashlib.sha256()
    size = 0
    own = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True)
    try:
        with http.stream("GET", spec.url) as resp, tmp.open("wb") as fh:
            resp.raise_for_status()
            for block in resp.iter_bytes(1 << 20):
                fh.write(block)
                digest.update(block)
                size += len(block)
    except httpx.HTTPError as exc:
        raise ModelUnavailable(
            f"cannot download the {spec.title} model ({exc}). Check the internet connection, "
            f"or download {spec.url} yourself and save it as {tmp.with_name(spec.name)} "
            f"(SHA-256 {spec.sha256})."
        ) from exc
    finally:
        if own:
            http.close()
    if size != spec.size or digest.hexdigest() != spec.sha256:
        raise ModelUnavailable(
            f"the downloaded {spec.title} model is not the expected file "
            f"({size} bytes, SHA-256 {digest.hexdigest()}; expected {spec.size} bytes, "
            f"SHA-256 {spec.sha256}). It was not saved."
        )


def _replace(tmp: Path, target: Path) -> None:
    import time

    for attempt in range(5):  # Windows: a virus scanner may hold the new file for a moment
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2 * (attempt + 1))
