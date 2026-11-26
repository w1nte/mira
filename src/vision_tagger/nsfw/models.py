"""The model file of the NSFW check: a pinned download, verified before use (see
``animals.models``, which does the downloading)."""

from __future__ import annotations

from pathlib import Path

from ..animals.models import ModelFile, ModelUnavailable, ensure_model

#: Falconsai/nsfw_image_detection (a ViT-base/16 fine-tuned on 80,000 photos, two classes
#: "normal" and "nsfw"; Apache-2.0) as ONNX in half precision, pinned to one commit of
#: onnx-community/nsfw_image_detection-ONNX. The same scores as the full-precision model (the
#: 8-bit variants scored a harmless photo more than twice as high), half its download.
VIT_NSFW = ModelFile(
    name="nsfw_vit_fp16.onnx",
    url=(
        "https://huggingface.co/onnx-community/nsfw_image_detection-ONNX/resolve/"
        "1ceb3c7fe1e9f3f2507e6df577437f23a9149fd5/onnx/model_fp16.onnx"
    ),
    sha256="edf335fa01a597a65e339ca953609b3741477470e20aa3e683a0bdde104d8471",
    size=171_801_382,
    license="Apache-2.0",
    title="NSFW classifier (ViT)",
)

#: index of the "nsfw" class in the model's output
NSFW_CLASS = 1


def default_model_dir(settings) -> Path:
    """``nsfw.model_dir``, else ``models/nsfw`` under the face model root."""
    if settings.nsfw.model_dir:
        return Path(settings.nsfw.model_dir).expanduser()
    faces = settings.faces
    root = Path(faces.model_root).expanduser() if faces.model_root else Path.home() / ".insightface"
    return root / "models" / "nsfw"


__all__ = ["NSFW_CLASS", "VIT_NSFW", "ModelUnavailable", "default_model_dir", "ensure_model"]
