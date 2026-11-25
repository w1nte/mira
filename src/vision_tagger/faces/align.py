"""Face alignment: put the eyes, nose and mouth of every face at the same position.

Aligned portraits can be cross-faded into smooth "morph" animations and compared side by side.
Uses the 5 InsightFace landmarks (eyes, nose tip, mouth corners) stored with each face and a
similarity transform (rotation + uniform scale + translation) onto the ArcFace template.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# ArcFace reference points for a 112x112 crop: left eye, right eye, nose, mouth left/right
ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def template(size: int, zoom: float = 0.42) -> np.ndarray:
    """ARCFACE_112 scaled into a ``size`` x ``size`` portrait with room for hair and chin.

    ``zoom`` is the width of the 112-px face box relative to the output.
    """
    scale = size * zoom / 112.0
    center = np.array([56.0, 72.0], dtype=np.float32)  # between eyes and mouth
    target = np.array([size / 2.0, size * 0.50], dtype=np.float32)
    return (ARCFACE_112 - center) * scale + target


@dataclass
class Aligned:
    image: Image.Image  # RGB, size x size
    sharpness: float
    frontal: float  # 1 = looking straight, 0 = profile


def frontalness(landmarks: np.ndarray) -> float:
    """How frontal a face is, from the nose position between the eyes (no image needed)."""
    le, re, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist <= 1e-3:
        return 0.0
    mid = (le + re) / 2.0
    axis = (re - le) / eye_dist
    yaw = float(np.dot(nose - mid, axis)) / eye_dist  # ~0 frontal, ~±0.5 profile
    return float(max(0.0, 1.0 - 2.2 * abs(yaw)))


def _sharpness(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalize_tone(rgb: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Pull brightness / contrast towards a common target so a sequence does not flicker."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lum = lab[:, :, 0]
    h, w = lum.shape
    core = lum[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]  # the face, not the background
    mean, std = float(core.mean()), float(core.std()) or 1.0
    target_mean, target_std = 150.0, 42.0
    adjusted = (lum - mean) * (target_std / std) + target_mean
    lab[:, :, 0] = np.clip(lum * (1 - strength) + adjusted * strength, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def align_face(
    image: Image.Image,
    landmarks: list[list[float]] | np.ndarray | None,
    bbox: tuple[float, float, float, float],
    size: int = 512,
    zoom: float = 0.42,
    tone: bool = True,
) -> Aligned:
    """Return an aligned ``size`` x ``size`` portrait (landmarks and bbox in ``image`` pixels)."""
    rgb = np.asarray(image.convert("RGB"))
    dst = template(size, zoom)
    if landmarks is not None and len(landmarks) == 5:
        src = np.asarray(landmarks, dtype=np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        frontal = frontalness(src)
    else:
        matrix, frontal = None, 0.5
    if matrix is None:  # no landmarks (or degenerate): centre the bounding box instead
        x1, y1, x2, y2 = bbox
        side = max(x2 - x1, y2 - y1)
        scale = (size * zoom) / max(side, 1.0)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        matrix = np.array(
            [[scale, 0, size / 2.0 - scale * cx], [0, scale, size * 0.50 - scale * cy]],
            dtype=np.float32,
        )
    warped = cv2.warpAffine(
        rgb, matrix, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    sharp = _sharpness(warped)
    if tone:
        warped = normalize_tone(warped)
    return Aligned(Image.fromarray(warped), sharpness=sharp, frontal=frontal)
