"""Face age estimation with face-api.js' AgeGenderNet, run through onnxruntime.

InsightFace's ``genderage`` model pulls every face towards ~25-35 years: babies and children
come out as young adults and old people far too young. AgeGenderNet (a small Xception, 430 KB,
MIT licensed, from https://github.com/vladmandic/face-api) was trained on datasets with many
children and old people (UTKFace, FG-NET, MegaAge, …) and is far better at both ends.

The network is published as quantised TensorFlow.js weights. :func:`build_onnx` rebuilds the
exact graph (TinyXception with two middle blocks, TF "SAME" padding) as an ONNX model once, so it
runs on the same onnxruntime (CUDA or CPU) as InsightFace. No TensorFlow needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

WEIGHTS_URL = "https://raw.githubusercontent.com/vladmandic/face-api/master/model/"
MANIFEST = "age_gender_model-weights_manifest.json"
WEIGHTS = "age_gender_model.bin"
CHECKSUMS = {  # SHA-256 of the published files (face-api 1.7.x); downloads are verified
    MANIFEST: "74b60bf3ebf981bf4e661e9cca3a185614e10b25d797118b62e2a68d3ff2569f",
    WEIGHTS: "49f92bbc0afef2342c4385e4a2b0acb6d7a727c6f0aa539499719a81acf9abaa",
}
ONNX_NAME = "age_gender_tinyxception.onnx"
INPUT_SIZE = 112
MEAN_RGB = (122.782, 117.001, 104.298)
MODEL_VERSION = "agegender-tx2-v1"


# --------------------------------------------------------------------------- weights
def _dequantize(raw: bytes, offset: int, spec: dict) -> tuple[np.ndarray, int]:
    shape = spec["shape"]
    count = int(np.prod(shape)) if shape else 1
    q = spec.get("quantization")
    if q:
        dtype = {"uint8": np.uint8, "uint16": np.uint16}[q["dtype"]]
        n = count * np.dtype(dtype).itemsize
        data = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).astype(np.float32)
        data = data * np.float32(q["scale"]) + np.float32(q["min"])
    else:
        n = count * 4
        data = np.frombuffer(raw, dtype=np.float32, count=count, offset=offset).copy()
    return data.reshape(shape), offset + n


def load_tfjs_weights(manifest_path: Path, weights_path: Path) -> dict[str, np.ndarray]:
    manifest = json.loads(manifest_path.read_text())
    raw = weights_path.read_bytes()
    out: dict[str, np.ndarray] = {}
    offset = 0
    for group in manifest:
        for spec in group["weights"]:
            out[spec["name"]], offset = _dequantize(raw, offset, spec)
    if offset != len(raw):
        raise ValueError(f"weights file size mismatch ({offset} != {len(raw)})")
    return out


# --------------------------------------------------------------------------- ONNX graph
def build_onnx(weights: dict[str, np.ndarray]):
    """Rebuild AgeGenderNet as an ONNX model (input: N x 3 x 112 x 112, normalised RGB)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes: list = []
    inits: list = []
    counter = [0]

    def name(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    def const(arr: np.ndarray, prefix: str) -> str:
        n = name(prefix)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), n))
        return n

    def conv(x: str, w_tf: np.ndarray, b: np.ndarray | None, stride: int, group: int = 1) -> str:
        # TF filters: [h, w, in, out]; depthwise: [h, w, in, multiplier] -> [in * m, 1, h, w]
        w = w_tf.transpose(3, 2, 0, 1) if group == 1 else w_tf.transpose(2, 3, 0, 1)
        inputs = [x, const(w, "w")]
        if b is not None:
            inputs.append(const(b, "b"))
        out = name("conv")
        nodes.append(
            helper.make_node(
                "Conv",
                inputs,
                [out],
                strides=[stride, stride],
                auto_pad="SAME_UPPER",  # TF "same": extra padding at the end
                group=group,
                kernel_shape=list(w.shape[2:]),
            )
        )
        return out

    def relu(x: str) -> str:
        out = name("relu")
        nodes.append(helper.make_node("Relu", [x], [out]))
        return out

    def add(a: str, b: str) -> str:
        out = name("add")
        nodes.append(helper.make_node("Add", [a, b], [out]))
        return out

    def sep_conv(x: str, prefix: str) -> str:
        dw = weights[f"{prefix}/depthwise_filter"]
        pw = weights[f"{prefix}/pointwise_filter"]
        bias = weights[f"{prefix}/bias"]
        y = conv(x, dw, None, 1, group=dw.shape[2])
        return conv(y, pw, bias, 1)

    def reduction(x: str, prefix: str, activate_input: bool = True) -> str:
        out = relu(x) if activate_input else x
        out = sep_conv(out, f"{prefix}/separable_conv0")
        out = sep_conv(relu(out), f"{prefix}/separable_conv1")
        pooled = name("pool")
        nodes.append(
            helper.make_node(
                "MaxPool",
                [out],
                [pooled],
                kernel_shape=[3, 3],
                strides=[2, 2],
                auto_pad="SAME_UPPER",
            )
        )
        exp = conv(
            x,
            weights[f"{prefix}/expansion_conv/filters"],
            weights[f"{prefix}/expansion_conv/bias"],
            2,
        )
        return add(pooled, exp)

    def main_block(x: str, prefix: str) -> str:
        out = sep_conv(relu(x), f"{prefix}/separable_conv0")
        out = sep_conv(relu(out), f"{prefix}/separable_conv1")
        out = sep_conv(relu(out), f"{prefix}/separable_conv2")
        return add(out, x)

    x = relu(
        conv(
            "input",
            weights["entry_flow/conv_in/filters"],
            weights["entry_flow/conv_in/bias"],
            2,
        )
    )
    x = reduction(x, "entry_flow/reduction_block_0", activate_input=False)
    x = reduction(x, "entry_flow/reduction_block_1")
    x = main_block(x, "middle_flow/main_block_0")
    x = main_block(x, "middle_flow/main_block_1")
    x = reduction(x, "exit_flow/reduction_block")
    x = relu(sep_conv(x, "exit_flow/separable_conv"))
    pooled = name("gap")
    nodes.append(helper.make_node("GlobalAveragePool", [x], [pooled]))
    flat = name("flat")
    nodes.append(helper.make_node("Flatten", [pooled], [flat], axis=1))
    nodes.append(
        helper.make_node(
            "Gemm",
            [
                flat,
                const(weights["fc/age/weights"], "age_w"),
                const(weights["fc/age/bias"], "age_b"),
            ],
            ["age"],
        )
    )
    gender_logits = name("gender")
    nodes.append(
        helper.make_node(
            "Gemm",
            [
                flat,
                const(weights["fc/gender/weights"], "gen_w"),
                const(weights["fc/gender/bias"], "gen_b"),
            ],
            [gender_logits],
        )
    )
    nodes.append(helper.make_node("Softmax", [gender_logits], ["gender"], axis=1))
    graph = helper.make_graph(
        nodes,
        "agegender_tinyxception",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["N", 3, INPUT_SIZE, INPUT_SIZE]
            )
        ],
        [
            helper.make_tensor_value_info("age", TensorProto.FLOAT, ["N", 1]),
            helper.make_tensor_value_info("gender", TensorProto.FLOAT, ["N", 2]),
        ],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


# --------------------------------------------------------------------------- preprocessing
def face_crop(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    """The face box, padded to a square with black and resized to 112 px (as face-api does)."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, round(x1)), max(0, round(y1))
    x2, y2 = min(image.width, round(x2)), min(image.height, round(y2))
    crop = image.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))).convert("RGB")
    side = max(crop.width, crop.height)
    square = Image.new("RGB", (side, side), (0, 0, 0))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)


def to_tensor(crops: list[Image.Image], flip: bool = False) -> np.ndarray:
    arr = np.stack([np.asarray(c, dtype=np.float32) for c in crops])  # N, H, W, 3
    if flip:
        arr = arr[:, :, ::-1, :]
    arr = (arr - np.array(MEAN_RGB, dtype=np.float32)) / 256.0
    return np.ascontiguousarray(arr.transpose(0, 3, 1, 2))


# --------------------------------------------------------------------------- runtime
class AgeNet:
    """Lazy-loading age estimator. ``predict`` returns ages in years (float)."""

    def __init__(self, model_dir: Path, providers: list[str] | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.providers = providers or ["CPUExecutionProvider"]
        self._session = None
        self._lock = threading.Lock()

    @property
    def onnx_path(self) -> Path:
        return self.model_dir / ONNX_NAME

    def ensure_model(self) -> Path:
        if self.onnx_path.exists():
            return self.onnx_path
        self.model_dir.mkdir(parents=True, exist_ok=True)
        manifest, weights = self.model_dir / MANIFEST, self.model_dir / WEIGHTS
        for target in (manifest, weights):
            if not target.exists():
                _download(WEIGHTS_URL + target.name, target)
        model = build_onnx(load_tfjs_weights(manifest, weights))
        tmp = self.onnx_path.with_suffix(".tmp")
        tmp.write_bytes(model.SerializeToString())
        tmp.replace(self.onnx_path)
        log.info("built %s", self.onnx_path)
        return self.onnx_path

    def load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort

            path = self.ensure_model()
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(str(path), opts, providers=self.providers)

    def predict(self, crops: list[Image.Image], tta: bool = True) -> np.ndarray:
        """Ages for 112 x 112 crops (see ``face_crop``); ``tta`` also averages the mirrored face."""
        if not crops:
            return np.zeros(0, dtype=np.float32)
        self.load()
        assert self._session is not None
        ages = self._session.run(["age"], {"input": to_tensor(crops)})[0][:, 0]
        if tta:
            flipped = self._session.run(["age"], {"input": to_tensor(crops, flip=True)})[0][:, 0]
            ages = (ages + flipped) / 2.0
        return np.clip(ages, 0.0, 100.0)

    def predict_faces(
        self, image: Image.Image, bboxes: list[tuple[float, float, float, float]]
    ) -> list[float]:
        return [float(a) for a in self.predict([face_crop(image, b) for b in bboxes])]


def _download(url: str, target: Path) -> None:
    import httpx

    log.info("downloading %s", url)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.content
    digest = hashlib.sha256(data).hexdigest()
    expected = CHECKSUMS.get(target.name)
    if expected and digest != expected:
        raise ValueError(f"checksum mismatch for {url}: {digest}")
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)
    log.info("saved %s (%d bytes)", target, len(data))


# --------------------------------------------------------------------------- combining models
def blend_age(insightface_age: float | None, agenet_age: float | None) -> float | None:
    """Combine both estimates where each is strong.

    AgeNet is reliable for babies, children and teenagers, where InsightFace often says 25-45;
    InsightFace is better for old people, whom AgeNet makes too young. So children and teens
    (by AgeNet's reading) use AgeNet alone; from about 35 on both count equally.
    """
    if agenet_age is None:
        return insightface_age
    if insightface_age is None:
        return agenet_age
    w = min(1.0, max(0.0, (agenet_age - 18.0) / (35.0 - 18.0))) * 0.5
    return (1.0 - w) * agenet_age + w * insightface_age


def default_model_dir(cfg) -> Path:
    root = Path(cfg.model_root) if cfg.model_root else Path.home() / ".insightface"
    return Path(cfg.age_model_dir) if cfg.age_model_dir else root / "models" / "agegender_faceapi"


class FaceAgeEstimator:
    """Face ages as configured by ``faces.age_model`` ("blend", "agenet" or "insightface").

    ``estimate`` takes face boxes in ``image`` pixels. InsightFace's own estimate can be passed
    in (the faces stage has it already); otherwise its genderage model is run on the boxes.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mode = getattr(cfg, "age_model", "blend")
        self._agenet: AgeNet | None = None
        self._genderage = None
        self._lock = threading.Lock()

    def _providers(self) -> list[str]:
        from .analyzer import available_providers

        return available_providers(self.cfg.providers)

    @property
    def uses_agenet(self) -> bool:
        return self.mode in ("blend", "agenet")

    def load(self) -> None:
        if self.uses_agenet and self._agenet is None:
            self._agenet = AgeNet(default_model_dir(self.cfg), self._providers())
            self._agenet.load()

    def unload(self) -> None:
        self._agenet = None
        self._genderage = None

    def _insightface_ages(
        self, image: Image.Image, boxes: list[tuple[float, float, float, float]]
    ) -> list[float | None]:
        with self._lock:
            if self._genderage is None:
                from insightface.model_zoo import model_zoo

                root = (
                    Path(self.cfg.model_root)
                    if self.cfg.model_root
                    else Path.home() / ".insightface"
                )
                path = root / "models" / self.cfg.model / "genderage.onnx"
                if not path.exists():
                    from insightface.app import FaceAnalysis

                    FaceAnalysis(name=self.cfg.model, root=str(root), allowed_modules=["detection"])
                model = model_zoo.get_model(str(path), providers=self._providers())
                model.prepare(ctx_id=0 if "CUDAExecutionProvider" in self._providers() else -1)
                self._genderage = model
        from insightface.app.common import Face as IFace

        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        out: list[float | None] = []
        for box in boxes:
            face = IFace(bbox=np.asarray(box, dtype=np.float32))
            _gender, age = self._genderage.get(bgr, face)
            out.append(float(age))
        return out

    def estimate(
        self,
        image: Image.Image,
        boxes: list[tuple[float, float, float, float]],
        insightface_ages: list[float | None] | None = None,
    ) -> list[int | None]:
        if not boxes:
            return []
        if self.mode != "agenet" and insightface_ages is None:
            insightface_ages = self._insightface_ages(image, boxes)
        if not self.uses_agenet:
            return [None if a is None else round(a) for a in insightface_ages or []]
        self.load()
        assert self._agenet is not None
        net = self._agenet.predict_faces(image, boxes)
        if self.mode == "agenet":
            return [round(a) for a in net]
        assert insightface_ages is not None
        out: list[int | None] = []
        for raw, a in zip(insightface_ages, net, strict=True):
            b = blend_age(raw, a)
            out.append(None if b is None else round(b))
        return out
