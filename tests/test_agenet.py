from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from vision_tagger.config import FacesConfig
from vision_tagger.faces import agenet as A

# --------------------------------------------------------------------------- reference model
# An independent NumPy implementation of the TensorFlow ops used by face-api.js' TinyXception
# (NHWC, "SAME" padding with the extra pixel at the end), to check the ONNX translation.


def _same_pad(size: int, k: int, stride: int) -> tuple[int, int]:
    out = -(-size // stride)
    total = max((out - 1) * stride + k - size, 0)
    return total // 2, total - total // 2


def _pad(x, k, stride, value=0.0):
    (t, b), (left, r) = _same_pad(x.shape[1], k, stride), _same_pad(x.shape[2], k, stride)
    return np.pad(x, ((0, 0), (t, b), (left, r), (0, 0)), constant_values=value)


def tf_conv(x, w, b, stride):  # w: [k, k, in, out]
    k = w.shape[0]
    xp = _pad(x, k, stride)
    n, h, wd, _ = x.shape
    oh, ow = -(-h // stride), -(-wd // stride)
    out = np.zeros((n, oh, ow, w.shape[3]), np.float64)
    for i in range(k):
        for j in range(k):
            patch = xp[:, i : i + stride * oh : stride, j : j + stride * ow : stride, :]
            out += patch @ w[i, j]
    return out + (b if b is not None else 0)


def tf_depthwise(x, w):  # w: [k, k, c, 1], stride 1
    k = w.shape[0]
    xp = _pad(x, k, 1)
    _n, h, wd, _ = x.shape
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(k):
        for j in range(k):
            out += xp[:, i : i + h, j : j + wd, :] * w[i, j, :, 0]
    return out


def tf_maxpool(x):
    xp = _pad(x, 3, 2, value=-np.inf)
    n, h, wd, c = x.shape
    oh, ow = -(-h // 2), -(-wd // 2)
    out = np.full((n, oh, ow, c), -np.inf)
    for i in range(3):
        for j in range(3):
            out = np.maximum(out, xp[:, i : i + 2 * oh : 2, j : j + 2 * ow : 2, :])
    return out


def reference_age(weights, x_nhwc):
    W = weights
    relu = lambda t: np.maximum(t, 0)  # noqa: E731

    def sep(t, p):
        t = tf_depthwise(t, W[f"{p}/depthwise_filter"])
        return tf_conv(t, W[f"{p}/pointwise_filter"], W[f"{p}/bias"], 1)

    def reduction(t, p, act=True):
        out = relu(t) if act else t
        out = sep(out, f"{p}/separable_conv0")
        out = sep(relu(out), f"{p}/separable_conv1")
        out = tf_maxpool(out)
        return out + tf_conv(t, W[f"{p}/expansion_conv/filters"], W[f"{p}/expansion_conv/bias"], 2)

    def main(t, p):
        out = sep(relu(t), f"{p}/separable_conv0")
        out = sep(relu(out), f"{p}/separable_conv1")
        out = sep(relu(out), f"{p}/separable_conv2")
        return out + t

    t = relu(tf_conv(x_nhwc, W["entry_flow/conv_in/filters"], W["entry_flow/conv_in/bias"], 2))
    t = reduction(t, "entry_flow/reduction_block_0", act=False)
    t = reduction(t, "entry_flow/reduction_block_1")
    t = main(t, "middle_flow/main_block_0")
    t = main(t, "middle_flow/main_block_1")
    t = reduction(t, "exit_flow/reduction_block")
    t = relu(sep(t, "exit_flow/separable_conv"))
    pooled = t.mean(axis=(1, 2))
    return pooled @ W["fc/age/weights"] + W["fc/age/bias"]


def random_weights(seed=0):
    rng = np.random.default_rng(seed)
    shapes = {"entry_flow/conv_in/filters": [3, 3, 3, 32], "entry_flow/conv_in/bias": [32]}

    def sep(p, cin, cout):
        shapes[f"{p}/depthwise_filter"] = [3, 3, cin, 1]
        shapes[f"{p}/pointwise_filter"] = [1, 1, cin, cout]
        shapes[f"{p}/bias"] = [cout]

    def red(p, cin, cout):
        sep(f"{p}/separable_conv0", cin, cout)
        sep(f"{p}/separable_conv1", cout, cout)
        shapes[f"{p}/expansion_conv/filters"] = [1, 1, cin, cout]
        shapes[f"{p}/expansion_conv/bias"] = [cout]

    red("entry_flow/reduction_block_0", 32, 64)
    red("entry_flow/reduction_block_1", 64, 128)
    for m in (0, 1):
        for c in (0, 1, 2):
            sep(f"middle_flow/main_block_{m}/separable_conv{c}", 128, 128)
    red("exit_flow/reduction_block", 128, 256)
    sep("exit_flow/separable_conv", 256, 512)
    shapes.update({"fc/age/weights": [512, 1], "fc/age/bias": [1]})
    shapes.update({"fc/gender/weights": [512, 2], "fc/gender/bias": [2]})
    return {k: (rng.normal(size=v) * 0.15).astype(np.float32) for k, v in shapes.items()}


def test_onnx_graph_matches_tensorflow_semantics():
    import onnxruntime as ort

    weights = random_weights()
    model = A.build_onnx(weights)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, A.INPUT_SIZE, A.INPUT_SIZE, 3)).astype(np.float32)
    got = session.run(["age", "gender"], {"input": x.transpose(0, 3, 1, 2).copy()})
    want = reference_age({k: v.astype(np.float64) for k, v in weights.items()}, x)
    assert got[0].shape == (2, 1)
    np.testing.assert_allclose(got[0], want, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(got[1].sum(axis=1), 1.0, rtol=1e-5)  # gender softmax


def test_tfjs_weights_are_dequantized(tmp_path):
    q = np.array([0, 128, 255], np.uint8)
    f = np.array([1.5, -2.0], np.float32)
    (tmp_path / "w.bin").write_bytes(q.tobytes() + f.tobytes())
    (tmp_path / "m.json").write_text(
        json.dumps(
            [
                {
                    "paths": ["w.bin"],
                    "weights": [
                        {
                            "name": "a",
                            "shape": [3],
                            "dtype": "float32",
                            "quantization": {"dtype": "uint8", "scale": 0.5, "min": -1.0},
                        },
                        {"name": "b", "shape": [2], "dtype": "float32"},
                    ],
                }
            ]
        )
    )
    w = A.load_tfjs_weights(tmp_path / "m.json", tmp_path / "w.bin")
    np.testing.assert_allclose(w["a"], [-1.0, 63.0, 126.5])
    np.testing.assert_allclose(w["b"], [1.5, -2.0])
    (tmp_path / "w.bin").write_bytes(q.tobytes() + f.tobytes() + b"xx")
    with pytest.raises(ValueError, match="size mismatch"):
        A.load_tfjs_weights(tmp_path / "m.json", tmp_path / "w.bin")


def test_face_crop_pads_to_square_like_face_api():
    im = Image.new("RGB", (200, 100), (255, 255, 255))
    crop = A.face_crop(im, (50, 10, 90, 90))  # 40 x 80 box -> black bars left and right
    assert crop.size == (A.INPUT_SIZE, A.INPUT_SIZE)
    arr = np.asarray(crop)
    assert arr[56, 2].max() < 10 and arr[56, 56].min() > 245
    x = A.to_tensor([crop])
    assert x.shape == (1, 3, A.INPUT_SIZE, A.INPUT_SIZE)
    np.testing.assert_allclose(x[0, :, 56, 56], (255 - np.array(A.MEAN_RGB)) / 256, rtol=1e-5)


def test_blend_uses_agenet_for_children_and_both_for_adults():
    assert A.blend_age(28, 3) == 3  # InsightFace calls a toddler 28
    assert A.blend_age(43, 12) == 12
    assert A.blend_age(78, 64) == pytest.approx(71)  # old age: both count equally
    assert 20 < A.blend_age(40, 26) < 30
    assert A.blend_age(None, 7) == 7 and A.blend_age(33, None) == 33


class _Net:
    def predict_faces(self, image, boxes):
        return [5.0 for _ in boxes]


@pytest.mark.parametrize(
    ("mode", "expected"), [("blend", [5]), ("agenet", [5]), ("insightface", [30])]
)
def test_estimator_modes(mode, expected, monkeypatch):
    est = A.FaceAgeEstimator(FacesConfig(age_model=mode, providers=["CPUExecutionProvider"]))
    monkeypatch.setattr(est, "load", lambda: setattr(est, "_agenet", _Net()))
    monkeypatch.setattr(est, "_insightface_ages", lambda im, boxes: [30.0] * len(boxes))
    im = Image.new("RGB", (64, 64))
    assert est.estimate(im, [(0, 0, 32, 32)]) == expected
    assert est.estimate(im, [(0, 0, 32, 32)], insightface_ages=[30]) == expected
    assert est.estimate(im, []) == []


def test_download_verifies_checksum(tmp_path, monkeypatch):
    import httpx

    def handler(request):
        return httpx.Response(200, content=b"not the model")

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )
    with pytest.raises(ValueError, match="checksum"):
        A._download("https://example.invalid/" + A.WEIGHTS, tmp_path / A.WEIGHTS)
    assert not (tmp_path / A.WEIGHTS).exists()


@pytest.mark.integration
def test_real_agenet_on_sample_faces(tmp_path):
    """Downloads the 430 KB model and checks it on InsightFace's sample photo (adults)."""
    from insightface.data import get_image

    from vision_tagger.faces.analyzer import FaceAnalyzer

    cfg = FacesConfig(providers=["CPUExecutionProvider"], age_model_dir=tmp_path)
    im = Image.fromarray(np.ascontiguousarray(get_image("t1")[:, :, ::-1]))
    faces = FaceAnalyzer(cfg).detect(im)
    ages = A.AgeNet(tmp_path).predict_faces(im, [f.bbox for f in faces])
    assert len(ages) >= 5
    # a cast of adults aged 28-34: AgeNet reads adults a little young, but never as children
    assert 18 <= float(np.median(ages)) <= 40, ages
    assert all(10 <= a <= 60 for a in ages), ages
    blended = A.FaceAgeEstimator(cfg).estimate(im, [f.bbox for f in faces], [f.age for f in faces])
    assert 20 <= float(np.median(blended)) <= 40 and all(10 <= a <= 60 for a in blended)
