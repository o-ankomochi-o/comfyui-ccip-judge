"""DWPose ONNX wrapper for keypoint extraction.

Replicates the SimCC-decoded inference path used in the original
LoRA evaluation notebook (yzd-v/DWPose: yolox_l + dw-ll_ucoco_384),
with one addition: when the legacy result's keypoint confidences
collapse, detection is retried with an anime-trained detector
(dghs-imgutils, already a dependency for CCIP). DWPose's YOLOX is
trained on photos and misses anime characters — especially line art /
flat colors — which used to fall back to a full-image bbox, collapse
the keypoint confidences below the 0.3 gate, and turn every image into
fail_score (OKS=0.0 / Angle=1.0 -> auto-dislike). Images the legacy
path already handles keep byte-identical results (regression-free).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from threading import Lock
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .common import pil_image_sha256, pil_to_cv2_bgr

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


_DET_SESSION = None
_POSE_SESSION = None
_POSE_RESULT_CACHE = OrderedDict()
_POSE_RESULT_CACHE_LOCK = Lock()
_POSE_RESULT_CACHE_MAX = 256

DWPOSE_REPO_ID = "yzd-v/DWPose"
DWPOSE_REVISION = "f7c16a3d45ad3783db41471848c80fbc281cabac"
YOLOX_MODEL_SHA256 = "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411"
POSE_MODEL_SHA256 = "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843"

POSE_METHOD_A = "A"
POSE_METHOD_B = "B"
POSE_METHOD_B_PRIME = "Bprime"
POSE_METHODS = (POSE_METHOD_A, POSE_METHOD_B, POSE_METHOD_B_PRIME)

ANIME_DETECT_LEVEL = "m"
ANIME_DETECT_VERSION = "v1.1"
ANIME_DETECT_REPO_ID = "deepghs/anime_person_detection"
ANIME_DETECT_MODEL_NAME = (
    f"person_detect_{ANIME_DETECT_VERSION}_{ANIME_DETECT_LEVEL}"
)
ANIME_DETECT_REVISION = "3f1b8fb2c369858927a9f1ee52bea656acc3be16"


@dataclass(frozen=True)
class PoseDiagnostics:
    method: str
    detector_used: str = "none"
    failure_reason: str = ""
    confident_keypoints: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "detector_used": self.detector_used,
            "failure_reason": self.failure_reason,
            "confident_keypoints": self.confident_keypoints,
            "bbox": list(self.bbox) if self.bbox is not None else None,
        }


def normalize_pose_method(method: str) -> str:
    value = str(method or POSE_METHOD_A).strip().lower().replace("'", "")
    aliases = {
        "a": POSE_METHOD_A,
        "current": POSE_METHOD_A,
        "0.3.0": POSE_METHOD_A,
        "b": POSE_METHOD_B,
        "anime": POSE_METHOD_B,
        "anime_only": POSE_METHOD_B,
        "bprime": POSE_METHOD_B_PRIME,
        "b_prime": POSE_METHOD_B_PRIME,
        "anime_first": POSE_METHOD_B_PRIME,
    }
    if value not in aliases:
        raise ValueError(f"unknown pose method {method!r}; expected one of {POSE_METHODS}")
    return aliases[value]


def _providers():
    chosen = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        chosen.append("CUDAExecutionProvider")
    chosen.append("CPUExecutionProvider")
    return chosen


def _require_runtime():
    if ort is None:
        raise RuntimeError("onnxruntime is required for DWPose. pip install onnxruntime")
    if cv2 is None:
        raise RuntimeError(
            "opencv-contrib-python is required for DWPose. "
            "pip install opencv-contrib-python"
        )


def _get_pose_session():
    global _POSE_SESSION
    if _POSE_SESSION is None:
        _require_runtime()
        from huggingface_hub import hf_hub_download
        pose_path = hf_hub_download(
            repo_id=DWPOSE_REPO_ID,
            filename="dw-ll_ucoco_384.onnx",
            revision=DWPOSE_REVISION,
        )
        _POSE_SESSION = ort.InferenceSession(pose_path, providers=_providers())
    return _POSE_SESSION


def _get_det_session():
    """Return the cached photo-trained YOLOX session.

    Method A calls this first for exact 0.3.0 compatibility. Method B never
    calls it. Method Bprime calls it only after the anime detector fails.
    """
    global _DET_SESSION
    if _DET_SESSION is None:
        _require_runtime()
        from huggingface_hub import hf_hub_download
        det_path = hf_hub_download(
            repo_id=DWPOSE_REPO_ID,
            filename="yolox_l.onnx",
            revision=DWPOSE_REVISION,
        )
        _DET_SESSION = ort.InferenceSession(det_path, providers=_providers())
    return _DET_SESSION


def _detect_person_anime(
    pil_image: Image.Image, suppress_errors: bool = False
):
    """Largest person bbox from the anime-trained detector (dghs-imgutils).
    Returns None when imgutils is unavailable or nothing is detected."""
    try:
        from imgutils.detect import detect_person
    except ImportError as e:
        if suppress_errors:
            return None
        raise RuntimeError(
            "dghs-imgutils is required for the anime person detector"
        ) from e
    try:
        results = detect_person(
            pil_image,
            level=ANIME_DETECT_LEVEL,
            version=ANIME_DETECT_VERSION,
            model_name=ANIME_DETECT_MODEL_NAME,
        )
    except Exception as e:
        if suppress_errors:
            return None
        raise RuntimeError(f"anime person detector failed: {e}") from e
    boxes = [np.asarray(box, dtype=np.float32) for box, _label, _score in results]
    if not boxes:
        return None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    return boxes[int(np.argmax(areas))]


def _expand_bbox(bbox: np.ndarray, w: int, h: int, scale: float = 1.2) -> np.ndarray:
    """Pad a tight detection box so the pose model keeps some context
    (standard practice for top-down pose estimators)."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    return np.array([
        max(0.0, cx - bw / 2), max(0.0, cy - bh / 2),
        min(float(w), cx + bw / 2), min(float(h), cy + bh / 2),
    ], dtype=np.float32)


def _clip_valid_boxes(boxes: np.ndarray, scores: np.ndarray, w: int, h: int):
    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    scores = np.asarray(scores, dtype=np.float32)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid], scores[valid]


def _detect_person_legacy_raw(
    image_bgr: np.ndarray, det_session
) -> Tuple[np.ndarray, np.ndarray]:
    """The exact 0.3.0 raw-output path, retained only as method A."""
    h, w = image_bgr.shape[:2]
    input_size = (640, 640)
    ratio = min(input_size[0] / h, input_size[1] / w)
    resized = cv2.resize(image_bgr, (int(w * ratio), int(h * ratio)))
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    padded[: resized.shape[0], : resized.shape[1]] = resized
    img_input = padded.transpose(2, 0, 1)[None].astype(np.float32)

    outputs = det_session.run(None, {det_session.get_inputs()[0].name: img_input})[0]
    predictions = outputs[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy /= ratio

    person_scores = scores[:, 0]
    keep = person_scores > 0.3
    if not keep.any():
        # No detection. Callers decide what that means: the fail-explicit
        # path treats it as a real failure, the legacy A/B path substitutes
        # a full-image bbox (the old silent-fallback behaviour).
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    # Intentionally no clipping here: method A is the exact 0.3.0 baseline.
    return boxes_xyxy[keep], person_scores[keep]


def _demo_postprocess(outputs: np.ndarray, img_size=(640, 640), p6: bool = False):
    """Decode raw YOLOX grid/stride predictions (official ONNX algorithm)."""
    decoded = np.asarray(outputs, dtype=np.float32).copy()
    grids = []
    expanded_strides = []
    strides = [8, 16, 32, 64] if p6 else [8, 16, 32]
    for stride in strides:
        hsize, wsize = img_size[0] // stride, img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    grid = np.concatenate(grids, axis=1)
    expanded_stride = np.concatenate(expanded_strides, axis=1)
    if decoded.shape[1] != grid.shape[1]:
        raise RuntimeError(
            "unexpected YOLOX output anchor count "
            f"{decoded.shape[1]} (expected {grid.shape[1]})"
        )
    decoded[..., :2] = (decoded[..., :2] + grid) * expanded_stride
    decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * expanded_stride
    return decoded


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0.0, xx2 - xx1 + 1)
        ih = np.maximum(0.0, yy2 - yy1 + 1)
        inter = iw * ih
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = order[np.where(iou <= threshold)[0] + 1]
    return keep


def _multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    nms_threshold: float = 0.45,
    score_threshold: float = 0.1,
):
    detections = []
    for class_index in range(scores.shape[1]):
        class_scores = scores[:, class_index]
        valid = class_scores > score_threshold
        if not valid.any():
            continue
        valid_boxes = boxes[valid]
        valid_scores = class_scores[valid]
        keep = _nms(valid_boxes, valid_scores, nms_threshold)
        if keep:
            classes = np.full((len(keep), 1), class_index, dtype=np.float32)
            detections.append(
                np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], classes], axis=1
                )
            )
    if not detections:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(detections, axis=0)


def _detect_person_official(
    image_bgr: np.ndarray, det_session
) -> Tuple[np.ndarray, np.ndarray]:
    """Official YOLOX ONNX decode + class-aware NMS, person class only."""
    h, w = image_bgr.shape[:2]
    input_size = (640, 640)
    ratio = min(input_size[0] / h, input_size[1] / w)
    resized = cv2.resize(image_bgr, (int(w * ratio), int(h * ratio)))
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    padded[: resized.shape[0], : resized.shape[1]] = resized
    img_input = padded.transpose(2, 0, 1)[None].astype(np.float32)

    raw = det_session.run(None, {det_session.get_inputs()[0].name: img_input})[0]
    predictions = _demo_postprocess(raw, input_size)[0]
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy /= ratio

    detections = _multiclass_nms(boxes_xyxy, class_scores)
    person = detections[
        (detections[:, 5] == 0) & (detections[:, 4] > 0.3)
    ]
    if len(person) == 0:
        return _clip_valid_boxes([], [], w, h)
    return _clip_valid_boxes(person[:, :4], person[:, 4], w, h)


def _detect_keypoints(image_bgr: np.ndarray, bbox: np.ndarray, pose_session,
                      model_input_size=(288, 384)) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    h, w = crop.shape[:2]
    ratio = min(model_input_size[0] / w, model_input_size[1] / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(crop, (new_w, new_h))
    padded = np.zeros((model_input_size[1], model_input_size[0], 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    img_input = padded.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_input = (img_input - mean) / std
    img_input = img_input.transpose(2, 0, 1)[None].astype(np.float32)

    outputs = pose_session.run(None, {pose_session.get_inputs()[0].name: img_input})
    simcc_x = outputs[0]
    simcc_y = outputs[1]
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]
    if simcc_y.ndim == 3:
        simcc_y = simcc_y[0]

    x_indices = np.argmax(simcc_x, axis=1).astype(np.float32)
    y_indices = np.argmax(simcc_y, axis=1).astype(np.float32)
    x_scores = np.max(simcc_x, axis=1)
    y_scores = np.max(simcc_y, axis=1)
    scores = np.minimum(x_scores, y_scores)

    SIMCC_SPLIT_RATIO = 2.0
    keypoints_x = x_indices / SIMCC_SPLIT_RATIO
    keypoints_y = y_indices / SIMCC_SPLIT_RATIO
    keypoints = np.stack([keypoints_x, keypoints_y], axis=1)

    keypoints[:, 0] = keypoints[:, 0] / ratio + x1
    keypoints[:, 1] = keypoints[:, 1] / ratio + y1

    return keypoints, scores


# Below this many confident (>0.3) body keypoints the legacy result is
# considered a detection collapse and the anime-detector retry kicks in.
# OKS needs >=3 common keypoints; 6 leaves margin without touching images
# the legacy path already handles (validation set minimum was 9).
_RETRY_BELOW = 6


def _confident_kp(result: Optional[dict], conf: float = 0.3) -> int:
    if result is None:
        return 0
    sc = np.asarray(result["scores"])[:17]
    return int((sc > conf).sum())


def _pose_for_bbox(img_bgr: np.ndarray, bbox: np.ndarray) -> Optional[dict]:
    kp, sc = _detect_keypoints(img_bgr, bbox, _get_pose_session())
    if kp is None:
        return None
    return {
        "keypoints": kp,
        "scores": sc,
        "bbox": bbox,
        "image_shape": img_bgr.shape[:2],
    }


def _select_largest_box(boxes: np.ndarray) -> Optional[np.ndarray]:
    if len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(np.argmax(areas))]


def _pose_with_detector(
    img_bgr: np.ndarray, bbox: Optional[np.ndarray], detector: str
) -> Optional[dict]:
    if bbox is None:
        return None
    pose = _pose_for_bbox(img_bgr, bbox)
    if pose is not None:
        pose["detector_used"] = detector
    return pose


def _diagnostics(method: str, pose: Optional[dict], reason: str = ""):
    if pose is None:
        return PoseDiagnostics(method=method, failure_reason=reason or "no_person")
    bbox = tuple(float(v) for v in np.asarray(pose["bbox"]).tolist())
    count = _confident_kp(pose)
    return PoseDiagnostics(
        method=method,
        detector_used=pose.get("detector_used", "unknown"),
        failure_reason=reason if count < 3 else "",
        confident_keypoints=count,
        bbox=bbox,
    )


def _extract_pose_impl(
    pil_image: Image.Image,
    method: str,
    legacy_full_image_fallback: bool = False,
):
    if pil_image is None:
        return None, PoseDiagnostics(method=method, failure_reason="no_image")
    img_bgr = pil_to_cv2_bgr(pil_image)
    h, w = img_bgr.shape[:2]

    if method == POSE_METHOD_A:
        boxes, _ = _detect_person_legacy_raw(img_bgr, _get_det_session())
        if len(boxes) == 0 and legacy_full_image_fallback:
            boxes = np.array([[0, 0, w, h]], dtype=np.float32)
        legacy = _pose_with_detector(
            img_bgr, _select_largest_box(boxes), "legacy_yolox_raw"
        )
        n_legacy = _confident_kp(legacy)
        if legacy_full_image_fallback or n_legacy >= _RETRY_BELOW:
            return legacy, _diagnostics(method, legacy, "low_keypoint_confidence")

        anime_bbox = _detect_person_anime(pil_image, suppress_errors=True)
        retry = _pose_with_detector(
            img_bgr,
            _expand_bbox(anime_bbox, w, h) if anime_bbox is not None else None,
            "anime",
        )
        selected = retry if _confident_kp(retry) > n_legacy else legacy
        return selected, _diagnostics(method, selected, "low_keypoint_confidence")

    anime_bbox = _detect_person_anime(pil_image)
    anime = _pose_with_detector(
        img_bgr,
        _expand_bbox(anime_bbox, w, h) if anime_bbox is not None else None,
        "anime",
    )
    if method == POSE_METHOD_B:
        return anime, _diagnostics(method, anime, "low_keypoint_confidence")

    n_anime = _confident_kp(anime)
    if n_anime >= _RETRY_BELOW:
        return anime, _diagnostics(method, anime)

    boxes, _ = _detect_person_official(img_bgr, _get_det_session())
    official = _pose_with_detector(
        img_bgr, _select_largest_box(boxes), "official_yolox"
    )
    selected = official if _confident_kp(official) > n_anime else anime
    return selected, _diagnostics(method, selected, "low_keypoint_confidence")


def extract_pose_with_diagnostics(
    pil_image: Image.Image, method: str = POSE_METHOD_A
):
    """Extract pose plus stable diagnostics for A/B/Bprime re-judging."""
    normalized = normalize_pose_method(method)
    return _extract_pose_impl(pil_image, normalized)


def extract_pose(
    pil_image: Image.Image,
    use_anime_detector: bool = True,
    method: Optional[str] = None,
) -> Optional[dict]:
    """Run person detection + DWPose keypoints on a PIL image.

    The legacy path (photo-trained YOLOX) runs first so images it already
    handles keep byte-identical results. Only when its keypoint confidences
    collapse (< _RETRY_BELOW of 17 above 0.3 — the line-art failure signature
    that produced OKS=0.0 / Angle=1.0 mass dislikes) is the anime-trained
    detector (dghs-imgutils) tried, and the result with more confident
    keypoints wins. When neither detector finds a person the function returns
    None (fail-explicit) instead of silently estimating pose on a full-image
    bbox; use_anime_detector=False reproduces the old behaviour exactly,
    including that fallback (A/B validation).

    Returns dict {'keypoints': (133, 2), 'scores': (133,), 'bbox': (4,), 'image_shape': (h, w)}
    or None on failure.
    """
    selected_method = normalize_pose_method(method or POSE_METHOD_A)
    pose, _ = _extract_pose_impl(
        pil_image,
        selected_method,
        legacy_full_image_fallback=(method is None and not use_anime_detector),
    )
    return pose


def extract_pose_cached(
    pil_image: Image.Image,
    method: str = POSE_METHOD_A,
) -> Optional[dict]:
    """Share DWPose results between OKS and Angle nodes for identical pixels."""
    normalized = normalize_pose_method(method)
    key = normalized, pil_image_sha256(pil_image)
    with _POSE_RESULT_CACHE_LOCK:
        if key in _POSE_RESULT_CACHE:
            value = _POSE_RESULT_CACHE.pop(key)
            _POSE_RESULT_CACHE[key] = value
            return value
    value = extract_pose(pil_image, method=normalized)
    with _POSE_RESULT_CACHE_LOCK:
        _POSE_RESULT_CACHE[key] = value
        while len(_POSE_RESULT_CACHE) > _POSE_RESULT_CACHE_MAX:
            _POSE_RESULT_CACHE.popitem(last=False)
    return value


def clear_pose_cache():
    with _POSE_RESULT_CACHE_LOCK:
        _POSE_RESULT_CACHE.clear()
