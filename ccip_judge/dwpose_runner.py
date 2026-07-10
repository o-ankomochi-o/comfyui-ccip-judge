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

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .common import pil_to_cv2_bgr

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
        raise RuntimeError("opencv-python is required for DWPose. pip install opencv-python")


def _get_pose_session():
    global _POSE_SESSION
    if _POSE_SESSION is None:
        _require_runtime()
        from huggingface_hub import hf_hub_download
        pose_path = hf_hub_download(repo_id="yzd-v/DWPose", filename="dw-ll_ucoco_384.onnx")
        _POSE_SESSION = ort.InferenceSession(pose_path, providers=_providers())
    return _POSE_SESSION


def _get_det_session():
    """YOLOX (photo-trained) — lazy, only loaded when the anime detector
    finds nothing, so the common path never pays for it."""
    global _DET_SESSION
    if _DET_SESSION is None:
        _require_runtime()
        from huggingface_hub import hf_hub_download
        det_path = hf_hub_download(repo_id="yzd-v/DWPose", filename="yolox_l.onnx")
        _DET_SESSION = ort.InferenceSession(det_path, providers=_providers())
    return _DET_SESSION


def _detect_person_anime(pil_image: Image.Image):
    """Largest person bbox from the anime-trained detector (dghs-imgutils).
    Returns None when imgutils is unavailable or nothing is detected."""
    try:
        from imgutils.detect import detect_person
    except ImportError:
        return None
    try:
        results = detect_person(pil_image)
    except Exception:
        return None  # detector failure falls back to YOLOX
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


def _detect_person(image_bgr: np.ndarray, det_session) -> Tuple[np.ndarray, np.ndarray]:
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
        return np.array([[0, 0, w, h]], dtype=np.float32), np.array([1.0], dtype=np.float32)
    return boxes_xyxy[keep], person_scores[keep]


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


def extract_pose(pil_image: Image.Image, use_anime_detector: bool = True) -> Optional[dict]:
    """Run person detection + DWPose keypoints on a PIL image.

    The legacy path (photo-trained YOLOX, full-image fallback) runs first so
    images it already handles keep byte-identical results. Only when its
    keypoint confidences collapse (< _RETRY_BELOW of 17 above 0.3 — the
    line-art failure signature that produced OKS=0.0 / Angle=1.0 mass
    dislikes) is the anime-trained detector (dghs-imgutils) tried, and the
    result with more confident keypoints wins. use_anime_detector=False
    reproduces the legacy behaviour exactly (A/B validation).

    Returns dict {'keypoints': (133, 2), 'scores': (133,), 'bbox': (4,), 'image_shape': (h, w)}
    or None on failure.
    """
    if pil_image is None:
        return None
    img_bgr = pil_to_cv2_bgr(pil_image)
    h, w = img_bgr.shape[:2]

    boxes, _ = _detect_person(img_bgr, _get_det_session())
    legacy = None
    if len(boxes) > 0:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        legacy = _pose_for_bbox(img_bgr, boxes[int(np.argmax(areas))])

    n_legacy = _confident_kp(legacy)
    if not use_anime_detector or n_legacy >= _RETRY_BELOW:
        return legacy

    anime_bbox = _detect_person_anime(pil_image)
    if anime_bbox is None:
        return legacy
    retry = _pose_for_bbox(img_bgr, _expand_bbox(anime_bbox, w, h))
    return retry if _confident_kp(retry) > n_legacy else legacy
