"""Anime-person detection plus DWPose keypoint extraction.

Version 0.5.0 uses the anime-trained detector from dghs-imgutils as the only
person-bbox source. A pre-Study-1 evaluation found identical pass decisions
for the legacy, anime-only and YOLOX-fallback strategies on all 1,526 unique
v10cf images. The legacy photo-trained YOLOX never supplied a successful
primary result, so it was removed as dead weight.

Detection and pose-estimation failures return ``None``. They are never replaced
with a full-image bbox or a numeric fail score.
"""

from __future__ import annotations

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


_POSE_SESSION = None
_POSE_RESULT_CACHE = OrderedDict()
_POSE_RESULT_CACHE_LOCK = Lock()
_POSE_RESULT_CACHE_MAX = 256

DWPOSE_REPO_ID = "yzd-v/DWPose"
DWPOSE_REVISION = "f7c16a3d45ad3783db41471848c80fbc281cabac"

ANIME_DETECT_LEVEL = "m"
ANIME_DETECT_VERSION = "v1.1"
ANIME_DETECT_MODEL_NAME = (
    f"person_detect_{ANIME_DETECT_VERSION}_{ANIME_DETECT_LEVEL}"
)


def _providers():
    chosen = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        chosen.append("CUDAExecutionProvider")
    chosen.append("CPUExecutionProvider")
    return chosen


def _require_runtime():
    if ort is None:
        raise RuntimeError(
            "onnxruntime is required for DWPose. pip install onnxruntime"
        )
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
        _POSE_SESSION = ort.InferenceSession(
            pose_path,
            providers=_providers(),
        )
    return _POSE_SESSION


def _detect_person_anime(pil_image: Image.Image) -> Optional[np.ndarray]:
    """Return the largest valid anime-person bbox, or ``None``."""
    try:
        from imgutils.detect import detect_person
    except ImportError as e:
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
        raise RuntimeError(f"anime person detector failed: {e}") from e

    boxes = []
    for box, _label, _score in results:
        candidate = np.asarray(box, dtype=np.float32)
        if (
            candidate.shape == (4,)
            and candidate[2] > candidate[0]
            and candidate[3] > candidate[1]
        ):
            boxes.append(candidate)
    if not boxes:
        return None
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    return boxes[int(np.argmax(areas))]


def _expand_bbox(
    bbox: np.ndarray,
    width: int,
    height: int,
    scale: float = 1.2,
) -> np.ndarray:
    """Pad a tight bbox so the top-down pose model retains context."""
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    box_width = (x2 - x1) * scale
    box_height = (y2 - y1) * scale
    return np.array(
        [
            max(0.0, center_x - box_width / 2),
            max(0.0, center_y - box_height / 2),
            min(float(width), center_x + box_width / 2),
            min(float(height), center_y + box_height / 2),
        ],
        dtype=np.float32,
    )


def _detect_keypoints(
    image_bgr: np.ndarray,
    bbox: np.ndarray,
    pose_session,
    model_input_size=(288, 384),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    x1, y1, x2, y2 = bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    height, width = crop.shape[:2]
    ratio = min(
        model_input_size[0] / width,
        model_input_size[1] / height,
    )
    new_width, new_height = int(width * ratio), int(height * ratio)
    resized = cv2.resize(crop, (new_width, new_height))
    padded = np.zeros(
        (model_input_size[1], model_input_size[0], 3),
        dtype=np.uint8,
    )
    padded[:new_height, :new_width] = resized

    image_input = padded.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_input = (image_input - mean) / std
    image_input = image_input.transpose(2, 0, 1)[None].astype(np.float32)

    outputs = pose_session.run(
        None,
        {pose_session.get_inputs()[0].name: image_input},
    )
    simcc_x, simcc_y = outputs[0], outputs[1]
    if simcc_x.ndim == 3:
        simcc_x = simcc_x[0]
    if simcc_y.ndim == 3:
        simcc_y = simcc_y[0]

    x_indices = np.argmax(simcc_x, axis=1).astype(np.float32)
    y_indices = np.argmax(simcc_y, axis=1).astype(np.float32)
    x_scores = np.max(simcc_x, axis=1)
    y_scores = np.max(simcc_y, axis=1)
    scores = np.minimum(x_scores, y_scores)

    simcc_split_ratio = 2.0
    keypoints = np.stack(
        [
            x_indices / simcc_split_ratio,
            y_indices / simcc_split_ratio,
        ],
        axis=1,
    )
    keypoints[:, 0] = keypoints[:, 0] / ratio + x1
    keypoints[:, 1] = keypoints[:, 1] / ratio + y1
    return keypoints, scores


def extract_pose(pil_image: Image.Image) -> Optional[dict]:
    """Extract DWPose keypoints using the anime detector's largest person."""
    if pil_image is None:
        return None

    bbox = _detect_person_anime(pil_image)
    if bbox is None:
        return None

    image_bgr = pil_to_cv2_bgr(pil_image)
    height, width = image_bgr.shape[:2]
    bbox = _expand_bbox(bbox, width, height)
    keypoints, scores = _detect_keypoints(
        image_bgr,
        bbox,
        _get_pose_session(),
    )
    if keypoints is None:
        return None
    return {
        "keypoints": keypoints,
        "scores": scores,
        "bbox": bbox,
        "image_shape": image_bgr.shape[:2],
        "detector_used": "anime",
    }


def extract_pose_cached(pil_image: Image.Image) -> Optional[dict]:
    """Share DWPose results between OKS and Angle for identical RGB pixels."""
    key = pil_image_sha256(pil_image)
    with _POSE_RESULT_CACHE_LOCK:
        if key in _POSE_RESULT_CACHE:
            value = _POSE_RESULT_CACHE.pop(key)
            _POSE_RESULT_CACHE[key] = value
            return value

    value = extract_pose(pil_image)
    with _POSE_RESULT_CACHE_LOCK:
        _POSE_RESULT_CACHE[key] = value
        while len(_POSE_RESULT_CACHE) > _POSE_RESULT_CACHE_MAX:
            _POSE_RESULT_CACHE.popitem(last=False)
    return value


def clear_pose_cache():
    with _POSE_RESULT_CACHE_LOCK:
        _POSE_RESULT_CACHE.clear()
