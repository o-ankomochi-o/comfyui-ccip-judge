"""DWPose ONNX wrapper for keypoint extraction.

Replicates the SimCC-decoded inference path used in the original
LoRA evaluation notebook (yzd-v/DWPose: yolox_l + dw-ll_ucoco_384).
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


def _get_sessions():
    global _DET_SESSION, _POSE_SESSION
    if _DET_SESSION is not None and _POSE_SESSION is not None:
        return _DET_SESSION, _POSE_SESSION

    if ort is None:
        raise RuntimeError("onnxruntime is required for DWPose. pip install onnxruntime")
    if cv2 is None:
        raise RuntimeError("opencv-python is required for DWPose. pip install opencv-python")

    from huggingface_hub import hf_hub_download

    det_path = hf_hub_download(repo_id="yzd-v/DWPose", filename="yolox_l.onnx")
    pose_path = hf_hub_download(repo_id="yzd-v/DWPose", filename="dw-ll_ucoco_384.onnx")

    providers = ort.get_available_providers()
    chosen = []
    if "CUDAExecutionProvider" in providers:
        chosen.append("CUDAExecutionProvider")
    chosen.append("CPUExecutionProvider")

    _DET_SESSION = ort.InferenceSession(det_path, providers=chosen)
    _POSE_SESSION = ort.InferenceSession(pose_path, providers=chosen)
    return _DET_SESSION, _POSE_SESSION


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


def extract_pose(pil_image: Image.Image) -> Optional[dict]:
    """Run YOLOX person detection + DWPose keypoints on a PIL image.

    Returns dict {'keypoints': (133, 2), 'scores': (133,), 'bbox': (4,), 'image_shape': (h, w)}
    or None on failure.
    """
    if pil_image is None:
        return None
    det_session, pose_session = _get_sessions()

    img_bgr = pil_to_cv2_bgr(pil_image)
    boxes, _ = _detect_person(img_bgr, det_session)
    if len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))
    bbox = boxes[best_idx]

    kp, sc = _detect_keypoints(img_bgr, bbox, pose_session)
    if kp is None:
        return None
    return {
        "keypoints": kp,
        "scores": sc,
        "bbox": bbox,
        "image_shape": img_bgr.shape[:2],
    }
