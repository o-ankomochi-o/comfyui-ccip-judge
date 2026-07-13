"""Pose targets: score against the SAME data that conditions generation.

Root cause this closes (2026-07-14): when the pose is authored (OpenPose
JSON driving OpenPoseStudio), re-estimating the reference pose from a
rendered image adds an extraction step that can fail -- and did (a bust-shot
reference yielded 1 confident keypoint; every OKS became NaN). Authored
keypoints are ground truth; read them directly.

Provider model (only what is needed today is implemented):
- openpose_json  : authored keypoints -> pose dict, no estimation (here)
- image reference: legacy path via dwpose_runner.extract_pose (unchanged)
- rigged 3D scene: future provider; projecting rig joints through the camera
  yields the same dict, so it plugs in here without touching the scorers.

OpenPose BODY-18 and DWPose/COCO-17 disagree in joint order and set, so the
mapping is BY NAME (A2); the neck has no COCO slot and is dropped. Points
with confidence 0 keep score 0 -- invisible, never fake-visible (A3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: COCO-17 slot -> OpenPose BODY-18 index, by joint name
_COCO17_FROM_OPENPOSE18 = {
    0: 0,    # nose
    1: 15,   # left eye
    2: 14,   # right eye
    3: 17,   # left ear
    4: 16,   # right ear
    5: 5,    # left shoulder
    6: 2,    # right shoulder
    7: 6,    # left elbow
    8: 3,    # right elbow
    9: 7,    # left wrist
    10: 4,   # right wrist
    11: 11,  # left hip
    12: 8,   # right hip
    13: 12,  # left knee
    14: 9,   # right knee
    15: 13,  # left ankle
    16: 10,  # right ankle
}

#: task-specific evaluable joint sets (A4): COCO indices
KEYPOINT_SETS = {
    "portrait": tuple(range(0, 11)),   # face + shoulders + elbows + wrists
    "full_body": tuple(range(0, 17)),
}


def load_openpose_json(path: str) -> dict:
    """Authored OpenPose BODY-18 JSON -> the pose dict the scorers consume
    ({keypoints (17,2), scores (17,), bbox, source}). Raises on malformed
    input -- an authored reference that cannot be parsed is a setup error,
    not a soft failure."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    people = data.get("people") or []
    if not people:
        raise ValueError(f"openpose json has no people: {path}")
    flat = people[0].get("pose_keypoints_2d") or []
    if len(flat) < 18 * 3:
        raise ValueError(
            f"openpose json needs 18 body keypoints (x,y,c), got "
            f"{len(flat) // 3}: {path}")
    pts = np.asarray(flat, dtype=np.float32).reshape(-1, 3)
    kp = np.zeros((17, 2), dtype=np.float32)
    sc = np.zeros(17, dtype=np.float32)
    for coco_i, op_i in _COCO17_FROM_OPENPOSE18.items():
        kp[coco_i] = pts[op_i, :2]
        sc[coco_i] = pts[op_i, 2]
    vis = kp[sc > 0]
    if len(vis) == 0:
        raise ValueError(f"openpose json has no visible keypoints: {path}")
    bbox = [float(vis[:, 0].min()), float(vis[:, 1].min()),
            float(vis[:, 0].max()), float(vis[:, 1].max())]
    return {"keypoints": kp, "scores": sc, "bbox": bbox,
            "canvas": (data.get("canvas_width"), data.get("canvas_height")),
            "source": "openpose_json"}
