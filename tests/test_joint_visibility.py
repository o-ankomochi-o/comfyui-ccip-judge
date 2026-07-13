"""Joint visibility rules (calibrated 2026-07-13 on ju_fufu training set).

DWPose clamps out-of-frame joints to the image border: in the calibration
measurement 54% of framing-impossible ("phantom") joints sat within 2% of
the frame edge, at scores overlapping real joints (phantom ankle 0.179 vs
real shoulder 0.177). Score alone cannot separate them -- position can.
"""

from __future__ import annotations

import numpy as np

from ccip_judge.pose_target import visible_joints


def _kp_sc(entries, n=17):
    kp = np.zeros((n, 2), dtype=np.float32)
    sc = np.zeros(n, dtype=np.float32)
    for i, (x, y, s) in entries.items():
        kp[i] = (x, y)
        sc[i] = s
    return kp, sc


def test_border_clamped_joint_is_invisible_on_detected_pose():
    # ankle clamped to the bottom edge with a "confident" score: phantom
    kp, sc = _kp_sc({5: (400, 600, 0.9), 15: (468, 1150, 0.9)})
    pose = {"keypoints": kp, "scores": sc, "image_shape": (1152, 832)}
    vis = visible_joints(kp, sc, pose, score_threshold=0.3)
    assert vis[5]          # mid-frame joint counts
    assert not vis[15]     # border-clamped joint never counts


def test_authored_pose_keeps_border_coordinates():
    # an authored JSON legitimately places a joint at the canvas edge; only
    # DETECTOR output clamps out-of-frame joints there, so authored poses
    # (no image_shape) must skip the border rule
    kp, sc = _kp_sc({5: (400, 600, 1.0), 15: (468, 1150, 1.0)})
    pose = {"keypoints": kp, "scores": sc, "canvas": (832, 1152),
            "source": "openpose_json"}
    vis = visible_joints(kp, sc, pose, score_threshold=0.3)
    assert vis[5]
    assert vis[15]         # edge coordinate is a position, not a clamp
