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


def test_oks_measures_calibrated_confidence_shoulders():
    # the v4 preflight refusal: a generated portrait with both shoulders
    # detected at the CORRECT positions but confidence 0.17 -- below the old
    # uncalibrated 0.3 gate, above the calibrated one. It must be measured,
    # not refused as missing_required_joints.
    from ccip_judge.oks_score import compute_oks_diag

    kp = np.zeros((17, 2), dtype=np.float32)
    sc = np.zeros(17, dtype=np.float32)
    for i in range(11):
        kp[i] = (100 + i * 10, 100 + i * 5)
        sc[i] = 0.9
    ref = {"keypoints": kp.copy(), "scores": sc.copy(),
           "bbox": [0.0, 0.0, 832.0, 1152.0], "source": "openpose_json"}
    gen_sc = sc.copy()
    gen_sc[5] = gen_sc[6] = 0.17
    gen = {"keypoints": kp.copy(), "scores": gen_sc,
           "bbox": [0.0, 0.0, 832.0, 1152.0], "image_shape": (1152, 832)}
    score, reason = compute_oks_diag(ref, gen, keypoint_set="portrait")
    assert reason == ""
    assert score is not None and score > 0.99   # identical positions
