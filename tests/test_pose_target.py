"""PoseTarget: generation conditions and scoring truth from the SAME data.

Root-fix 2026-07-14 (evocomfy pilot incident): the judge re-ESTIMATED the
reference pose from a rendered image and got 1 confident keypoint -> a whole
night of NaN. When the pose is authored as OpenPose JSON, the judge must
read those keypoints directly. OpenPose BODY-18 and DWPose/COCO-17 disagree
in joint order and set, so the mapping is by NAME, never by index.
"""

from __future__ import annotations

import json

import numpy as np


def _portrait_json(tmp_path):
    # nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank,
    # Lhip, Lkne, Lank, Reye, Leye, Rear, Lear  (OpenPose BODY-18 order)
    pts = [
        (416, 190, 1), (416, 285, 1), (330, 300, 1), (285, 430, 1),
        (315, 555, 1), (502, 300, 1), (547, 430, 1), (517, 555, 1),
        (365, 545, 1), (350, 760, 0), (345, 1010, 0), (467, 545, 1),
        (482, 760, 0), (487, 1010, 0), (394, 175, 1), (438, 175, 1),
        (375, 188, 1), (457, 188, 1),
    ]
    flat = [v for p in pts for v in p]
    p = tmp_path / "pose.json"
    p.write_text(json.dumps({"canvas_width": 832, "canvas_height": 1152,
                             "people": [{"pose_keypoints_2d": flat}]}),
                 encoding="utf-8")
    return p


def test_openpose_json_maps_to_coco17_by_name(tmp_path):
    from ccip_judge.pose_target import load_openpose_json

    pose = load_openpose_json(str(_portrait_json(tmp_path)))
    kp = np.asarray(pose["keypoints"])
    sc = np.asarray(pose["scores"])
    assert kp.shape == (17, 2) and sc.shape == (17,)
    # COCO order: 0 nose, 1 Leye, 2 Reye, 5 Lsho, 6 Rsho, ...
    assert tuple(kp[0]) == (416, 190)      # nose
    assert tuple(kp[1]) == (438, 175)      # LEFT eye = OpenPose idx 15
    assert tuple(kp[2]) == (394, 175)      # RIGHT eye = OpenPose idx 14
    assert tuple(kp[5]) == (502, 300)      # LEFT shoulder = OpenPose idx 5
    assert tuple(kp[6]) == (330, 300)      # RIGHT shoulder = OpenPose idx 2
    # confidence 0 keypoints stay invisible (score 0), never fake-visible
    assert sc[14] == 0.0 and sc[15] == 0.0   # knees marked c=0 above
    assert sc[0] == 1.0
    # bbox spans the VISIBLE keypoints; source is recorded
    x1, y1, x2, y2 = pose["bbox"]
    assert x1 <= 285 and x2 >= 547 and y1 <= 175
    assert pose["source"] == "openpose_json"
    # neck (OpenPose 1) has no COCO slot and must not leak into any joint
    assert not any(tuple(k) == (416, 285) for k in kp)
