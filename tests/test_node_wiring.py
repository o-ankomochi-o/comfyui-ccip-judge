"""Node-level wiring for pose targets and failure reasons (A1/A5/A6).

- OKSScore: reference_pose_json bypasses reference-image estimation entirely;
  a reasons list rides along with the scores.
- Strictness: when a pose JSON is given, a broken one raises -- no silent
  image fallback inside an experiment (fallback is a config decision, not a
  node's own initiative).
"""

from __future__ import annotations

import json
import math

import numpy as np


def _write_pose_json(tmp_path):
    pts = [(416, 190, 1), (416, 285, 1), (330, 300, 1), (285, 430, 1),
           (315, 555, 1), (502, 300, 1), (547, 430, 1), (517, 555, 1),
           (365, 545, 1), (350, 760, 1), (345, 1010, 1), (467, 545, 1),
           (482, 760, 1), (487, 1010, 1), (394, 175, 1), (438, 175, 1),
           (375, 188, 1), (457, 188, 1)]
    p = tmp_path / "pose.json"
    p.write_text(json.dumps({"canvas_width": 832, "canvas_height": 1152,
                             "people": [{"pose_keypoints_2d":
                                         [v for q in pts for v in q]}]}),
                 encoding="utf-8")
    return p


def test_oks_node_scores_against_json_and_reports_reasons(tmp_path, monkeypatch):
    from ccip_judge import oks_score as m
    from ccip_judge.pose_target import load_openpose_json

    ref = load_openpose_json(str(_write_pose_json(tmp_path)))

    # generated: image A matches the json pose exactly; image B has no person
    match = {"keypoints": ref["keypoints"].copy(),
             "scores": np.where(ref["scores"] > 0, 0.9, 0.0),
             "bbox": ref["bbox"]}
    fakes = iter([match, None])
    monkeypatch.setattr(m, "extract_pose", lambda img: next(fakes))
    monkeypatch.setattr(m, "comfy_image_to_pil_list", lambda x: list(x))

    node = m.OKSScore()
    scores, passes, info, reasons = node.score(
        ["imgA", "imgB"], threshold=0.5, reference_folder="",
        fail_score=0.0, reference_pose_json=str(tmp_path / "pose.json"))
    assert scores[0] > 0.99 and passes[0] is True and reasons[0] == ""
    assert math.isnan(scores[1]) and passes[1] is False
    assert reasons[1] == "generated_no_person"
    assert "reference_source=openpose_json" in info


def test_oks_node_refuses_broken_json_no_silent_fallback(tmp_path, monkeypatch):
    import pytest

    from ccip_judge import oks_score as m

    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(m, "comfy_image_to_pil_list", lambda x: list(x))
    with pytest.raises(Exception):
        m.OKSScore().score(["img"], threshold=0.5, reference_folder="/refs",
                           fail_score=0.0, reference_pose_json=str(bad))


def test_angle_node_uses_json_reference(tmp_path, monkeypatch):
    # A5: angle features (face/shoulder geometry) computed straight from the
    # authored keypoints -- no reference-image estimation. Reasons ride along.
    from ccip_judge import angle_score as m
    from ccip_judge.pose_target import load_openpose_json

    ref = load_openpose_json(str(_write_pose_json(tmp_path)))
    match = {"keypoints": ref["keypoints"].copy(),
             "scores": np.where(ref["scores"] > 0, 0.9, 0.0),
             "bbox": ref["bbox"]}
    fakes = iter([match, None])
    monkeypatch.setattr(m, "extract_pose", lambda img: next(fakes))
    monkeypatch.setattr(m, "comfy_image_to_pil_list", lambda x: list(x))

    scores, passes, info, reasons = m.AngleScore().score(
        ["imgA", "imgB"], threshold=0.5, reference_folder="",
        fail_score=1.0, reference_pose_json=str(tmp_path / "pose.json"))
    assert scores[0] < 0.01 and passes[0] is True and reasons[0] == ""
    assert math.isnan(scores[1]) and reasons[1] == "generated_no_person"
    assert "reference_source=openpose_json" in info
