"""compute_oks with task joint sets (A4) and failure taxonomy (A6).

The pilot incident produced only `detect_failed=pose`; diagnosing it took a
live SSH session. The scorer must say WHICH precondition broke and with what
counts, and portrait-style tasks must evaluate only the joints the framing
can contain.
"""

from __future__ import annotations

import numpy as np

from ccip_judge.oks_score import compute_oks_diag


def _pose(visible_idx, n=17, base=100.0):
    kp = np.zeros((n, 2), dtype=np.float32)
    sc = np.zeros(n, dtype=np.float32)
    for i in visible_idx:
        kp[i] = (base + i * 10, base + i * 5)
        sc[i] = 0.9
    return {"keypoints": kp, "scores": sc,
            "bbox": [0.0, 0.0, 400.0, 600.0]}


def test_reference_sparse_is_named_with_counts():
    # The exact incident: reference has 1 confident keypoint, generated 13.
    ref = _pose([0])
    gen = _pose(range(13))
    score, reason = compute_oks_diag(ref, gen)
    assert score is None
    assert reason.startswith("insufficient_common_keypoints")
    assert "ref=1" in reason and "gen=13" in reason and "common=1" in reason


def test_portrait_joint_set_scores_upper_body_pair():
    # Same upper-body pair fails under the full-body expectation logic only
    # if commons are too few; with the portrait set (face..wrists) the
    # comparison is legitimate and returns a real score.
    ref = _pose(range(11))
    gen = _pose(range(11))
    score, reason = compute_oks_diag(ref, gen, keypoint_set="portrait")
    assert reason == ""
    assert score is not None and score > 0.99   # identical poses

    # joints OUTSIDE the set must not affect the score
    gen2 = _pose(list(range(11)) + [15, 16])
    gen2["keypoints"][15] = (999, 999)
    score2, _ = compute_oks_diag(ref, gen2, keypoint_set="portrait")
    assert abs(score2 - score) < 1e-6


def test_generated_no_person_is_named():
    score, reason = compute_oks_diag(_pose(range(11)), None)
    assert score is None and reason == "generated_no_person"
