import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ccip_judge.dwpose_runner import PoseDiagnostics
from ccip_judge.rejudge import (
    REJUDGE_FIELDS,
    MethodReferences,
    RejudgeThresholds,
    _score_pose,
    rejudge_image,
    validate_judge_sha,
    write_rejudge_csv,
)


class RejudgeContractTests(unittest.TestCase):
    def test_output_has_exact_frozen_14_columns(self):
        pose = {
            "keypoints": np.zeros((133, 2), dtype=np.float32),
            "scores": np.ones(133, dtype=np.float32),
            "bbox": np.array([1, 2, 30, 40], dtype=np.float32),
        }
        diagnostics = PoseDiagnostics(
            method="B",
            detector_used="anime",
            confident_keypoints=17,
            bbox=(1, 2, 30, 40),
        )
        refs = MethodReferences(poses=[pose], angle_features=[{"x": 1}])
        with (
            patch(
                "ccip_judge.rejudge.extract_pose_with_diagnostics",
                return_value=(pose, diagnostics),
            ),
            patch(
                "ccip_judge.rejudge._score_pose",
                return_value=(0.8, 0.2, 1, "", ""),
            ),
            patch(
                "ccip_judge.rejudge.detector_sha256",
                return_value="d" * 64,
            ),
        ):
            row = rejudge_image(
                Image.new("RGB", (8, 8)),
                "a" * 64,
                "B",
                refs,
                RejudgeThresholds(),
                "b" * 40,
            )
        self.assertEqual(tuple(row), REJUDGE_FIELDS)
        self.assertEqual(len(row), 14)
        self.assertEqual(row["pass"], "true")

    def test_csv_header_order_is_stable(self):
        row = {field: "" for field in REJUDGE_FIELDS}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            write_rejudge_csv([row], path)
            with path.open(encoding="utf-8", newline="") as fp:
                reader = csv.reader(fp)
                self.assertEqual(tuple(next(reader)), REJUDGE_FIELDS)

    def test_invalid_reference_ratio_is_rejected(self):
        with self.assertRaises(ValueError):
            RejudgeThresholds(min_valid_reference_ratio=0)

    def test_study1_valid_reference_count_is_oks_count(self):
        pose = object()
        refs = MethodReferences(poses=[object(), object()], angle_features=[{}])
        with (
            patch(
                "ccip_judge.rejudge.compute_oks",
                side_effect=[0.8, 0.7],
            ),
            patch(
                "ccip_judge.rejudge.compute_angle_features",
                return_value={"x": 1},
            ),
            patch(
                "ccip_judge.rejudge.valid_angle_features",
                return_value=True,
            ),
            patch(
                "ccip_judge.rejudge.angle_distance",
                return_value=0.2,
            ),
        ):
            result = _score_pose(
                pose,
                refs,
                RejudgeThresholds(pass_rule="oks"),
            )
        self.assertEqual(result[2], 2)

    def test_dirty_judge_requires_explicit_smoke_override(self):
        dirty = f"{'a' * 40}-dirty"
        with self.assertRaises(RuntimeError):
            validate_judge_sha(dirty)
        validate_judge_sha(dirty, allow_dirty=True)
        with self.assertRaises(RuntimeError):
            validate_judge_sha("unknown", allow_dirty=True)

    def test_oks_pass_rule_does_not_fail_on_unavailable_angle(self):
        pose = {
            "keypoints": np.zeros((133, 2), dtype=np.float32),
            "scores": np.ones(133, dtype=np.float32),
            "bbox": np.array([1, 2, 30, 40], dtype=np.float32),
        }
        diagnostics = PoseDiagnostics(
            method="B",
            detector_used="anime",
            confident_keypoints=17,
            bbox=(1, 2, 30, 40),
        )
        refs = MethodReferences(poses=[pose], angle_features=[{"x": 1}])
        with (
            patch(
                "ccip_judge.rejudge.extract_pose_with_diagnostics",
                return_value=(pose, diagnostics),
            ),
            patch(
                "ccip_judge.rejudge._score_pose",
                return_value=(
                    0.8,
                    float("nan"),
                    0,
                    "",
                    "insufficient_angle_references",
                ),
            ),
            patch("ccip_judge.rejudge.detector_sha256", return_value="d" * 64),
        ):
            row = rejudge_image(
                Image.new("RGB", (8, 8)),
                "a" * 64,
                "B",
                refs,
                RejudgeThresholds(pass_rule="oks"),
                "b" * 40,
            )
        self.assertEqual(row["pass"], "true")
        self.assertEqual(row["failure_reason"], "")
        self.assertEqual(row["angle"], "")


if __name__ == "__main__":
    unittest.main()
