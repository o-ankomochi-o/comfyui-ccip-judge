import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ccip_judge import dwpose_runner as dw


def _pose(detector, count):
    return {
        "keypoints": np.zeros((133, 2), dtype=np.float32),
        "scores": np.r_[np.ones(count), np.zeros(133 - count)].astype(np.float32),
        "bbox": np.array([1, 2, 30, 40], dtype=np.float32),
        "image_shape": (64, 64),
        "detector_used": detector,
    }


class YoloPostprocessTests(unittest.TestCase):
    def test_demo_postprocess_decodes_grid_and_stride(self):
        raw = np.zeros((1, 8400, 85), dtype=np.float32)
        decoded = dw._demo_postprocess(raw)
        np.testing.assert_allclose(decoded[0, 0, :4], [0, 0, 8, 8])
        np.testing.assert_allclose(decoded[0, 1, :4], [8, 0, 8, 8])

    def test_demo_postprocess_rejects_wrong_anchor_count(self):
        with self.assertRaisesRegex(RuntimeError, "anchor count"):
            dw._demo_postprocess(np.zeros((1, 10, 85), dtype=np.float32))

    def test_nms_suppresses_overlapping_lower_score_box(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [30, 30, 40, 40]])
        scores = np.array([0.9, 0.8, 0.7])
        self.assertEqual(dw._nms(boxes, scores, 0.45), [0, 2])


class PoseMethodRoutingTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (64, 64))
        self.image_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        self.boxes = np.array([[1, 2, 30, 40]], dtype=np.float32)

    def _pose_with_detector(self, _image, bbox, detector):
        if bbox is None:
            return None
        counts = {
            "legacy_yolox_raw": 2,
            "anime": 7,
            "official_yolox": 9,
        }
        return _pose(detector, counts[detector])

    def test_method_a_retries_anime_after_legacy_collapse(self):
        with (
            patch.object(dw, "pil_to_cv2_bgr", return_value=self.image_bgr),
            patch.object(dw, "_get_det_session", return_value=object()),
            patch.object(dw, "_detect_person_legacy_raw", return_value=(self.boxes, [0.9])),
            patch.object(dw, "_detect_person_anime", return_value=self.boxes[0]),
            patch.object(dw, "_pose_with_detector", side_effect=self._pose_with_detector),
        ):
            pose, diag = dw.extract_pose_with_diagnostics(self.image, "A")
        self.assertEqual(pose["detector_used"], "anime")
        self.assertEqual(diag.confident_keypoints, 7)

    def test_method_b_never_loads_yolox(self):
        with (
            patch.object(dw, "pil_to_cv2_bgr", return_value=self.image_bgr),
            patch.object(dw, "_get_det_session") as get_yolox,
            patch.object(dw, "_detect_person_anime", return_value=self.boxes[0]),
            patch.object(dw, "_pose_with_detector", side_effect=self._pose_with_detector),
        ):
            pose, diag = dw.extract_pose_with_diagnostics(self.image, "B")
        get_yolox.assert_not_called()
        self.assertEqual(pose["detector_used"], "anime")
        self.assertEqual(diag.method, "B")

    def test_method_bprime_falls_back_to_official_yolox(self):
        def routed(_image, bbox, detector):
            if bbox is None:
                return None
            return _pose(detector, 2 if detector == "anime" else 9)

        with (
            patch.object(dw, "pil_to_cv2_bgr", return_value=self.image_bgr),
            patch.object(dw, "_get_det_session", return_value=object()),
            patch.object(dw, "_detect_person_anime", return_value=self.boxes[0]),
            patch.object(dw, "_detect_person_official", return_value=(self.boxes, [0.9])),
            patch.object(dw, "_pose_with_detector", side_effect=routed),
        ):
            pose, diag = dw.extract_pose_with_diagnostics(self.image, "Bprime")
        self.assertEqual(pose["detector_used"], "official_yolox")
        self.assertEqual(diag.confident_keypoints, 9)

    def test_method_b_surfaces_anime_detector_runtime_errors(self):
        with (
            patch.object(dw, "pil_to_cv2_bgr", return_value=self.image_bgr),
            patch.object(
                dw,
                "_detect_person_anime",
                side_effect=RuntimeError("detector CUDA failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
                dw.extract_pose_with_diagnostics(self.image, "B")


if __name__ == "__main__":
    unittest.main()
