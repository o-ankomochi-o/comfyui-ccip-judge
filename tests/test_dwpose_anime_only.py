import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from ccip_judge import dwpose_runner as dw


class AnimeOnlyPoseTests(unittest.TestCase):
    def setUp(self):
        dw.clear_pose_cache()
        self.image = Image.new("RGB", (100, 120), "white")
        self.keypoints = np.zeros((133, 2), dtype=np.float32)
        self.scores = np.ones(133, dtype=np.float32)

    def test_extract_pose_uses_anime_bbox_and_no_yolox_session_exists(self):
        with (
            patch.object(
                dw,
                "_detect_person_anime",
                return_value=np.array([10, 20, 80, 100], dtype=np.float32),
            ) as detect,
            patch.object(dw, "_get_pose_session", return_value=object()),
            patch.object(
                dw,
                "_detect_keypoints",
                return_value=(self.keypoints, self.scores),
            ),
        ):
            pose = dw.extract_pose(self.image)

        detect.assert_called_once_with(self.image)
        self.assertEqual(pose["detector_used"], "anime")
        self.assertEqual(pose["image_shape"], (120, 100))
        self.assertFalse(hasattr(dw, "_get_det_session"))
        self.assertFalse(hasattr(dw, "_detect_person_official"))
        self.assertFalse(hasattr(dw, "_detect_person_legacy_raw"))

    def test_no_anime_detection_is_an_explicit_failure(self):
        with patch.object(dw, "_detect_person_anime", return_value=None):
            self.assertIsNone(dw.extract_pose(self.image))

    def test_pose_cache_avoids_duplicate_detection(self):
        pose = {
            "keypoints": self.keypoints,
            "scores": self.scores,
            "bbox": np.array([1, 2, 90, 110], dtype=np.float32),
            "image_shape": (120, 100),
            "detector_used": "anime",
        }
        with patch.object(dw, "extract_pose", return_value=pose) as extract:
            first = dw.extract_pose_cached(self.image)
            second = dw.extract_pose_cached(self.image)
        self.assertIs(first, second)
        extract.assert_called_once_with(self.image)


if __name__ == "__main__":
    unittest.main()
