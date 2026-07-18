import unittest

import numpy as np

from ccip_judge.angle_score import valid_angle_features
from ccip_judge.oks_score import valid_oks_reference


class ReferenceValidationTests(unittest.TestCase):
    def test_oks_reference_requires_three_confident_body_keypoints(self):
        pose = {
            "keypoints": np.zeros((133, 2), dtype=np.float32),
            "scores": np.r_[0.9, 0.8, np.zeros(131)],
        }
        self.assertFalse(valid_oks_reference(pose))
        pose["scores"][2] = 0.7
        self.assertTrue(valid_oks_reference(pose))

    def test_angle_reference_requires_two_features(self):
        features = {
            "face_shoulder_ratio": None,
            "shoulder_tilt": None,
            "torso_length_ratio": None,
            "face_compression": 0.1,
        }
        self.assertFalse(valid_angle_features(features))
        features["shoulder_tilt"] = 0.2
        self.assertTrue(valid_angle_features(features))


if __name__ == "__main__":
    unittest.main()
