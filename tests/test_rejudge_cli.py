import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "extras" / "rejudge_pose_methods.py"
)
SPEC = importlib.util.spec_from_file_location("rejudge_pose_methods", SCRIPT_PATH)
REJUDGE_CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REJUDGE_CLI)


class RejudgeCliTests(unittest.TestCase):
    def _args(self, pose_path):
        return SimpleNamespace(
            reference_pose_json=str(pose_path),
            reference_folder=None,
            keypoint_set="full_body",
            methods=["A", "B", "Bprime"],
            oks_threshold=0.5,
            angle_threshold=0.5,
            min_valid_reference_ratio=0.5,
            pass_rule="oks",
        )

    def test_provenance_freezes_authored_pose_and_run_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pose = root / "action-reach.json"
            pose.write_text(
                json.dumps(
                    {
                        "canvas_width": 832,
                        "canvas_height": 1152,
                        "people": [{"pose_keypoints_2d": [1, 2, 1] * 18}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "scores.csv"
            args = self._args(pose)
            path = REJUDGE_CLI._write_or_validate_provenance(
                args, "a" * 40, output
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["reference"]["kind"], "openpose_json")
            self.assertEqual(len(saved["reference"]["sha256"]), 64)
            self.assertEqual(
                saved["reference"]["keypoint_set"], "full_body"
            )
            self.assertEqual(saved["methods"], ["A", "B", "Bprime"])

            args.oks_threshold = 0.6
            with self.assertRaises(RuntimeError):
                REJUDGE_CLI._write_or_validate_provenance(
                    args, "a" * 40, output
                )


if __name__ == "__main__":
    unittest.main()
