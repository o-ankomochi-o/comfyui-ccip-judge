import tempfile
import unittest
import sys
import types
from unittest.mock import patch
from pathlib import Path

import numpy as np

from ccip_judge.image_router import (
    ImageRouter,
    _clear_previous_outputs,
    _image_output,
    _image_matcher,
    _sanitize_prefix,
)


class RouterSafetyTests(unittest.TestCase):
    def test_prefix_cannot_escape_directory(self):
        sanitized = _sanitize_prefix("../../bad/name", "liked")
        self.assertNotIn("..", sanitized)
        self.assertNotIn("/", sanitized)
        self.assertNotIn("\\", sanitized)

    def test_clear_only_removes_owned_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owned = root / "liked_idx00001_old.png"
            unrelated = root / "reference.png"
            owned.write_bytes(b"owned")
            unrelated.write_bytes(b"keep")
            deleted, failures = _clear_previous_outputs(
                ((tmp, _image_matcher("liked")),)
            )
            self.assertEqual(deleted, 1)
            self.assertEqual(failures, [])
            self.assertFalse(owned.exists())
            self.assertTrue(unrelated.exists())

    def test_router_rejects_length_mismatch(self):
        image = np.zeros((2, 8, 8, 3), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "image count"):
            ImageRouter().route(
                [image],
                [True],
                [""],
                [""],
                ["liked"],
                ["disliked"],
                [False],
                [""],
                [0.213],
                [0.5],
                [0.5],
            )

    def test_empty_branch_uses_comfy_execution_blocker(self):
        class FakeBlocker:
            def __init__(self, message):
                self.message = message

        fake_graph = types.ModuleType("comfy_execution.graph")
        fake_graph.ExecutionBlocker = FakeBlocker
        fake_package = types.ModuleType("comfy_execution")
        with patch.dict(
            sys.modules,
            {
                "comfy_execution": fake_package,
                "comfy_execution.graph": fake_graph,
            },
        ):
            output = _image_output([])
        self.assertIsInstance(output, FakeBlocker)
        self.assertIsNone(output.message)


if __name__ == "__main__":
    unittest.main()
