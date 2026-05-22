"""Three-stage filter node.

Takes three FLOAT lists (CCIP / OKS / Angle) and ANDs the per-image
pass decisions to produce one BOOLEAN list usable by ImageRouter.
The node also accepts the three thresholds directly so it can be
configured without trusting the upstream Score nodes' own widgets.
"""

from __future__ import annotations

from typing import List

import numpy as np


class ThreeStageFilter:
    """Combine CCIP / OKS / Angle scores into a single pass mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ccip_distance": ("FLOAT", {"forceInput": True}),
                "oks": ("FLOAT", {"forceInput": True}),
                "angle_distance": ("FLOAT", {"forceInput": True}),
                "ccip_threshold": ("FLOAT", {"default": 0.213, "min": 0.0, "max": 2.0, "step": 0.001}),
                "oks_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "angle_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("pass_mask", "info")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "filter"
    CATEGORY = "image_judge"

    def filter(self, ccip_distance, oks, angle_distance,
               ccip_threshold, oks_threshold, angle_threshold):
        # When INPUT_IS_LIST is True, scalar widgets arrive as 1-element lists.
        ccip_t = float(ccip_threshold[0]) if isinstance(ccip_threshold, list) else float(ccip_threshold)
        oks_t = float(oks_threshold[0]) if isinstance(oks_threshold, list) else float(oks_threshold)
        ang_t = float(angle_threshold[0]) if isinstance(angle_threshold, list) else float(angle_threshold)

        n = min(len(ccip_distance), len(oks), len(angle_distance))
        if n == 0:
            return ([], "no scores")

        passes: List[bool] = []
        n_ccip_ok = 0
        n_oks_ok = 0
        n_ang_ok = 0
        for i in range(n):
            c = float(ccip_distance[i])
            k = float(oks[i])
            a = float(angle_distance[i])
            c_ok = c < ccip_t
            k_ok = k > oks_t
            a_ok = a < ang_t
            n_ccip_ok += int(c_ok)
            n_oks_ok += int(k_ok)
            n_ang_ok += int(a_ok)
            passes.append(c_ok and k_ok and a_ok)

        passed = int(np.sum(passes))
        info = (
            f"filter n={n} pass={passed}/{n} "
            f"(CCIP<{ccip_t}:{n_ccip_ok}, OKS>{oks_t}:{n_oks_ok}, Angle<{ang_t}:{n_ang_ok})"
        )
        return (passes, info)
