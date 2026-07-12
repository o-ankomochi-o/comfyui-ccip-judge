"""OKS score node.

Computes Object Keypoint Similarity between each generated image and
the reference pool. With multiple references the per-image score is
averaged (as requested by the user).

Fail-explicit contract: when pose extraction fails for a generated
image the score is NaN (not a configurable stand-in value), so the
failure survives the pipeline — NaN compares False against any
threshold (always disliked) and ImageRouter writes it as an empty CSV
cell plus a detect_failed flag. The fail_score widget is kept only so
existing workflow JSONs keep loading; its value is ignored.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .common import comfy_image_to_pil_list, load_reference_images
from .dwpose_runner import extract_pose


OKS_SIGMAS = np.array([
    .026, .025, .025, .035, .035,
    .079, .079, .072, .072, .062, .062,
    .107, .107, .087, .087, .089, .089,
])


def _extract_first(pose_data):
    kp = np.asarray(pose_data["keypoints"])
    sc = np.asarray(pose_data["scores"])
    if kp.ndim == 3:
        kp = kp[0]
    if sc.ndim == 2:
        sc = sc[0]
    return kp, sc


def _get_bbox(pose_data):
    bbox = np.asarray(pose_data["bbox"])
    if bbox.ndim == 2:
        bbox = bbox[0]
    return bbox


def _normalize(kp, bbox):
    x1, y1, x2, y2 = bbox
    w = max(float(x2 - x1), 1.0)
    h = max(float(y2 - y1), 1.0)
    out = kp.astype(np.float32).copy()
    out[:, 0] = (kp[:, 0] - x1) / w
    out[:, 1] = (kp[:, 1] - y1) / h
    return out


def compute_oks(ref_pose, gen_pose, score_threshold=0.3, min_common=3):
    if ref_pose is None or gen_pose is None:
        return None
    ref_kp, ref_sc = _extract_first(ref_pose)
    gen_kp, gen_sc = _extract_first(gen_pose)
    ref_kp, ref_sc = ref_kp[:17], ref_sc[:17]
    gen_kp, gen_sc = gen_kp[:17], gen_sc[:17]

    common = (ref_sc > score_threshold) & (gen_sc > score_threshold)
    if int(common.sum()) < min_common:
        return None

    ref_norm = _normalize(ref_kp, _get_bbox(ref_pose))
    gen_norm = _normalize(gen_kp, _get_bbox(gen_pose))
    dists = np.linalg.norm(ref_norm - gen_norm, axis=1)
    e = dists ** 2 / (2 * OKS_SIGMAS ** 2 + 1e-8)
    ks = np.exp(-e)
    return float(ks[common].mean())


class OKSScore:
    """Per-image OKS averaged across the reference pool."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reference_folder": ("STRING", {"default": "", "multiline": False}),
                # Deprecated: failures now always score NaN. Kept so existing
                # workflow JSONs that set this widget keep loading.
                "fail_score": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("oks", "pass_mask", "info")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "score"
    CATEGORY = "image_judge"

    def score(self, image, threshold, reference_folder, fail_score, reference_image=None):
        ref_pils = load_reference_images(reference_image, reference_folder)
        if not ref_pils:
            raise RuntimeError(
                "OKS_Score: no reference images. "
                "Connect reference_image or set reference_folder."
            )
        gen_pils = comfy_image_to_pil_list(image)
        if not gen_pils:
            return ([], [], "no input images")

        ref_poses = [extract_pose(img) for img in ref_pils]
        ref_poses = [p for p in ref_poses if p is not None]
        if not ref_poses:
            raise RuntimeError("OKS_Score: pose extraction failed for all reference images.")

        scores: List[float] = []
        passes: List[bool] = []
        n_detect_fail = 0
        for gen in gen_pils:
            gp = extract_pose(gen)
            per_ref = []
            if gp is not None:
                per_ref = [v for v in (compute_oks(rp, gp) for rp in ref_poses)
                           if v is not None]
            if not per_ref:
                scores.append(float("nan"))
                passes.append(False)
                n_detect_fail += 1
                continue
            mean_oks = float(np.mean(per_ref))
            scores.append(mean_oks)
            passes.append(mean_oks > threshold)

        valid = [s for s in scores if not math.isnan(s)]
        info = (
            f"OKS | refs={len(ref_poses)} | n={len(scores)} | "
            f"mean={float(np.mean(valid)) if valid else float('nan'):.4f} | "
            f"pass={sum(passes)}/{len(passes)} (>{threshold}) | "
            f"detect_fail={n_detect_fail}"
        )
        return (scores, passes, info)
