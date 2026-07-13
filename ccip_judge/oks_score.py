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


def compute_oks_diag(ref_pose, gen_pose, score_threshold=0.3, min_common=3,
                     keypoint_set: str = ""):
    """(score | None, reason). Failure taxonomy (A6) instead of a bare None
    -- the 07-14 incident took a live debugging session because the only
    signal was `detect_failed=pose`. keypoint_set (A4) restricts evaluation
    to the joints the task's framing can contain (portrait: face..wrists)."""
    if ref_pose is None:
        return None, "reference_invalid"
    if gen_pose is None:
        return None, "generated_no_person"
    ref_kp, ref_sc = _extract_first(ref_pose)
    gen_kp, gen_sc = _extract_first(gen_pose)
    ref_kp, ref_sc = ref_kp[:17], ref_sc[:17]
    gen_kp, gen_sc = gen_kp[:17], gen_sc[:17]

    from .pose_target import KEYPOINT_SETS
    subset = np.zeros(17, dtype=bool)
    subset[list(KEYPOINT_SETS.get(keypoint_set, range(17)))] = True

    ref_vis = (ref_sc > score_threshold) & subset
    gen_vis = (gen_sc > score_threshold) & subset
    common = ref_vis & gen_vis
    if int(common.sum()) < min_common:
        return None, (
            "insufficient_common_keypoints"
            f"(ref={int(ref_vis.sum())},gen={int(gen_vis.sum())},"
            f"common={int(common.sum())},set={keypoint_set or 'all'})")

    # P1 (07-14): the two poses live in DIFFERENT frames -- the authored
    # reference spans its canvas, the generated bbox comes from person
    # detection and may cover an unrelated crop. Normalize each side over
    # the extent of the COMMON visible joints (uniform scale), which is
    # frame- and crop-invariant by construction.
    def _norm_common(kp):
        pts = kp[common].astype(np.float32)
        lo = pts.min(axis=0)
        scale = max(float((pts.max(axis=0) - lo).max()), 1.0)
        return (kp.astype(np.float32) - lo) / scale

    ref_norm = _norm_common(ref_kp)
    gen_norm = _norm_common(gen_kp)
    dists = np.linalg.norm(ref_norm - gen_norm, axis=1)
    e = dists ** 2 / (2 * OKS_SIGMAS ** 2 + 1e-8)
    ks = np.exp(-e)
    return float(ks[common].mean()), ""


def compute_oks(ref_pose, gen_pose, score_threshold=0.3, min_common=3):
    score, _reason = compute_oks_diag(ref_pose, gen_pose, score_threshold,
                                      min_common)
    return score


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
                # Authored keypoints (A1): when set, the reference comes from
                # this OpenPose JSON and no reference image is estimated. A
                # broken JSON RAISES -- fallback to images is a config-level
                # decision, never this node's own initiative (A5).
                "reference_pose_json": ("STRING", {"default": "", "multiline": False}),
                # task joint set (A4): "", "portrait", "full_body"
                "keypoint_set": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("oks", "pass_mask", "info", "reasons")
    OUTPUT_IS_LIST = (True, True, False, True)
    FUNCTION = "score"
    CATEGORY = "image_judge"

    def score(self, image, threshold, reference_folder, fail_score,
              reference_image=None, reference_pose_json="", keypoint_set=""):
        if reference_pose_json:
            from .pose_target import load_openpose_json
            ref_poses = [load_openpose_json(reference_pose_json)]
            ref_source = "openpose_json"
        else:
            ref_pils = load_reference_images(reference_image, reference_folder)
            if not ref_pils:
                raise RuntimeError(
                    "OKS_Score: no reference. Connect reference_image, set "
                    "reference_folder, or set reference_pose_json.")
            ref_poses = [p for p in (extract_pose(img) for img in ref_pils)
                         if p is not None]
            if not ref_poses:
                raise RuntimeError(
                    "OKS_Score: pose extraction failed for all reference images.")
            ref_source = "image"
        gen_pils = comfy_image_to_pil_list(image)
        if not gen_pils:
            return ([], [], "no input images", [])

        scores: List[float] = []
        passes: List[bool] = []
        reasons: List[str] = []
        n_detect_fail = 0
        for gen in gen_pils:
            gp = extract_pose(gen)
            per_ref = []
            last_reason = ""
            for rp in ref_poses:
                v, reason = compute_oks_diag(rp, gp, keypoint_set=keypoint_set)
                if v is not None:
                    per_ref.append(v)
                else:
                    last_reason = reason
            if not per_ref:
                scores.append(float("nan"))
                passes.append(False)
                reasons.append(last_reason or "unknown")
                n_detect_fail += 1
                continue
            mean_oks = float(np.mean(per_ref))
            scores.append(mean_oks)
            passes.append(mean_oks > threshold)
            reasons.append("")

        valid = [s for s in scores if not math.isnan(s)]
        info = (
            f"OKS | refs={len(ref_poses)} | reference_source={ref_source} | "
            f"keypoint_set={keypoint_set or 'all'} | n={len(scores)} | "
            f"mean={float(np.mean(valid)) if valid else float('nan'):.4f} | "
            f"pass={sum(passes)}/{len(passes)} (>{threshold}) | "
            f"detect_fail={n_detect_fail}"
        )
        return (scores, passes, info, reasons)
