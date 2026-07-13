"""Angle score node.

Builds 4 angle-related features from DWPose keypoints
(face_shoulder_ratio, shoulder_tilt, torso_length_ratio, face_compression)
and returns the RMS distance to the reference features. With multiple
references the per-image distance is averaged.

Fail-explicit contract: extraction failure scores NaN (always fails any
threshold comparison, surfaces as an empty CSV cell + detect_failed flag
via ImageRouter). The fail_score widget is kept only for workflow-JSON
compatibility; its value is ignored.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .common import comfy_image_to_pil_list, load_reference_images
from .dwpose_runner import extract_pose
from .oks_score import _extract_first


def compute_angle_features(pose_data, score_threshold: float | None = None):
    if pose_data is None:
        return None
    from .pose_target import SCORE_THRESHOLD, visible_joints

    if score_threshold is None:
        score_threshold = SCORE_THRESHOLD
    kp, sc = _extract_first(pose_data)
    kp, sc = kp[:17], sc[:17]
    vis = visible_joints(kp, sc, pose_data, score_threshold)
    feats = {}

    if vis[0]:
        face_y = kp[0][1]
    elif vis[1] and vis[2]:
        face_y = (kp[1][1] + kp[2][1]) / 2
    else:
        face_y = None

    left_sh = bool(vis[5])
    right_sh = bool(vis[6])
    if left_sh and right_sh:
        shoulder_y = (kp[5][1] + kp[6][1]) / 2
    elif left_sh:
        shoulder_y = kp[5][1]
    elif right_sh:
        shoulder_y = kp[6][1]
    else:
        shoulder_y = None

    if vis[1] and vis[2]:
        face_width = abs(kp[1][0] - kp[2][0])
    elif vis[3] and vis[4]:
        face_width = abs(kp[3][0] - kp[4][0]) * 0.7
    else:
        face_width = None

    if face_y is not None and shoulder_y is not None and face_width is not None and face_width > 1:
        feats["face_shoulder_ratio"] = float((shoulder_y - face_y) / face_width)
    else:
        feats["face_shoulder_ratio"] = None

    if left_sh and right_sh:
        feats["shoulder_tilt"] = float(np.arctan2(kp[6][1] - kp[5][1], kp[6][0] - kp[5][0]))
    else:
        feats["shoulder_tilt"] = None

    hips_visible = bool(vis[11]) and bool(vis[12])
    if left_sh and right_sh and hips_visible:
        sh_y = (kp[5][1] + kp[6][1]) / 2
        hip_y = (kp[11][1] + kp[12][1]) / 2
        sh_w = abs(kp[5][0] - kp[6][0])
        if sh_w > 1:
            feats["torso_length_ratio"] = float((hip_y - sh_y) / sh_w)
        else:
            feats["torso_length_ratio"] = None
    else:
        feats["torso_length_ratio"] = None

    if vis[0] and vis[1] and vis[2]:
        eye_y = (kp[1][1] + kp[2][1]) / 2
        eye_w = abs(kp[1][0] - kp[2][0])
        if eye_w > 1:
            feats["face_compression"] = float((kp[0][1] - eye_y) / eye_w)
        else:
            feats["face_compression"] = None
    else:
        feats["face_compression"] = None

    return feats


def angle_distance(ref_feats, gen_feats) -> Optional[float]:
    if ref_feats is None or gen_feats is None:
        return None
    keys = ["face_shoulder_ratio", "shoulder_tilt", "torso_length_ratio", "face_compression"]
    diffs = []
    for k in keys:
        rv = ref_feats.get(k)
        gv = gen_feats.get(k)
        if rv is None or gv is None:
            continue
        if k == "shoulder_tilt":
            diffs.append((rv - gv) ** 2 * 0.5)
        else:
            diffs.append((rv - gv) ** 2)
    if len(diffs) < 2:
        return None
    return float(np.sqrt(np.mean(diffs)))


class AngleScore:
    """Camera-angle similarity (distance) averaged across the reference pool."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01}),
                "reference_folder": ("STRING", {"default": "", "multiline": False}),
                # Deprecated: failures now always score NaN. Kept so existing
                # workflow JSONs that set this widget keep loading.
                "fail_score": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                # Authored keypoints (A5): angle features computed straight
                # from the JSON. Insufficient features RAISE -- no silent
                # image fallback inside an experiment.
                "reference_pose_json": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("angle_distance", "pass_mask", "info", "reasons")
    OUTPUT_IS_LIST = (True, True, False, True)
    FUNCTION = "score"
    CATEGORY = "image_judge"

    def score(self, image, threshold, reference_folder, fail_score,
              reference_image=None, reference_pose_json=""):
        if reference_pose_json:
            from .pose_target import load_openpose_json
            rf = compute_angle_features(load_openpose_json(reference_pose_json))
            n_valid = sum(1 for v in (rf or {}).values() if v is not None)
            if rf is None or n_valid < 2:
                raise RuntimeError(
                    "Angle_Score: insufficient_angle_features in "
                    f"{reference_pose_json} (valid={n_valid}, need>=2)")
            ref_feats = [rf]
            ref_source = "openpose_json"
        else:
            ref_pils = load_reference_images(reference_image, reference_folder)
            if not ref_pils:
                raise RuntimeError(
                    "Angle_Score: no reference. Connect reference_image, set "
                    "reference_folder, or set reference_pose_json.")
            ref_feats = []
            for img in ref_pils:
                p = extract_pose(img)
                rf = compute_angle_features(p) if p is not None else None
                if rf is not None:
                    ref_feats.append(rf)
            if not ref_feats:
                raise RuntimeError(
                    "Angle_Score: failed to extract angle features from references.")
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
            gf = compute_angle_features(gp) if gp is not None else None
            per_ref = []
            if gf is not None:
                per_ref = [v for v in (angle_distance(rf, gf) for rf in ref_feats)
                           if v is not None]
            if not per_ref:
                scores.append(float("nan"))
                passes.append(False)
                reasons.append("generated_no_person" if gp is None
                               else "insufficient_angle_features")
                n_detect_fail += 1
                continue
            mean_d = float(np.mean(per_ref))
            scores.append(mean_d)
            passes.append(mean_d < threshold)
            reasons.append("")

        valid = [s for s in scores if not math.isnan(s)]
        info = (
            f"Angle | refs={len(ref_feats)} | reference_source={ref_source} | "
            f"n={len(scores)} | "
            f"mean={float(np.mean(valid)) if valid else float('nan'):.4f} | "
            f"pass={sum(passes)}/{len(passes)} (<{threshold}) | "
            f"detect_fail={n_detect_fail}"
        )
        return (scores, passes, info, reasons)
