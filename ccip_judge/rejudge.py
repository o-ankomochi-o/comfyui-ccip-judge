"""Deterministic A/B/Bprime pose re-judging support.

This module is intentionally independent from evocomfy. Its only integration
surface is the stable CSV schema in ``REJUDGE_FIELDS``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .angle_score import (
    angle_distance,
    compute_angle_features,
    valid_angle_features,
)
from .common import file_sha256
from .dwpose_runner import (
    ANIME_DETECT_MODEL_NAME,
    ANIME_DETECT_REPO_ID,
    ANIME_DETECT_REVISION,
    POSE_METHODS,
    POSE_MODEL_SHA256,
    YOLOX_MODEL_SHA256,
    extract_pose_with_diagnostics,
    normalize_pose_method,
)
from .oks_score import (
    compute_oks,
    valid_oks_reference,
)


REJUDGE_FIELDS = (
    "image_sha256",
    "method",
    "oks",
    "angle",
    "pass",
    "failure_reason",
    "detector_used",
    "confident_keypoints",
    "valid_reference_count",
    "bbox",
    "latency_ms",
    "judge_sha",
    "detector_model_sha256",
    "pose_model_sha256",
)


@dataclass(frozen=True)
class RejudgeThresholds:
    oks: float = 0.5
    angle: float = 0.5
    min_valid_reference_ratio: float = 0.5
    pass_rule: str = "oks"

    def __post_init__(self):
        if not 0 < self.min_valid_reference_ratio <= 1:
            raise ValueError("min_valid_reference_ratio must be in (0, 1]")
        if self.pass_rule not in {"oks", "oks_and_angle"}:
            raise ValueError("pass_rule must be 'oks' or 'oks_and_angle'")


@dataclass
class MethodReferences:
    poses: list
    angle_features: list


def _minimum_valid(total: int, ratio: float) -> int:
    return max(1, math.ceil(total * ratio))


def prepare_method_references(
    reference_images: Sequence[Image.Image], method: str
) -> MethodReferences:
    normalized = normalize_pose_method(method)
    poses = []
    features = []
    for image in reference_images:
        pose, _ = extract_pose_with_diagnostics(image, normalized)
        if valid_oks_reference(pose):
            poses.append(pose)
        angle = compute_angle_features(pose) if pose is not None else None
        if valid_angle_features(angle):
            features.append(angle)
    if not poses:
        raise RuntimeError(
            f"method {normalized}: no reference has 3 confident body keypoints"
        )
    if not features:
        raise RuntimeError(
            f"method {normalized}: no reference has 2 usable angle features"
        )
    return MethodReferences(poses=poses, angle_features=features)


def _format_score(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.8f}"


def _score_pose(pose, references: MethodReferences, thresholds: RejudgeThresholds):
    if pose is None:
        return float("nan"), float("nan"), 0, "no_person", "no_person"

    oks_values = [
        score
        for score in (compute_oks(ref, pose) for ref in references.poses)
        if score is not None
    ]
    angle_features = compute_angle_features(pose)
    angle_values = []
    if valid_angle_features(angle_features):
        angle_values = [
            score
            for score in (
                angle_distance(ref, angle_features)
                for ref in references.angle_features
            )
            if score is not None
        ]

    ratio = thresholds.min_valid_reference_ratio
    min_oks = _minimum_valid(len(references.poses), ratio)
    min_angle = _minimum_valid(len(references.angle_features), ratio)
    valid_count = (
        len(oks_values)
        if thresholds.pass_rule == "oks"
        else min(len(oks_values), len(angle_values))
    )
    oks_failure = (
        "insufficient_oks_references" if len(oks_values) < min_oks else ""
    )
    angle_failure = (
        "insufficient_angle_references" if len(angle_values) < min_angle else ""
    )
    return (
        float("nan") if oks_failure else float(np.mean(oks_values)),
        float("nan") if angle_failure else float(np.mean(angle_values)),
        valid_count,
        oks_failure,
        angle_failure,
    )


@lru_cache(maxsize=1)
def anime_detector_model_sha256() -> str:
    """Hash the frozen artifact and reject a changed model loaded by imgutils.

    imgutils currently resolves its model from ``main`` internally and does not
    expose a revision parameter. Comparing the current artifact with the frozen
    revision prevents a silent weight change from entering a research run.
    """
    from huggingface_hub import hf_hub_download

    filename = f"{ANIME_DETECT_MODEL_NAME}/model.onnx"
    frozen_path = hf_hub_download(
        repo_id=ANIME_DETECT_REPO_ID,
        filename=filename,
        revision=ANIME_DETECT_REVISION,
    )
    current_path = hf_hub_download(
        repo_id=ANIME_DETECT_REPO_ID,
        filename=filename,
        revision="main",
    )
    frozen_sha = file_sha256(frozen_path)
    current_sha = file_sha256(current_path)
    if current_sha != frozen_sha:
        raise RuntimeError(
            "anime detector artifact on main differs from the frozen revision "
            f"{ANIME_DETECT_REVISION}; refusing to mix model weights"
        )
    return frozen_sha


def detector_sha256(detector_used: str) -> str:
    if detector_used == "anime":
        return anime_detector_model_sha256()
    if detector_used in {"legacy_yolox_raw", "official_yolox"}:
        return YOLOX_MODEL_SHA256
    return ""


def current_judge_sha(repo_root: str | os.PathLike | None = None) -> str:
    override = os.environ.get("CCIP_JUDGE_SHA")
    if override:
        return override
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_judge_sha(judge_sha: str, allow_dirty: bool = False) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", judge_sha):
        return
    if allow_dirty and re.fullmatch(r"[0-9a-f]{40}-dirty", judge_sha):
        return
    raise RuntimeError(
        "re-judging requires a committed 40-character judge SHA; "
        "commit the implementation first, or use --allow-dirty only for "
        "development smoke tests"
    )


def rejudge_image(
    image: Image.Image,
    image_sha256: str,
    method: str,
    references: MethodReferences,
    thresholds: RejudgeThresholds,
    judge_sha: str,
) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
        raise ValueError("image_sha256 must be 64 lowercase hexadecimal characters")
    normalized = normalize_pose_method(method)
    start = time.perf_counter()
    pose, diagnostics = extract_pose_with_diagnostics(image, normalized)
    oks, angle, valid_count, oks_failure, angle_failure = _score_pose(
        pose, references, thresholds
    )
    latency_ms = (time.perf_counter() - start) * 1000
    failure = oks_failure or diagnostics.failure_reason
    passed = not failure and not math.isnan(oks) and oks > thresholds.oks
    if thresholds.pass_rule == "oks_and_angle":
        failure = failure or angle_failure
        passed = (
            passed
            and not failure
            and not math.isnan(angle)
            and angle < thresholds.angle
        )
    return {
        "image_sha256": image_sha256,
        "method": normalized,
        "oks": _format_score(oks),
        "angle": _format_score(angle),
        "pass": str(bool(passed)).lower(),
        "failure_reason": failure,
        "detector_used": diagnostics.detector_used,
        "confident_keypoints": diagnostics.confident_keypoints,
        "valid_reference_count": valid_count,
        "bbox": json.dumps(diagnostics.as_dict()["bbox"], separators=(",", ":")),
        "latency_ms": f"{latency_ms:.3f}",
        "judge_sha": judge_sha,
        "detector_model_sha256": detector_sha256(diagnostics.detector_used),
        "pose_model_sha256": POSE_MODEL_SHA256,
    }


def write_rejudge_csv(rows: Iterable[dict], output_path: str | os.PathLike):
    with open(output_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=REJUDGE_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
