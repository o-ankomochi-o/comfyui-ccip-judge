"""Image router node.

Splits an IMAGE batch into liked / disliked branches based on a
pass_mask, and (optionally) saves them to disk. Both outputs are
always returned as IMAGE tensors so the workflow can route them
directly into IPAdapterAdvanced without an additional disk round-trip.

Saved filenames embed the original batch index so it is possible to
trace which generated image became liked or disliked. When optional
score inputs (ccip_distance, oks, angle_distance) are connected the
node also writes a CSV report next to the saved images.
"""

from __future__ import annotations

import csv
import math
import os
import re
from datetime import datetime
from typing import List, Optional

from PIL import Image

from .common import comfy_image_to_pil_list, pil_list_to_comfy_image


def _sanitize_prefix(prefix, default: str) -> str:
    """Force prefixes to be plain filename stems: no path separators,
    reserved characters, or traversal sequences, so a widget value can
    never write outside the configured directory."""
    s = str(prefix or "").strip()
    s = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", s)
    s = s.replace("..", "_")
    s = s.strip(" .")
    return s or default


def _save_batch(pil_list, indices, directory: str, prefix: str, ts: str) -> int:
    """Save with original batch index in the filename: {prefix}_idx{NN}_{ts}.png.
    Never overwrites: on a name collision a numeric suffix is appended."""
    if not directory or not pil_list:
        return 0
    os.makedirs(directory, exist_ok=True)
    n = 0
    for orig_idx, img in zip(indices, pil_list):
        base = f"{prefix}_idx{orig_idx:05d}_{ts}"
        path = os.path.join(directory, base + ".png")
        k = 1
        while os.path.exists(path):
            path = os.path.join(directory, f"{base}_{k}.png")
            k += 1
        img.save(path)
        n += 1
    return n


def _clear_previous_outputs(targets) -> tuple:
    """Delete only files this node itself wrote earlier: saved images named
    '{prefix}_idx*.png' and 'scores_*.csv' reports. Reference images or any
    other user files sharing the directory are left untouched.

    targets: iterable of (directory, matcher) where matcher(lowercase_name)
    decides whether a file belongs to this node. Returns (n_deleted, failures)
    with failures as human-readable strings — never silently swallowed."""
    n_deleted = 0
    failures: List[str] = []
    for directory, matcher in targets:
        if not directory or not os.path.isdir(directory):
            continue
        for fn in os.listdir(directory):
            fp = os.path.join(directory, fn)
            if not os.path.isfile(fp) or not matcher(fn.lower()):
                continue
            try:
                os.remove(fp)
                n_deleted += 1
            except OSError as e:
                failures.append(f"{fp}: {e}")
    return n_deleted, failures


def _image_matcher(prefix: str):
    head = f"{prefix.lower()}_idx"
    return lambda fn: fn.startswith(head) and fn.endswith(".png")


def _csv_matcher(fn: str) -> bool:
    return fn.startswith("scores_") and fn.endswith(".csv")


def _pop_scalar(value, default=None):
    """INPUT_IS_LIST wraps scalar widgets as 1-element lists; unwrap them."""
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _maybe_floats(values) -> Optional[List[float]]:
    """Accept None / [] / list[float] and return a clean list or None."""
    if values is None:
        return None
    if isinstance(values, list) and len(values) == 0:
        return None
    return [float(v) for v in values]


def _image_output(pil_list):
    if pil_list:
        return pil_list_to_comfy_image(pil_list)
    try:
        from comfy_execution.graph import ExecutionBlocker
    except ImportError:
        return pil_list_to_comfy_image([])
    return ExecutionBlocker(None)


class ImageRouter:
    """Route images into liked/disliked branches and optionally save to disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pass_mask": ("BOOLEAN", {"forceInput": True}),
                "save_liked_dir": ("STRING", {"default": "", "multiline": False}),
                "save_disliked_dir": ("STRING", {"default": "", "multiline": False}),
                "liked_prefix": ("STRING", {"default": "liked", "multiline": False}),
                "disliked_prefix": ("STRING", {"default": "disliked", "multiline": False}),
                "clear_dirs_before_save": ("BOOLEAN", {"default": False}),
                "csv_dir": ("STRING", {"default": "", "multiline": False}),
                "ccip_threshold": ("FLOAT", {"default": 0.213, "min": 0.0, "max": 2.0, "step": 0.001}),
                "oks_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "angle_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "ccip_distance": ("FLOAT", {"forceInput": True}),
                "oks": ("FLOAT", {"forceInput": True}),
                "angle_distance": ("FLOAT", {"forceInput": True}),
                # A6/P2: per-image failure taxonomy from the score nodes
                "pose_reasons": ("STRING", {"forceInput": True}),
                "angle_reasons": ("STRING", {"forceInput": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("liked", "disliked", "info")
    FUNCTION = "route"
    CATEGORY = "image_judge"
    OUTPUT_NODE = True  # writes to disk; allows running with no downstream consumers

    def route(self, image, pass_mask, save_liked_dir, save_disliked_dir,
              liked_prefix, disliked_prefix, clear_dirs_before_save, csv_dir,
              ccip_threshold, oks_threshold, angle_threshold,
              ccip_distance=None, oks=None, angle_distance=None,
              pose_reasons=None, angle_reasons=None):
        save_liked_dir = _pop_scalar(save_liked_dir, "")
        save_disliked_dir = _pop_scalar(save_disliked_dir, "")
        liked_prefix = _sanitize_prefix(_pop_scalar(liked_prefix, "liked"), "liked")
        disliked_prefix = _sanitize_prefix(_pop_scalar(disliked_prefix, "disliked"), "disliked")
        clear_dirs_before_save = bool(_pop_scalar(clear_dirs_before_save, False))
        csv_dir = _pop_scalar(csv_dir, "")
        ccip_t = float(_pop_scalar(ccip_threshold, 0.213))
        oks_t = float(_pop_scalar(oks_threshold, 0.5))
        ang_t = float(_pop_scalar(angle_threshold, 0.5))

        # `image` arrives as a list of IMAGE tensors (one per upstream batch).
        pil_all: List[Image.Image] = []
        for img_tensor in image:
            pil_all.extend(comfy_image_to_pil_list(img_tensor))

        mask = [bool(v) for v in pass_mask]
        if len(pil_all) != len(mask):
            raise RuntimeError(
                f"ImageRouter: image count ({len(pil_all)}) != pass_mask "
                f"length ({len(mask)}); check upstream wiring / stale caches."
            )
        n = len(pil_all)

        liked_idx = [i for i in range(n) if mask[i]]
        disliked_idx = [i for i in range(n) if not mask[i]]
        liked_pils = [pil_all[i] for i in liked_idx]
        disliked_pils = [pil_all[i] for i in disliked_idx]

        n_cleared = 0
        clear_failures: List[str] = []
        if clear_dirs_before_save:
            n_cleared, clear_failures = _clear_previous_outputs((
                (save_liked_dir, _image_matcher(liked_prefix)),
                (save_disliked_dir, _image_matcher(disliked_prefix)),
                (csv_dir, _csv_matcher),
            ))
            for msg in clear_failures:
                print(f"[ImageRouter] WARNING: could not delete {msg}")

        # Microsecond timestamp: two runs inside the same second must not
        # overwrite each other's files.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        n_liked_saved = _save_batch(liked_pils, liked_idx, save_liked_dir, liked_prefix, ts)
        n_disliked_saved = _save_batch(disliked_pils, disliked_idx, save_disliked_dir, disliked_prefix, ts)

        # Optional CSV report when score lists are connected.
        ccip_list = _maybe_floats(ccip_distance)
        oks_list = _maybe_floats(oks)
        ang_list = _maybe_floats(angle_distance)
        for name, lst in (("ccip_distance", ccip_list), ("oks", oks_list),
                          ("angle_distance", ang_list)):
            if lst is not None and len(lst) != n:
                raise RuntimeError(
                    f"ImageRouter: {name} length ({len(lst)}) != image "
                    f"count ({n}); check upstream wiring / stale caches."
                )
        csv_path = None
        if csv_dir and (ccip_list or oks_list or ang_list):
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, f"scores_{ts}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                w.writerow([
                    "index", "ccip", "oks", "angle",
                    "ccip_ok", "oks_ok", "angle_ok", "verdict", "detect_failed",
                    "pose_debug", "angle_debug",
                ])
                for i in range(n):
                    c = ccip_list[i] if ccip_list and i < len(ccip_list) else ""
                    k = oks_list[i] if oks_list and i < len(oks_list) else ""
                    a = ang_list[i] if ang_list and i < len(ang_list) else ""
                    # NaN is the fail-explicit marker from the score nodes:
                    # written as an empty cell + a detect_failed tag so the
                    # failure is distinguishable from a genuinely low score.
                    c_fail = c != "" and math.isnan(c)
                    k_fail = k != "" and math.isnan(k)
                    a_fail = a != "" and math.isnan(a)
                    fails = []
                    if k_fail or a_fail:
                        fails.append("pose")
                    if c_fail:
                        fails.append("ccip")
                    c_ok = "" if c == "" or c_fail else c < ccip_t
                    k_ok = "" if k == "" or k_fail else k > oks_t
                    a_ok = "" if a == "" or a_fail else a < ang_t
                    verdict = "LIKED" if mask[i] else "disliked"
                    pose_debug = ""
                    if pose_reasons and i < len(pose_reasons):
                        pose_debug = str(pose_reasons[i] or "")
                    angle_debug = ""
                    if angle_reasons and i < len(angle_reasons):
                        angle_debug = str(angle_reasons[i] or "")
                    w.writerow([
                        i,
                        f"{c:.4f}" if c != "" and not c_fail else "",
                        f"{k:.4f}" if k != "" and not k_fail else "",
                        f"{a:.4f}" if a != "" and not a_fail else "",
                        c_ok, k_ok, a_ok, verdict, "+".join(fails),
                        pose_debug, angle_debug,
                    ])

        liked_image = _image_output(liked_pils)
        disliked_image = _image_output(disliked_pils)

        info_parts = [
            f"router n={n} liked={len(liked_pils)} disliked={len(disliked_pils)}",
            f"saved liked={n_liked_saved} disliked={n_disliked_saved}",
        ]
        if clear_dirs_before_save:
            info_parts.append(f"cleared={n_cleared}")
            if clear_failures:
                info_parts.append(f"clear_failed={len(clear_failures)}")
        if csv_path:
            info_parts.append(f"csv={csv_path}")
        info = " | ".join(info_parts)
        return (liked_image, disliked_image, info)
