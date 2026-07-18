"""Shared helpers: image format conversions and reference loading."""

from __future__ import annotations

import os
import glob
import hashlib
from typing import List, Optional

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP")


def comfy_image_to_pil_list(image) -> List[Image.Image]:
    """Convert a ComfyUI IMAGE tensor (B, H, W, C) in [0, 1] to a list of PIL images."""
    if image is None:
        return []
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim == 3:
        arr = arr[None]
    pil_list = []
    for frame in arr:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        pil_list.append(Image.fromarray(frame))
    return pil_list


def pil_list_to_comfy_image(pil_list: List[Image.Image]):
    """Convert a list of PIL images back into a ComfyUI IMAGE tensor."""
    if torch is None:
        raise RuntimeError("torch is required to build IMAGE tensors")
    if not pil_list:
        # An empty branch must never masquerade as a real black reference
        # image. ImageRouter converts this condition to an ExecutionBlocker
        # when running inside ComfyUI; the zero batch is the safe fallback for
        # tests and non-Comfy callers.
        return torch.empty((0, 8, 8, 3), dtype=torch.float32)
    arrays = []
    for img in pil_list:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arrays.append(np.asarray(img).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrays, axis=0))


def pil_image_sha256(img: Image.Image) -> str:
    """Stable hash of canonical RGB pixels for in-memory ComfyUI images."""
    rgb = img.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB\0".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_images(
    reference_image=None,
    reference_folder: str = "",
) -> List[Image.Image]:
    """Resolve the reference pool.

    Priority: reference_image (IMAGE) takes precedence when provided.
    Falls back to reading every supported image in reference_folder.
    Returns [] when neither source yields images.
    """
    if reference_image is not None:
        pil_refs = comfy_image_to_pil_list(reference_image)
        if pil_refs:
            return pil_refs

    folder = (reference_folder or "").strip()
    if not folder:
        return []
    if not os.path.isdir(folder):
        return []

    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    paths = sorted(set(paths))
    pil_refs = []
    for path in paths:
        try:
            pil_refs.append(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"[CCIPJudge] WARNING: skipped reference image {path}: {e}")
            continue
    return pil_refs


def pil_to_cv2_bgr(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR ndarray."""
    arr = np.asarray(img.convert("RGB"))
    return arr[:, :, ::-1].copy()


def detect_character_present(img: Image.Image) -> Optional[bool]:
    """Anime-character presence check for the fail-explicit contract.

    Person detector first; face detector as backup because bust/face-only
    shots can slip past the person detector. Returns None (= check skipped,
    score normally) when imgutils.detect is unavailable or the detectors
    themselves error — only a clean "both detectors ran and found nothing"
    returns False, so images are never mass-failed by an environment issue.
    """
    try:
        from imgutils.detect import detect_person, detect_faces
    except ImportError:
        return None
    try:
        if detect_person(img):
            return True
        return bool(detect_faces(img))
    except Exception:
        return None
