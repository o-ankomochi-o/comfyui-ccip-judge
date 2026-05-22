"""Shared helpers: image format conversions and reference loading."""

from __future__ import annotations

import os
import glob
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
        return torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    arrays = []
    for img in pil_list:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arrays.append(np.asarray(img).astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrays, axis=0))


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
        except Exception:
            continue
    return pil_refs


def pil_to_cv2_bgr(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR ndarray."""
    arr = np.asarray(img.convert("RGB"))
    return arr[:, :, ::-1].copy()
