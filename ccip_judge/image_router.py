"""Image router node.

Splits an IMAGE batch into liked / disliked branches based on a
pass_mask, and (optionally) saves them to disk. Both outputs are
always returned as IMAGE tensors so the workflow can route them
directly into IPAdapterAdvanced without an additional disk round-trip.
"""

from __future__ import annotations

import os
import time
from typing import List

import numpy as np
from PIL import Image

from .common import comfy_image_to_pil_list, pil_list_to_comfy_image


def _save_batch(pil_list, directory: str, prefix: str) -> int:
    if not directory or not pil_list:
        return 0
    os.makedirs(directory, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    n = 0
    for i, img in enumerate(pil_list):
        name = f"{prefix}_{ts}_{i:05d}.png"
        img.save(os.path.join(directory, name))
        n += 1
    return n


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
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("liked", "disliked", "info")
    FUNCTION = "route"
    CATEGORY = "image_judge"
    OUTPUT_NODE = True  # writes to disk; allows running with no downstream consumers

    def route(self, image, pass_mask, save_liked_dir, save_disliked_dir,
              liked_prefix, disliked_prefix, clear_dirs_before_save):
        # INPUT_IS_LIST wraps every required input as a list, including scalar widgets.
        save_liked_dir = save_liked_dir[0] if isinstance(save_liked_dir, list) else save_liked_dir
        save_disliked_dir = save_disliked_dir[0] if isinstance(save_disliked_dir, list) else save_disliked_dir
        liked_prefix = liked_prefix[0] if isinstance(liked_prefix, list) else liked_prefix
        disliked_prefix = disliked_prefix[0] if isinstance(disliked_prefix, list) else disliked_prefix
        clear_dirs_before_save = (
            clear_dirs_before_save[0] if isinstance(clear_dirs_before_save, list)
            else clear_dirs_before_save
        )

        # `image` arrives as a list of IMAGE tensors (one per upstream batch).
        # Flatten them into a single PIL list to mirror per-element pass_mask.
        pil_all: List[Image.Image] = []
        for img_tensor in image:
            pil_all.extend(comfy_image_to_pil_list(img_tensor))

        mask = [bool(v) for v in pass_mask]
        n = min(len(pil_all), len(mask))

        liked_pils = [pil_all[i] for i in range(n) if mask[i]]
        disliked_pils = [pil_all[i] for i in range(n) if not mask[i]]

        if clear_dirs_before_save:
            for d in (save_liked_dir, save_disliked_dir):
                if d and os.path.isdir(d):
                    for fn in os.listdir(d):
                        fp = os.path.join(d, fn)
                        if os.path.isfile(fp) and fn.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".webp")
                        ):
                            try:
                                os.remove(fp)
                            except OSError:
                                pass

        n_liked_saved = _save_batch(liked_pils, save_liked_dir, liked_prefix or "liked")
        n_disliked_saved = _save_batch(disliked_pils, save_disliked_dir, disliked_prefix or "disliked")

        liked_image = pil_list_to_comfy_image(liked_pils)
        disliked_image = pil_list_to_comfy_image(disliked_pils)

        info = (
            f"router n={n} liked={len(liked_pils)} disliked={len(disliked_pils)} "
            f"| saved liked={n_liked_saved} disliked={n_disliked_saved}"
        )
        return (liked_image, disliked_image, info)
