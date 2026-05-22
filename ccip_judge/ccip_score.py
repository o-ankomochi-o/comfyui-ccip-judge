"""CCIP score node.

Computes the mean CCIP distance between each image in the input batch
and the reference image pool. Reference comes from either an IMAGE
(batch) input or a folder path.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .common import comfy_image_to_pil_list, load_reference_images


DEFAULT_MODEL = "ccip-caformer_b36-24"


def _ccip_extract_many(pil_images, model: str):
    from imgutils.metrics import ccip_extract_feature
    feats = []
    for img in pil_images:
        feats.append(ccip_extract_feature(img, model=model))
    return feats


class CCIPScore:
    """Compute CCIP mean-distance per generated image against a reference pool."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.213, "min": 0.0, "max": 2.0, "step": 0.001}),
                "model": ("STRING", {"default": DEFAULT_MODEL, "multiline": False}),
                "reference_folder": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("distance", "pass_mask", "info")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "score"
    CATEGORY = "image_judge"

    def score(self, image, threshold, model, reference_folder, reference_image=None):
        ref_pils = load_reference_images(reference_image, reference_folder)
        if not ref_pils:
            raise RuntimeError(
                "CCIP_Score: no reference images. "
                "Connect reference_image or set reference_folder."
            )
        gen_pils = comfy_image_to_pil_list(image)
        if not gen_pils:
            return ([], [], "no input images")

        model_name = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL

        ref_feats = _ccip_extract_many(ref_pils, model_name)

        from imgutils.metrics import ccip_difference, ccip_extract_feature

        distances: List[float] = []
        passes: List[bool] = []
        for gen in gen_pils:
            gf = ccip_extract_feature(gen, model=model_name)
            dists = [ccip_difference(gf, rf, model=model_name) for rf in ref_feats]
            mean = float(np.mean(dists))
            distances.append(mean)
            passes.append(mean < threshold)

        info = (
            f"CCIP model={model_name} | refs={len(ref_pils)} | "
            f"n={len(distances)} | mean={float(np.mean(distances)):.4f} | "
            f"pass={sum(passes)}/{len(passes)} (<{threshold})"
        )
        return (distances, passes, info)
