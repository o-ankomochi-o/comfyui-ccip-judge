"""Score overlay node.

Annotates each image with its CCIP / OKS / Angle scores and the
LIKED / disliked verdict, drawn as a caption band along the top of
the image. Intended to be connected to PreviewImage so the user can
inspect per-image judgement directly in the ComfyUI graph.
"""

from __future__ import annotations

from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from .common import comfy_image_to_pil_list, pil_list_to_comfy_image


def _pop_scalar(value, default=None):
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def _maybe_floats(values) -> Optional[List[float]]:
    if values is None:
        return None
    if isinstance(values, list) and len(values) == 0:
        return None
    return [float(v) for v in values]


def _load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort TrueType font; falls back to PIL bitmap font."""
    candidates = [
        "DejaVuSans.ttf",  # bundled with PIL on many platforms
        "arial.ttf",
        "Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _draw_caption(img: Image.Image, text: str, verdict_ok: bool) -> Image.Image:
    """Draw a translucent caption band along the top of the image."""
    w, h = img.size
    band_h = max(28, h // 24)
    font_size = max(14, band_h - 10)
    font = _load_font(font_size)

    out = img.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bg_color = (0, 128, 0, 200) if verdict_ok else (160, 32, 32, 200)
    draw.rectangle([(0, 0), (w, band_h)], fill=bg_color)
    draw.text((8, max(2, (band_h - font_size) // 2)), text, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(out, overlay).convert("RGB")
    return out


class ScoreOverlay:
    """Draw a per-image score caption for visual confirmation in ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pass_mask": ("BOOLEAN", {"forceInput": True}),
                "ccip_threshold": ("FLOAT", {"default": 0.213, "min": 0.0, "max": 2.0, "step": 0.001}),
                "oks_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "angle_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "ccip_distance": ("FLOAT", {"forceInput": True}),
                "oks": ("FLOAT", {"forceInput": True}),
                "angle_distance": ("FLOAT", {"forceInput": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("annotated",)
    FUNCTION = "overlay"
    CATEGORY = "image_judge"

    def overlay(self, image, pass_mask, ccip_threshold, oks_threshold, angle_threshold,
                ccip_distance=None, oks=None, angle_distance=None):
        ccip_t = float(_pop_scalar(ccip_threshold, 0.213))
        oks_t = float(_pop_scalar(oks_threshold, 0.5))
        ang_t = float(_pop_scalar(angle_threshold, 0.5))

        pil_all: List[Image.Image] = []
        for img_tensor in image:
            pil_all.extend(comfy_image_to_pil_list(img_tensor))

        mask = [bool(v) for v in pass_mask]
        n = min(len(pil_all), len(mask))

        ccip_list = _maybe_floats(ccip_distance)
        oks_list = _maybe_floats(oks)
        ang_list = _maybe_floats(angle_distance)

        annotated: List[Image.Image] = []
        for i in range(n):
            verdict = "LIKED" if mask[i] else "disliked"
            parts = [f"idx={i:02d}"]
            if ccip_list and i < len(ccip_list):
                v = ccip_list[i]
                tag = "OK" if v < ccip_t else "NG"
                parts.append(f"CCIP={v:.3f}[{tag}]")
            if oks_list and i < len(oks_list):
                v = oks_list[i]
                tag = "OK" if v > oks_t else "NG"
                parts.append(f"OKS={v:.3f}[{tag}]")
            if ang_list and i < len(ang_list):
                v = ang_list[i]
                tag = "OK" if v < ang_t else "NG"
                parts.append(f"Angle={v:.3f}[{tag}]")
            parts.append(f"-> {verdict}")
            label = "  ".join(parts)
            annotated.append(_draw_caption(pil_all[i], label, mask[i]))

        return (pil_list_to_comfy_image(annotated),)
