"""Dependency checks with actionable error messages."""

from __future__ import annotations

import sys


def missing_imgutils_message() -> str:
    return (
        "CCIPJudge: required package 'dghs-imgutils' is not installed. "
        "CCIP Score and anime-person pose detection cannot run.\n"
        "\n"
        "Install it into ComfyUI's Python:\n"
        f'  "{sys.executable}" -m pip install -r '
        "custom_nodes/comfyui-ccip-judge/requirements.txt\n"
        "\n"
        "Known issue on Python 3.13 (e.g. recent ComfyUI Windows portable):\n"
        "'pip install dghs-imgutils' fails because dghs-imgutils pins "
        "numpy<2, which has no Python 3.13 wheels. Either run ComfyUI on "
        "Python 3.10-3.12, or see 'Known issues' in the comfyui-ccip-judge "
        "README for a --no-deps workaround.\n"
        "\n"
        "[日本語] 必須パッケージ dghs-imgutils が見つかりません。"
        "CCIP Score とポーズ人物検出は実行できません。README の "
        "Known issues を参照してください。"
    )


def imgutils_available() -> bool:
    try:
        import imgutils  # noqa: F401

        return True
    except ImportError:
        return False


def require_imgutils_metrics():
    """Return imgutils.metrics, or raise an actionable RuntimeError."""
    try:
        from imgutils import metrics

        return metrics
    except ImportError as e:
        raise RuntimeError(missing_imgutils_message()) from e
