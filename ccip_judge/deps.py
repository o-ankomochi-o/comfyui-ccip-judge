"""Dependency checks with actionable error messages.

`imgutils` (PyPI: dghs-imgutils) is imported lazily by the CCIP score
node. A bare ModuleNotFoundError at node-execution time tells users
nothing about how to fix it, so the import goes through
require_imgutils(), which explains the install command and the known
Python 3.13 limitation (dghs-imgutils pins numpy<2, which has no
Python 3.13 wheels — see deepghs/imgutils#170).
"""

from __future__ import annotations

import sys


def missing_imgutils_message() -> str:
    return (
        "CCIPJudge: required package 'dghs-imgutils' is not installed, "
        "so CCIP Score cannot run.\n"
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
        "[日本語] 依存パッケージ dghs-imgutils が見つかりません。Python 3.13 "
        "環境では通常の pip install が失敗する既知の問題があります。README の "
        "Known issues を参照してください。"
    )


def imgutils_available() -> bool:
    try:
        import imgutils  # noqa: F401
        return True
    except ImportError:
        return False


def require_imgutils_metrics():
    """Return imgutils.metrics, or raise a RuntimeError that explains the fix."""
    try:
        from imgutils import metrics
        return metrics
    except ImportError as e:
        raise RuntimeError(missing_imgutils_message()) from e
