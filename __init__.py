"""comfyui-ccip-judge

Anime character image evaluation nodes for ComfyUI.
Implements the 3-stage filter (CCIP + OKS + Angle) used in the
LoRA evaluation pipeline, with downstream filter and router nodes
for IP-Adapter feedback workflows.
"""

from .ccip_judge.deps import imgutils_available, missing_imgutils_message
from .ccip_judge.ccip_score import CCIPScore
from .ccip_judge.oks_score import OKSScore
from .ccip_judge.angle_score import AngleScore
from .ccip_judge.three_stage_filter import ThreeStageFilter
from .ccip_judge.image_router import ImageRouter
from .ccip_judge.score_overlay import ScoreOverlay

NODE_CLASS_MAPPINGS = {
    "CCIPJudge_CCIPScore": CCIPScore,
    "CCIPJudge_OKSScore": OKSScore,
    "CCIPJudge_AngleScore": AngleScore,
    "CCIPJudge_ThreeStageFilter": ThreeStageFilter,
    "CCIPJudge_ImageRouter": ImageRouter,
    "CCIPJudge_ScoreOverlay": ScoreOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCIPJudge_CCIPScore": "CCIP Score",
    "CCIPJudge_OKSScore": "OKS Score",
    "CCIPJudge_AngleScore": "Angle Score",
    "CCIPJudge_ThreeStageFilter": "Three-Stage Filter",
    "CCIPJudge_ImageRouter": "Image Router (liked/disliked)",
    "CCIPJudge_ScoreOverlay": "Score Overlay (preview)",
}

WEB_DIRECTORY = None

# Surface missing dependencies at startup instead of a bare
# ModuleNotFoundError when the node first executes. Nodes still
# register so existing workflows keep loading.
if not imgutils_available():
    print(f"\n[comfyui-ccip-judge] WARNING:\n{missing_imgutils_message()}\n")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
