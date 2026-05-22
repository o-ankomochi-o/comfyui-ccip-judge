# comfyui-ccip-judge

ComfyUI custom nodes that automate the 3-stage character-image selection
pipeline (CCIP + OKS + Angle) used in the LoRA IP-Adapter feedback workflow.

The package replicates the scoring logic from the original Jupyter notebook
and exposes it as 5 composable nodes, so generated batches can be filtered
and routed back into `IPAdapterAdvanced` as liked / disliked references.

## Nodes

| Node                | Inputs                                                  | Outputs                          | Notes |
|---------------------|---------------------------------------------------------|----------------------------------|-------|
| **CCIP Score**      | `image`, `threshold`, `model`, `reference_folder`, optional `reference_image` | `distance` (FLOAT list), `pass_mask` (BOOLEAN list), `info` | Uses `dghs-imgutils`; default model `ccip-caformer_b36-24` |
| **OKS Score**       | `image`, `threshold`, `reference_folder`, `fail_score`, optional `reference_image` | `oks`, `pass_mask`, `info`       | Runs DWPose internally (downloads via HuggingFace Hub on first use); averages OKS across the reference pool |
| **Angle Score**     | same shape as OKS                                       | `angle_distance`, `pass_mask`, `info` | 4 angle features RMS-combined; averages across references |
| **Three-Stage Filter** | `ccip_distance`, `oks`, `angle_distance` (FLOAT lists) + thresholds | `pass_mask` (BOOLEAN list), `info` | ANDs the three per-image decisions |
| **Image Router**    | `image`, `pass_mask`, optional save dirs/prefixes        | `liked` (IMAGE), `disliked` (IMAGE), `info` | Always emits both IMAGE branches; saves to disk if directories are set |

All score nodes accept the reference pool either as an `IMAGE` batch
(connect multiple `LoadImage` via `ImageBatch`) or as a folder path. If
`reference_image` is connected it takes precedence over `reference_folder`.

## Installation

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/o-ankomochi-o/comfyui-ccip-judge
cd comfyui-ccip-judge
pip install -r requirements.txt
```

Requirements:

- `dghs-imgutils` (CCIP)
- `onnxruntime` (DWPose). Use `onnxruntime-gpu` for CUDA.
- `huggingface_hub` (downloads `yzd-v/DWPose` ONNX weights on first run)
- `opencv-python`, `numpy`, `Pillow`

## Typical workflow

```
KSampler -> VAEDecode -> [batch of generated images]
                            |
                            +-> CCIP Score   --(distance)----+
                            +-> OKS Score    --(oks)---------+--> Three-Stage Filter --(pass_mask)--+
                            +-> Angle Score  --(angle_dist)--+                                       |
                            |                                                                       v
                            +---------------------------------------------------> Image Router (liked / disliked)
                                                                                                     |
                                                                              IPAdapterAdvanced.image  / image_negative
```

See `examples/image_judge_minimal.json` for a minimal ComfyUI workflow.

## Score semantics

| Metric         | Threshold (default) | Direction       | Meaning                                              |
|----------------|---------------------|-----------------|------------------------------------------------------|
| CCIP distance  | `< 0.213`           | lower is better | Embedding distance to learned character              |
| OKS            | `> 0.5`             | higher is better| Pose match to references                             |
| Angle distance | `< 0.5`             | lower is better | Camera-angle match to references                     |

The original LoRA evaluation report used CCIP `ccip-caformer_b36-24`
(F1=0.94) with the default threshold 0.213.

## Reference aggregation

With multiple reference images, per-generated-image scores are averaged
across the pool. This matches the user-selected configuration ("平均")
and the original notebook's behaviour for CCIP.

## License

MIT
