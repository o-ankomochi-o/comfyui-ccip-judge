# comfyui-ccip-judge

ComfyUI custom nodes that automate the 3-stage character-image selection
pipeline (CCIP + OKS + Angle) used in the LoRA IP-Adapter feedback workflow.

The package replicates the scoring logic from the original Jupyter notebook
and exposes it as 6 composable nodes, so generated batches can be filtered
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
- `opencv-contrib-python`, `numpy`, `Pillow`

## Known issues

### `pip install dghs-imgutils` fails on Python 3.13

Recent ComfyUI Windows portable builds ship an embedded **Python 3.13**.
On 3.13, installing `dghs-imgutils` fails with a numpy source-build error
(`ERROR: Unknown compiler(s) ...`). This is an upstream limitation, not
something this repo's `requirements.txt` can fix:

- `dghs-imgutils` (all versions, including the latest) pins `numpy<2`
  ([deepghs/imgutils#170](https://github.com/deepghs/imgutils/issues/170)).
- The newest `numpy<2` release (1.26.4) predates Python 3.13 and has no
  3.13 wheels, so pip falls back to building numpy from source, which
  fails on machines without a C/C++ compiler.

When `imgutils` is missing, only **CCIP Score** is affected; the other
nodes still work. A warning with instructions is printed at ComfyUI
startup, and CCIP Score raises the same instructions if executed.

**Options:**

1. **Recommended:** run ComfyUI on **Python 3.10–3.12** (a portable build
   that ships 3.12, or your own venv). `pip install -r requirements.txt`
   then works as-is.

2. **Workaround on Python 3.13 (unverified):** install `dghs-imgutils`
   without its dependency pins, then install its runtime dependencies
   with a current numpy. The `numpy<2` pin is conservative; the CCIP code
   path is expected to work on numpy 2.x (upstream is already discussing
   relaxing the pin in
   [deepghs/imgutils#175](https://github.com/deepghs/imgutils/issues/175)):

   ```bash
   python -m pip install dghs-imgutils --no-deps
   python -m pip install numpy pillow opencv-contrib-python onnxruntime ^
       huggingface-hub hbutils hfutils scikit-learn scipy pandas tqdm ^
       deprecation filelock
   ```

   (On ComfyUI Windows portable, replace `python` with
   `python_embeded\python.exe`. If you upgraded setuptools while
   debugging, restore it with `pip install "setuptools<82"` to satisfy
   torch.) If a `ModuleNotFoundError` for another package appears at
   runtime, `pip install` that package the same way. Please report back
   whether this works for you.

3. Once upstream relaxes the numpy pin, a normal
   `pip install dghs-imgutils` will work on 3.13 and this section will be
   removed.

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

See `examples/ccip_judge_minimal.json` for an example ComfyUI workflow.
The example uses Inspire Pack's `LoadImagesFromDir` node for batch loading.

## A/B/Bprime detector evaluation

The repository includes three pose-detector strategies for the pre-Study-1
measurement validation:

- `A`: current 0.3.0 behavior (legacy raw YOLOX, then anime retry)
- `B`: anime detector only
- `Bprime`: anime detector first, officially decoded YOLOX fallback

They are exposed through `extras/rejudge_pose_methods.py`, not as normal
workflow widgets, so research comparison cannot accidentally change an
existing ComfyUI workflow. The tool emits the fixed 14-column CSV consumed by
evocomfy's blind-review tooling. See
[`docs/JUDGE_AB_HANDOFF.md`](docs/JUDGE_AB_HANDOFF.md).

The image files and generated review packs are research data and must not be
committed to this public repository.

## Score semantics

| Metric         | Threshold (default) | Direction       | Meaning                                              |
|----------------|---------------------|-----------------|------------------------------------------------------|
| CCIP distance  | `< 0.213`           | lower is better | Embedding distance to learned character              |
| OKS            | `> 0.5`             | higher is better| Pose match to references                             |
| Angle distance | `< 0.5`             | lower is better | Camera-angle match to references                     |

The original LoRA evaluation report used CCIP `ccip-caformer_b36-24`
(F1=0.94) with the default threshold 0.213.

### Detection failures (fail-explicit contract)

When a generated image's character cannot be detected, the affected
score is **NaN** instead of a substitute value:

- OKS / Angle: pose extraction failed, or fewer than half of the valid
  references can be compared.
- CCIP: neither the anime person detector nor the anime face detector
  found a character (CCIP itself has no detection step, so this
  presence check guards against confidently scoring an empty image).

NaN compares false against any threshold, so a detection failure can
never pass ThreeStageFilter regardless of settings. ImageRouter writes
the failure to the CSV as an **empty score cell** plus a
`detect_failed` column (`pose`, `ccip`, or `pose+ccip`), ScoreOverlay
renders it as `FAIL`, and the bundled Optuna driver counts failures as
the worst score (OKS 0.0 / Angle 1.0 / CCIP 1.0) in `mean_*` and
`composite` objectives — same penalty as the legacy fail_score
behaviour. The `fail_score` widgets on OKS / Angle are deprecated and
ignored; they remain only so existing workflow JSONs keep loading.

References with fewer than 3 confident body keypoints are excluded from OKS.
References with fewer than 2 usable angle features are excluded from Angle.
If no valid reference remains, the node raises instead of judging blind.

## Reference aggregation

With multiple reference images, per-generated-image scores are averaged over
the comparable valid references. At least 50% of each metric's valid reference
pool must be comparable; otherwise the generated score is NaN. CCIP averages
over all references that pass the character-presence check.

## License

MIT


## 0.4.0 — pose targets

Score against authored keypoints instead of re-estimating a reference image:
set `reference_pose_json` on OKS/Angle to an OpenPose BODY-18 JSON (named
mapping to COCO-17; c=0 points stay invisible). `keypoint_set`
(`portrait`/`full_body`) restricts OKS to the joints the task framing can
contain. Score nodes emit a per-image `reasons` list
(`generated_no_person`, `insufficient_common_keypoints(ref=..,gen=..,common=..)`,
...) which ImageRouter writes to a `pose_debug` CSV column. A broken JSON
raises -- no silent image fallback inside experiments.
