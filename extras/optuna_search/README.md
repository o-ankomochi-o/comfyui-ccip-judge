# Optuna parameter search for CCIP Judge

A standalone script that drives a running ComfyUI instance over the HTTP API,
mutates the IPAdapter / LoRA / sampler parameters per Optuna's suggestions,
and uses the CSV that **ImageRouter** writes as the objective function.

Supports two phases out of the box:

| Phase | Sampler | Use |
| --- | --- | --- |
| A. Exploration | `--sampler grid` | Equivalent to an XY-Plot sweep, fully enumerates the supplied grid points |
| B. Refinement | `--sampler tpe` (default) | Bayesian TPE; efficient at narrowing in on the best region |

## Install

```bash
cd <ComfyUI>/custom_nodes/comfyui-ccip-judge/extras/optuna_search
pip install -r requirements.txt
```

## Prepare the workflow

1. Open your tuning workflow in ComfyUI.
2. Make sure CCIP Score / OKS Score / Angle Score / ThreeStageFilter / **ImageRouter**
   are wired with `csv_dir` set to a folder this script can read.
3. Recommended: set `EmptyLatentImage.batch_size = 4` (one trial ~15 s).
4. Drop all required LoadImage references (CCIP × N, Pose × 1) once, so they
   persist between trials.
5. Export the workflow via **File → Save (API Format)** as `workflow_api.json`.

> The API-format JSON is a flat dict keyed by `node_id`, *not* the UI-format
> with `nodes`/`links`. The script will error out if you give it the wrong one.

## Configure search space

Copy the template and edit:

```bash
cp search_space.example.yaml search_space.yaml
$EDITOR search_space.yaml
```

Key fields:

| Field | Purpose |
| --- | --- |
| `comfyui_url` | ComfyUI HTTP endpoint (e.g. `http://127.0.0.1:8188`) |
| `workflow_template` | Path to the API-format JSON you just exported |
| `csv_dir` | Same path as `ImageRouter.csv_dir` in the workflow |
| `n_trials` | Total trial count (50 is a good start for TPE) |
| `sampler` | `tpe` / `grid` / `random` |
| `objective` | `liked_rate` (default), `liked_count`, `mean_ccip`, `mean_oks`, `mean_angle`, `composite` |
| `parameters` | Each parameter has `node_id`, `input_name`, `type`, plus range / choices |

Find `node_id` values by opening the API JSON and matching the `class_type` of
each block. Typical IDs in the bundled workflow:

| Node | Likely class_type | Common inputs to tune |
| --- | --- | --- |
| IPAdapterAdvanced | `IPAdapterAdvanced` | `weight`, `weight_type`, `start_at`, `end_at`, `embeds_scaling` |
| Power Lora Loader (rgthree) | `Power Lora Loader (rgthree)` | `lora_strength` (per LoRA entry — check the JSON shape) |
| KSampler | `KSampler` / `KSamplerAdvanced` | `cfg`, `steps`, `sampler_name`, `scheduler` |

## Run

```bash
# Phase A: grid sweep (enumerates the `grid:` lists in search_space.yaml)
python optuna_search.py --config search_space.yaml --sampler grid --n-trials 9

# Phase B: TPE refinement around the best region (resumes the same DB)
python optuna_search.py --config search_space.yaml --sampler tpe --n-trials 50 --resume
```

Output:
- `<study_name>_trials.csv` — all trials with params + objective values.
- `<study_name>.db` — SQLite store with full Optuna history (visualize via `optuna-dashboard <study>.db`).

## Visualize

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///ccip_judge_tuning.db
```

Open `http://127.0.0.1:8080` to inspect parallel-coordinate plots, importance
plots, slice plots, etc.

## Objective metrics

| objective | direction | meaning |
| --- | --- | --- |
| `liked_rate` | maximize | fraction of batch passing all 3 filters |
| `liked_count` | maximize | raw count (use when batch_size varies) |
| `mean_ccip` | minimize | average CCIP distance (lower = more similar) |
| `mean_oks` | maximize | average OKS (higher = closer to reference pose) |
| `mean_angle` | minimize | average angle distance |
| `composite` | maximize | `-mean_ccip + mean_oks - mean_angle` (weighted combo) |

Switch via the `objective:` and `direction:` keys in YAML.

## Tips

- **Pose detection failures (OKS=0.0, Angle=1.0) drag the means down.**
  Consider `liked_count` if you only care about "passed/not passed".
- **Start with a small grid** to confirm the wiring works (`--sampler grid --n-trials 3`)
  before launching a long TPE run.
- **Use `--resume`** to continue a study across sessions; the SQLite DB tracks state.
- **Different seeds per trial** are recommended for stable signal.  
  In the workflow, set `KSampler.control_after_generate = increment` so the
  seed advances naturally across the API runs.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `looks like UI-format` error | Used the wrong JSON. Re-export with "Save (API Format)" |
| `no scores_*.csv found` | ImageRouter not connected, `csv_dir` mismatch, or score inputs not wired |
| `KeyError: node_id 'XXX' not in workflow` | The id in search_space.yaml doesn't exist in the API JSON; open the JSON and update |
| ComfyUI returns 400 | The workflow has unconnected required inputs (typically LoadImage missing files) |
| Trials all return the same value | The mutated parameter isn't actually wired into the active path (e.g. wrong IPAdapter node id) |
