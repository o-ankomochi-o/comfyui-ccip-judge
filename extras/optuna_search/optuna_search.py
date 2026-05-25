"""Optuna-based parameter search that drives a running ComfyUI instance and
uses CCIP Judge's CSV output as the objective function.

The script:
  1. Loads an API-format workflow JSON (export via "Save (API Format)" in ComfyUI).
  2. For each Optuna trial, mutates the workflow's widget inputs per the
     parameter mapping in search_space.yaml.
  3. POSTs the mutated workflow to ComfyUI's /prompt endpoint.
  4. Polls /history until the prompt finishes.
  5. Reads the latest scores_<ts>.csv written by ImageRouter and computes
     the objective value (LIKED rate by default).
  6. Returns the value to Optuna so TPE / Grid / Random samplers can iterate.

Usage:
    pip install -r requirements.txt
    python optuna_search.py --config search_space.yaml
    python optuna_search.py --config search_space.yaml --sampler grid --n-trials 9
    python optuna_search.py --config search_space.yaml --resume
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml

try:
    import optuna
except ImportError as e:
    print("optuna is required. Run: pip install -r requirements.txt", file=sys.stderr)
    raise


def parse_args():
    p = argparse.ArgumentParser(description="Optuna search driver for ComfyUI + CCIP Judge.")
    p.add_argument("--config", default="search_space.yaml",
                   help="Path to search space YAML.")
    p.add_argument("--sampler", choices=["tpe", "grid", "random"], default=None,
                   help="Override the sampler from config.")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Override the trial count.")
    p.add_argument("--study-name", default=None,
                   help="Override the study name (used as DB key).")
    p.add_argument("--storage", default=None,
                   help="Override the optuna storage URL (default: sqlite local).")
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing study if present.")
    return p.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = ["comfyui_url", "workflow_template", "csv_dir", "parameters"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"search_space.yaml is missing required key: {key}")
    cfg.setdefault("n_trials", 30)
    cfg.setdefault("sampler", "tpe")
    cfg.setdefault("direction", "maximize")
    cfg.setdefault("objective", "liked_rate")
    cfg.setdefault("poll_interval_sec", 2.0)
    cfg.setdefault("trial_timeout_sec", 600)
    return cfg


def load_workflow_template(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        wf = json.load(f)
    if any(k in wf for k in ("nodes", "links")):
        raise ValueError(
            f"{path} looks like UI-format. Open the workflow in ComfyUI and "
            f"choose 'Save (API Format)' (File menu) to export the API-format JSON."
        )
    return wf


def apply_params(workflow: Dict[str, Any], trial: optuna.Trial,
                 parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Mutate the workflow dict per Optuna's trial suggestions."""
    wf = copy.deepcopy(workflow)
    for name, spec in parameters.items():
        node_id = str(spec["node_id"])
        input_name = spec["input_name"]
        ptype = spec["type"]

        if ptype == "float":
            value = trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                        step=spec.get("step"))
        elif ptype == "int":
            value = trial.suggest_int(name, int(spec["low"]), int(spec["high"]),
                                      step=int(spec.get("step", 1)))
        elif ptype == "categorical":
            value = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unsupported parameter type for '{name}': {ptype}")

        if node_id not in wf:
            raise KeyError(f"node_id '{node_id}' not in workflow (parameter '{name}')")
        wf[node_id]["inputs"][input_name] = value
    return wf


def queue_prompt(comfyui_url: str, prompt: Dict[str, Any], client_id: str) -> str:
    r = requests.post(f"{comfyui_url}/prompt",
                      json={"prompt": prompt, "client_id": client_id},
                      timeout=30)
    r.raise_for_status()
    body = r.json()
    if "prompt_id" not in body:
        raise RuntimeError(f"ComfyUI /prompt unexpected response: {body}")
    return body["prompt_id"]


def wait_for_prompt(comfyui_url: str, prompt_id: str,
                    timeout_sec: float, poll_sec: float) -> Dict[str, Any]:
    start = time.time()
    while time.time() - start < timeout_sec:
        r = requests.get(f"{comfyui_url}/history/{prompt_id}", timeout=10)
        if r.status_code == 200:
            history = r.json()
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI reported error for {prompt_id}: {status}")
                if status.get("completed", False) or status.get("status_str") == "success":
                    return history[prompt_id]
        time.sleep(poll_sec)
    raise TimeoutError(f"prompt {prompt_id} did not complete within {timeout_sec}s")


def find_latest_csv(csv_dir: str, after_ts: float) -> str | None:
    """Return path of the newest scores_*.csv whose mtime is > after_ts."""
    candidates = []
    for path in glob.glob(os.path.join(csv_dir, "scores_*.csv")):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime > after_ts:
            candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_objective(rows: List[Dict[str, str]], objective: str) -> float:
    if not rows:
        return float("nan")

    n = len(rows)
    n_liked = sum(1 for r in rows if r.get("verdict") == "LIKED")

    def safe_float(key, default=None):
        vals = []
        for r in rows:
            v = r.get(key, "")
            if v == "" or v is None:
                continue
            try:
                vals.append(float(v))
            except ValueError:
                continue
        return vals

    if objective == "liked_rate":
        return n_liked / n
    if objective == "liked_count":
        return float(n_liked)
    if objective == "mean_ccip":
        vs = safe_float("ccip")
        return sum(vs) / len(vs) if vs else float("nan")
    if objective == "mean_oks":
        vs = safe_float("oks")
        return sum(vs) / len(vs) if vs else float("nan")
    if objective == "mean_angle":
        vs = safe_float("angle")
        return sum(vs) / len(vs) if vs else float("nan")
    if objective == "composite":
        ccip = safe_float("ccip")
        oks = safe_float("oks")
        ang = safe_float("angle")
        score = 0.0
        if ccip: score += -1.0 * (sum(ccip) / len(ccip))
        if oks:  score += 1.0 * (sum(oks) / len(oks))
        if ang:  score += -1.0 * (sum(ang) / len(ang))
        return score
    raise ValueError(f"unknown objective: {objective}")


def build_sampler(name: str, parameters: Dict[str, Dict[str, Any]], seed: int = 42):
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "grid":
        grid = {}
        for pname, spec in parameters.items():
            ptype = spec["type"]
            if ptype == "categorical":
                grid[pname] = list(spec["choices"])
            elif "grid" in spec:
                grid[pname] = list(spec["grid"])
            elif ptype in ("float", "int"):
                lo, hi = spec["low"], spec["high"]
                grid[pname] = [lo, (lo + hi) / 2, hi]
            else:
                raise ValueError(f"cannot build grid for param '{pname}' (type {ptype})")
        return optuna.samplers.GridSampler(grid)
    raise ValueError(f"unknown sampler: {name}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.sampler:
        cfg["sampler"] = args.sampler
    if args.n_trials is not None:
        cfg["n_trials"] = args.n_trials

    workflow_template = load_workflow_template(cfg["workflow_template"])
    csv_dir = cfg["csv_dir"]
    comfyui_url = cfg["comfyui_url"].rstrip("/")
    client_id = f"optuna_search_{uuid.uuid4().hex[:8]}"

    sampler = build_sampler(cfg["sampler"], cfg["parameters"])
    study_name = args.study_name or cfg.get("study_name", "ccip_judge_tuning")
    storage = args.storage or cfg.get("storage") or f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        direction=cfg["direction"],
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=args.resume,
    )

    print(f"== Optuna study '{study_name}' (storage={storage}, sampler={cfg['sampler']}) ==")
    print(f"   direction={cfg['direction']}, objective={cfg['objective']}, n_trials={cfg['n_trials']}")

    def objective_fn(trial: optuna.Trial) -> float:
        wf = apply_params(workflow_template, trial, cfg["parameters"])
        start_ts = time.time()
        prompt_id = queue_prompt(comfyui_url, wf, client_id)
        wait_for_prompt(comfyui_url, prompt_id,
                        timeout_sec=cfg["trial_timeout_sec"],
                        poll_sec=cfg["poll_interval_sec"])
        csv_path = find_latest_csv(csv_dir, start_ts - 5)
        if csv_path is None:
            raise RuntimeError(
                f"no scores_*.csv found in {csv_dir} after this trial. "
                "Verify ImageRouter.csv_dir matches and that the workflow has all scores wired in."
            )
        rows = read_csv_rows(csv_path)
        value = compute_objective(rows, cfg["objective"])
        trial.set_user_attr("csv_path", csv_path)
        trial.set_user_attr("n_rows", len(rows))
        trial.set_user_attr("liked", sum(1 for r in rows if r.get("verdict") == "LIKED"))
        return value

    study.optimize(objective_fn, n_trials=cfg["n_trials"], show_progress_bar=True)

    print()
    print("=" * 70)
    print(f"Best objective: {study.best_value}")
    print(f"Best params:    {study.best_params}")
    print("=" * 70)

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    out_csv = Path(study_name + "_trials.csv")
    df.to_csv(out_csv, index=False)
    print(f"trials written to {out_csv}")


if __name__ == "__main__":
    main()
