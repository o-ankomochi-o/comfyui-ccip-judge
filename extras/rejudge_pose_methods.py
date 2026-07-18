"""Re-score an image manifest with pose methods A, B and Bprime.

Input manifest CSV columns:
  image_sha256,image_path

The full 64-character SHA-256 must describe the original file bytes. If the
field is empty it is calculated. If it is present it is verified.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ccip_judge.common import file_sha256, load_reference_images
from ccip_judge.dwpose_runner import POSE_METHODS
from ccip_judge.rejudge import (
    REJUDGE_FIELDS,
    RejudgeThresholds,
    current_judge_sha,
    prepare_method_references,
    rejudge_image,
    validate_judge_sha,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--reference-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing image/method rows to an existing validated CSV.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(POSE_METHODS),
        choices=list(POSE_METHODS),
    )
    parser.add_argument("--oks-threshold", type=float, default=0.5)
    parser.add_argument("--angle-threshold", type=float, default=0.5)
    parser.add_argument("--min-valid-reference-ratio", type=float, default=0.5)
    parser.add_argument(
        "--pass-rule",
        choices=["oks", "oks_and_angle"],
        default="oks",
        help="Study 1 uses 'oks'; use 'oks_and_angle' only for the full filter.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit an uncommitted judge only for development smoke tests.",
    )
    return parser.parse_args()


def load_manifest(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))
    required = {"image_sha256", "image_path"}
    if not rows:
        raise ValueError("manifest is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return rows


def main():
    args = parse_args()
    references = load_reference_images(None, args.reference_folder)
    if not references:
        raise RuntimeError(f"no reference images found in {args.reference_folder}")

    thresholds = RejudgeThresholds(
        oks=args.oks_threshold,
        angle=args.angle_threshold,
        min_valid_reference_ratio=args.min_valid_reference_ratio,
        pass_rule=args.pass_rule,
    )
    method_refs = {
        method: prepare_method_references(references, method)
        for method in args.methods
    }
    judge_sha = current_judge_sha()
    validate_judge_sha(judge_sha, allow_dirty=args.allow_dirty)
    output_path = Path(args.output)
    completed = set()
    if output_path.exists():
        if not args.resume:
            raise RuntimeError(
                f"{output_path} already exists; use --resume or choose a new output"
            )
        with output_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            if tuple(reader.fieldnames or ()) != REJUDGE_FIELDS:
                raise RuntimeError(
                    f"{output_path} does not have the frozen 14-column schema"
                )
            for row in reader:
                completed.add((row["image_sha256"], row["method"]))

    manifest = load_manifest(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_path.exists() else "w"
    written = 0
    with output_path.open(mode, encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=REJUDGE_FIELDS, extrasaction="raise")
        if mode == "w":
            writer.writeheader()
            fp.flush()
        for image_index, item in enumerate(manifest, start=1):
            path = Path(item["image_path"])
            actual_sha = file_sha256(path)
            expected_sha = (item.get("image_sha256") or "").strip().lower()
            if expected_sha and expected_sha != actual_sha:
                raise RuntimeError(
                    f"SHA-256 mismatch for {path}: "
                    f"manifest={expected_sha}, actual={actual_sha}"
                )
            missing_methods = [
                method
                for method in args.methods
                if (actual_sha, method) not in completed
            ]
            if not missing_methods:
                continue
            with Image.open(path) as source:
                image = source.convert("RGB")
            for method in missing_methods:
                writer.writerow(
                    rejudge_image(
                        image=image,
                        image_sha256=actual_sha,
                        method=method,
                        references=method_refs[method],
                        thresholds=thresholds,
                        judge_sha=judge_sha,
                    )
                )
                fp.flush()
                completed.add((actual_sha, method))
                written += 1
            if image_index % 25 == 0:
                print(f"processed {image_index}/{len(manifest)} images")
    print(f"wrote {written} new rows to {output_path}")


if __name__ == "__main__":
    main()
