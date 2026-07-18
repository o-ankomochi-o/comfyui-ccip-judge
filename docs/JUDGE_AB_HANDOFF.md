# Judge A/B/Bprime handoff

This repository produces pose re-judging measurements. It does not import
evocomfy. The integration boundary is a CSV file.

## Methods

- `A`: exact 0.3.0 ordering: legacy raw YOLOX first, anime retry after
  keypoint collapse.
- `B`: anime person detector only.
- `Bprime`: anime detector first, officially decoded/NMS-filtered YOLOX only
  after anime keypoint collapse.

`Bprime` is a time-boxed candidate. It must not block selection between `A`
and `B` if parity validation is not completed by the pre-registered deadline.

## Input manifest

UTF-8 CSV:

```csv
image_sha256,image_path
<64 lowercase hex characters>,<absolute or gpu01-local path>
```

`image_sha256` is the SHA-256 of the original file bytes. Files must not be
re-encoded. The CLI verifies existing hashes and calculates missing hashes.

## Run

Run inside the same Python environment as ComfyUI:

```powershell
python extras/rejudge_pose_methods.py `
  --manifest G:\path\judge_ab_index.csv `
  --reference-pose-json G:\path\action-reach.json `
  --keypoint-set full_body `
  --output G:\path\judge_ab_scores.csv `
  --methods A B Bprime `
  --oks-threshold 0.5 `
  --angle-threshold 0.5 `
  --min-valid-reference-ratio 0.5 `
  --pass-rule oks
```

For v10cf stability comparisons, `--reference-pose-json action-reach.json`
and `--keypoint-set full_body` are required. The OpenPose BODY-18 coordinates
are mapped to COCO-17 by joint name and scored directly; no reference image is
rendered or pose-estimated. This preserves the authored target definition used
by judge 0.4.3. `--reference-folder` remains available as a mutually exclusive
alternative for experiments whose target was defined by reference images.

The CLI writes `<output>.provenance.json` beside the frozen 14-column CSV. It
records the full reference SHA-256, joint set, thresholds, methods and judge
commit. Resume refuses a different provenance file, preventing measurements
from different reference definitions or thresholds from being mixed.

`--pass-rule oks` is the Study 1 requirement comparison. The Angle value is
still measured, but an unavailable or failing Angle does not change `pass`.
Use `oks_and_angle` only when explicitly evaluating the full three-stage
filter. Q2 must supply the newly calibrated threshold rather than silently
reusing the example value above.

If execution is interrupted, run the same command with `--resume`. Existing
rows are accepted only when their header exactly matches the frozen schema;
completed `(image_sha256, method)` pairs are skipped.

Formal re-judging refuses an uncommitted worktree so that `judge_sha` is an
unambiguous 40-character commit. `--allow-dirty` exists only for small
development smoke tests and must never be used for the 3,097-image run.

Do not run all 3,097 images first. Start with a smoke manifest containing
known success, known failure, line art, small subject and multi-person cases.

## Output contract

The output has exactly these 14 columns:

```text
image_sha256
method
oks
angle
pass
failure_reason
detector_used
confident_keypoints
valid_reference_count
bbox
latency_ms
judge_sha
detector_model_sha256
pose_model_sha256
```

For `--pass-rule oks`, `valid_reference_count` is the number of references
that contributed to OKS. For `oks_and_angle`, it is the smaller of the OKS
and Angle contributing-reference counts.

The score CSV contains method identities and must never be supplied directly
to the blind-review UI. evocomfy joins it to its independent image index,
selects the frozen sample, and creates a blinded pack without method names,
scores or old verdicts.

## Required gpu01 sequence

1. Preserve every source image until method selection is complete.
2. Pull the reviewed judge commit only after explicit approval.
3. Restart ComfyUI only after recording the previous judge SHA.
4. Run a small smoke manifest.
5. Verify 3 rows per image and all 14 columns.
6. Verify the provenance JSON contains the frozen `action-reach.json` SHA-256
   and `keypoint_set=full_body`.
7. Run the frozen 3,097-image manifest.
8. Copy the CSV and its provenance JSON to evocomfy; do not create an import
   dependency.
9. Restore the previous SHA if smoke validation fails.

## Freeze record

Before Q2 and Study 1, record:

- judge Git SHA and release tag
- Python and dependency lock
- ONNX Runtime provider
- anime detector artifact SHA-256
- YOLOX artifact SHA-256
- pose artifact SHA-256
- OKS and Angle thresholds
- minimum valid-reference ratio
- selected method
