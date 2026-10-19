# DOA Localization Scoring

## Purpose

`scoring/doa_localization` computes angular localization and thresholded
detection/localization metrics from a normalized DOA bundle. It does not run a
localization model or infer source activity from audio.

## Task Scenarios

- All `DOA/any/*` routes use this node as their scoring stage.
- It is the default and only scoring node for the DOA v1 task.

## Input

- Schema: `doa_normalized_bundle`.
- Receives exactly aligned reference and prediction frames from
  `normalization/doa_jsonl`.
- Active sources are selected according to reference activity and the declared
  prediction activity policy.

## Output

- Schema: `doa_localization_report`.
- Reports `mae`, `rmse`, `acc_10`, `acc_15`, `acc_30`,
  `localization_recall`, `localization_precision`, and `localization_f1`.
- Reports assigned errors, TP/FP/FN counts, per-recording summaries, and micro
  and macro-recording aggregation details. MAE and RMSE are lower-is-better;
  the remaining metrics are higher-is-better.

## Versioned Computation

- Node id: `scoring/doa_localization`.
- Version: `v1`.
- Uses circular azimuth distance in 2D and spherical great-circle distance in
  3D.
- Performs frame-local minimum-total-angular-cost one-to-one assignment before
  applying the strict `error_deg < threshold_deg` gate.
- Uses a deterministic pure-Python rectangular Hungarian implementation.
- Micro aggregation pools errors or counts; macro recording averages defined
  per-recording metric values.

## Runtime and Assets

- Runtime: `in_process`.
- Runs in the base package with no optional dependencies, checkpoints, or
  external assets.

## Source and References

- Source: local SURE angular-localization scoring implementation.

## Limitations

- Assignment minimizes raw angular cost; it does not maximize gated true
  positives.
- MAE and RMSE use assigned pairs only. Unmatched sources are represented in
  thresholded counts rather than converted into synthetic angular errors.
- Metrics with undefined denominators are reported as `null`.
