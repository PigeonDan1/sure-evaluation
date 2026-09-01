# DET EER

## Purpose

`scoring/det_eer` computes equal error rate from aligned speaker-verification
scores and binary labels. It reports the operating point where miss and false
alarm rates are equal; it does not perform embedding extraction or trial
scoring.

## Task Scenarios

- SV EER: `sv.any.eer.cosine_trial_scores_v1.det_eer_v1`.
- It is the default final metric node for the SV `eer` route.

## Input

- Schema: `scores_plus_binary_labels`.
- Inputs are one-dimensional score and label arrays with equal non-zero length.
- Labels must be binary: `1` for target and `0` for nontarget.
- Both target and nontarget trials are required, and scores must be finite.

## Output

- Schema: `eer_report`.
- Report fields include `metric_name`, `score`, `eer_percent`, `threshold`, and
  `unit`.
- EER is reported as a percentage; lower scores are better.

## Versioned Computation

- Node id: `scoring/det_eer`.
- Version: `v1`.
- Scores are sorted in descending order with stable tie handling to construct
  miss and false-alarm rates at distinct thresholds.
- Exact DET crossings are used directly. Otherwise, the first crossing is
  linearly interpolated; if no crossing exists, the closest DET point is used.
- Internal stages:
  - `det_curve`
  - `eer_linear_interpolation`

## Runtime and Assets

- Runtime: `in_process`.
- Dependency: NumPy from the base package.
- No model checkpoint, external binary, network access, or node-local setup is
  required.

## Source and References

- Local implementation:
  `src/sure_eval/evaluation/nodes/scoring/det_eer/node.py`.
- Shared DET implementation:
  `src/sure_eval/evaluation/nodes/scoring/common/sv_metrics.py`.

## Limitations

- The node accepts binary target/nontarget verification trials only.
- Trial weights, calibration, confidence intervals, and condition-level
  breakdowns are not implemented.
- The reported threshold is in the same score domain as the input scorer.
