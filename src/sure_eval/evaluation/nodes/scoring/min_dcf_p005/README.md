# minDCF p=0.05

## Purpose

`scoring/min_dcf_p005` computes normalized minimum detection cost from aligned
speaker-verification scores and binary labels. It searches all DET operating
points using a fixed target prior and costs; it does not calibrate scores or
choose a deployment threshold outside the evaluated trials.

## Task Scenarios

- SV minDCF: `sv.any.min_dcf.cosine_trial_scores_v1.min_dcf_p005_v1`.
- It is the default final metric node for the SV `min_dcf` route.

## Input

- Schema: `scores_plus_binary_labels`.
- Inputs are one-dimensional score and label arrays with equal non-zero length.
- Labels must be binary: `1` for target and `0` for nontarget.
- Both target and nontarget trials are required, and scores must be finite.

## Output

- Schema: `normalized_min_dcf_report`.
- Report fields include `metric_name`, `score`, `min_dcf`, `threshold`,
  `normalized`, `p_target`, `c_miss`, and `c_fa`.
- Normalized minDCF is unitless; lower scores are better.

## Versioned Computation

- Node id: `scoring/min_dcf_p005`.
- Version: `v1`.
- Parameters: `P_target=0.05`, `C_miss=1`, and `C_fa=1`.
- Detection cost is evaluated at each distinct descending score threshold and
  normalized by `min(C_miss * P_target, C_fa * (1 - P_target))`.
- The first minimum-cost operating point is returned.
- Internal stages:
  - `det_curve`
  - `normalized_detection_cost`

## Runtime and Assets

- Runtime: `in_process`.
- Dependency: NumPy from the base package.
- No model checkpoint, external binary, network access, or node-local setup is
  required.

## Source and References

- Local implementation:
  `src/sure_eval/evaluation/nodes/scoring/min_dcf_p005/node.py`.
- Shared DET implementation:
  `src/sure_eval/evaluation/nodes/scoring/common/sv_metrics.py`.

## Limitations

- The target prior and miss/false-alarm costs are fixed by the node version.
- The node accepts binary target/nontarget verification trials only.
- Trial weights, score calibration, confidence intervals, and condition-level
  breakdowns are not implemented.
