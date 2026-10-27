# RIR Late-Window IACC MAE

## Purpose

Reports the absolute difference between prediction and reference IACC over
the late binaural RIR window.

## Task Scenarios

- Selected by the RIR binaural IACC bundle or as an atomic metric.
- Not used for arrays with more than two channels; larger arrays use
  channel-0-relative spatial metrics.

## Input

- Schema: `binaural_rir_samples_jsonl`.
- Required roles: `sample_id`, `prediction_rir`, and `reference_rir`.
- Each file must have exactly two channels and matching sample rates. NumPy
  inputs require `sample_rate`.
- Rows align by `sample_id`.

## Output

- Schema: `rir_iacc_late_mae_score`.
- Lower is better; unit is a correlation coefficient.
- Each sample reports prediction IACC, reference IACC, and absolute error.
  Samples are weighted equally.

## Versioned Computation

- Node `scoring/rir_iacc_late_mae`, version `v1`.
- Measures the tail after the ch0 direct peak plus 80 ms, using an absolute
  normalized cross-correlation search within +/-1 ms.

## Runtime and Assets

- Runtime `in_process`; NumPy and the optional audio loader only.
- No checkpoint, binary, or node-local environment is required.

## Source and References

- Local implementation: `sure_eval.evaluation.tasks.rir.metrics`.
- Protocol follows the Rec-BIR late-window IACC computation.

## Limitations

Undefined correlation (silent or constant channels, or an empty selected
window) fails evaluation rather than receiving zero error.

- Requires exactly two channels.
- Broadband correlation does not use microphone spacing or
  frequency-dependent weighting.
