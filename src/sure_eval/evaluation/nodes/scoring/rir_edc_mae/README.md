# RIR EDC MAE

## Purpose

Reports mean absolute error between predicted and reference energy decay
curves (EDC) in decibels. The report also records EDC RMSE and correlation.

## Task Scenarios

The `RIR estimation` acoustic routes select `scoring/rir_edc_mae`, atomically
or in the default acoustic bundle.

## Input

The `rir_samples_jsonl` row contains `sample_id`, `prediction_rir`, and
`reference_rir`. Prediction and reference must have matching sample rates and
channel counts; NumPy arrays require `sample_rate`.

## Output

The `rir_edc_mae_score` is lower-is-better and measured in dB. Per-channel
rows include MAE, RMSE, correlation, and the number of valid decay samples.
Only reference EDC points at or above -40 dB are included.

## Versioned Computation

Node `scoring/rir_edc_mae`, version `v1`, computes the Schroeder reverse
cumulative energy curve, normalizes it to 0 dB at the first sample, and
compares aligned curves over the valid -40 dB range. Channel errors are
averaged per sample; sample MAE values are averaged equally.

## Runtime and Assets

The node runs in-process with NumPy and the optional audio loader. It has no
checkpoint, binary, or node-local environment.

## Source and References

Local SURE RIR implementation following the Rec-BIR EDC protocol in
`tasks/rir/metrics.py` and the EDC analysis described in MEAN-RIR.

## Limitations

Every channel must have a defined metric estimate. An unscorable channel
fails the evaluation with its sample and channel identified; it is never
silently omitted from the mean.

The curves share the input time origin and are truncated to the shorter RIR.
Samples with fewer than four valid EDC points are rejected.
An undefined optional correlation is reported as JSON `null`; it does not
remove a channel with a defined MAE, including a silent prediction against a
valid reference decay.
