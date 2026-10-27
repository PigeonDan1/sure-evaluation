# RT30 MAE

## Purpose

Reports the absolute error of the measured 30 dB reverberation decay time.

## Task Scenarios

The `RIR estimation` acoustic routes select `scoring/rt30_mae`, atomically or
in the default acoustic bundle.

## Input

The `rir_samples_jsonl` row contains `sample_id`, `prediction_rir`, and
`reference_rir`. Prediction and reference must have matching sample rates and
channel counts; NumPy arrays require `sample_rate`.

## Output

The `rt30_mae_score` is lower-is-better and measured in seconds. Per-sample
rows include predicted and reference RT30 and per-channel absolute errors.

## Versioned Computation

Node `scoring/rt30_mae`, version `v1`, computes the Schroeder energy decay and
measures the time between -5 dB and -35 dB. The metric averages channel errors
within each sample and then averages samples equally.

## Runtime and Assets

The node runs in-process with NumPy and the optional audio loader. It has no
checkpoint, binary, or node-local environment.

## Source and References

Local SURE RIR implementation following the Rec-BIR decay protocol in
`tasks/rir/metrics.py`.

## Limitations

Every channel must have a defined metric estimate. An unscorable channel
fails the evaluation with its sample and channel identified; it is never
silently omitted from the mean.

Inputs need a valid 30 dB decay interval. The estimator uses the recorded RIR
time origin and does not infer a room noise floor.
