# RT60 MAE

## Purpose

Reports absolute reverberation-time error for predicted and reference RIRs.
It does not estimate a room-independent quality score.

## Task Scenarios

The `RIR estimation` monaural and multichannel base routes select
`scoring/rt60_mae`. It is part of the default acoustic bundle and can run
as an atomic metric.

## Input

The `rir_samples_jsonl` row contains `sample_id`, `prediction_rir`, and
`reference_rir`; alignment is by `sample_id`. Both files need matching
sample rates and channel counts. NumPy arrays require `sample_rate`.

## Output

The `rt60_mae_score` trace has a lower-is-better score in seconds. Each
`per_sample` entry records predicted and reference RT60, their absolute
error, and per-channel values. The metric score averages sample errors
equally, after averaging all channel errors within each sample.

## Versioned Computation

Node `scoring/rt60_mae`, version `v1`, starts each channel 2.5 ms before its
strongest impulse, then fits the Schroeder energy decay from -5 to -35 dB.
If that interval is unavailable, it tries -5 to -25 dB. The fit slope is
extrapolated to 60 dB; predicted and reference RT60 values are compared per
channel before aggregation.

## Runtime and Assets

The node runs in-process with NumPy and the optional audio loader. It has no
checkpoint, binary, or node-local environment.

## Source and References

Local SURE RIR implementation in `tasks/rir/metrics.py`.

## Limitations

Every channel must have a defined metric estimate. An unscorable channel
fails the evaluation with its sample and channel identified; it is never
silently omitted from the mean.

If any channel lacks a valid decay interval, scoring fails for that sample.
The estimator uses the recorded RIR decay and does not infer an external
noise floor or a different time origin.
