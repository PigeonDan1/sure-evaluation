# RIR DRR MAE

## Purpose

Reports direct-to-reverberant ratio (DRR) error for predicted and reference
RIRs. It does not estimate speech enhancement quality.

## Task Scenarios

The `RIR estimation` monaural and multichannel acoustic routes select
`scoring/rir_drr_mae`, either atomically or in the shared base bundle.

## Input

The `rir_samples_jsonl` row contains `sample_id`, `prediction_rir`, and
`reference_rir`; alignment is by `sample_id`. Prediction and reference
need matching sample rates and channel counts. NumPy arrays require
`sample_rate`.

## Output

The `rir_drr_mae_score` trace reports lower-is-better dB error. Each
`per_sample` entry records predicted DRR, reference DRR, and absolute error
per channel. The metric averages all channel errors per sample,
then averages samples equally.

## Versioned Computation

Node `scoring/rir_drr_mae`, version `v1`, starts each channel 2.5 ms before
its strongest impulse. It divides energy from the strongest impulse through
the following 2.5 ms by later energy, converts the ratio with `10*log10`,
and compares prediction with reference.

## Runtime and Assets

The node runs in-process with NumPy and the optional audio loader. It has no
checkpoint, binary, or node-local environment.

## Source and References

Local SURE RIR implementation in `tasks/rir/metrics.py`.

## Limitations

Every channel must have a defined metric estimate. An unscorable channel
fails the evaluation with its sample and channel identified; it is never
silently omitted from the mean.

The direct-arrival window follows the strongest sample, not microphone
geometry metadata. A missing or zero early/late energy pair fails scoring.
