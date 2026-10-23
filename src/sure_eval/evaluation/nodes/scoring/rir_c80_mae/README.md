# RIR C80 MAE

## Purpose

Reports 80 ms clarity error for predicted and reference RIRs. It does not
estimate intelligibility directly.

## Task Scenarios

The `RIR estimation` monaural and multichannel acoustic routes select
`scoring/rir_c80_mae`, atomically or in the shared base bundle.

## Input

The `rir_samples_jsonl` row contains `sample_id`, `prediction_rir`, and
`reference_rir`; alignment is by `sample_id`. Prediction and reference
need matching sample rates and channel counts. NumPy arrays require
`sample_rate`.

## Output

The `rir_c80_mae_score` trace reports lower-is-better dB error. Each
`per_sample` entry records predicted clarity, reference clarity, and
absolute error per valid channel. The metric averages channels within
each sample and then averages samples equally.

## Versioned Computation

Node `scoring/rir_c80_mae`, version `v1`, starts each channel 2.5 ms before
its strongest impulse. It compares `10*log10(early_energy/late_energy)`
for prediction and reference, using an 80 ms early window beginning at the
strongest impulse.

## Runtime and Assets

The node runs in-process with NumPy and the optional audio loader. It has no
checkpoint, binary, or node-local environment.

## Source and References

Local SURE RIR implementation in `tasks/rir/metrics.py`.

## Limitations

Prediction and reference must describe the same source and room. A missing
or zero early/late energy pair is excluded; scoring fails if no channel
remains valid.
