# RIR NPM

## Purpose

Reports scale- and lag-invariant normalized projection misalignment (NPM) for
predicted and reference RIRs. It does not measure room-acoustic decay.

## Task Scenarios

The `RIR estimation` monaural and multichannel base routes select
`scoring/rir_npm`. It is part of the default NPM + RT60 bundle and can run
as an atomic metric.

## Input

The `rir_samples_jsonl` row contains `sample_id`, `prediction_rir`, and
`reference_rir`; alignment is by `sample_id`. Prediction and reference must
have the same channel count and sample rate. NumPy arrays require a row-level
`sample_rate`.

## Output

The `rir_npm_score` result and report trace include a lower-is-better score
in dB. Each `per_sample` entry includes reference and prediction energies,
the selected lag in samples, and the squared projection cosine. The metric
score is the equal-weight mean of sample scores.

## Versioned Computation

Node `scoring/rir_npm`, version `v1`, searches one common lag across channels
within +/-256 samples. It maximizes the absolute summed cross-correlation,
then reports `10 * log10(max(1 - cosine_sq, 1e-20))`. The reference and
prediction energies normalize `cosine_sq`, making the result scale invariant.

## Runtime and Assets

The node runs in-process with NumPy and the package's optional audio loader.
It has no checkpoint, binary, or node-local environment.

## Source and References

Local compatibility implementation of the Rec-BIR lag-invariant NPM protocol
in `tasks/rir/metrics.py`.

## Limitations

Inputs must already share channel order and time origin. The lag search is
limited to +/-256 samples; zero-energy inputs are rejected.
