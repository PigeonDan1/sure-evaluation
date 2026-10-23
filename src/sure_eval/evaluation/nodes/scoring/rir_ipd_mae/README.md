# RIR IPD MAE

## Purpose

Reports channel-0-relative interchannel phase-difference (IPD) error. It
does not perform phase unwrapping across frequency.

## Task Scenarios

The `RIR estimation` two-channel and larger multichannel spatial routes
select `scoring/rir_ipd_mae`, atomically or in a spatial bundle.

## Input

The `multichannel_rir_samples_jsonl` row contains `sample_id`,
`prediction_rir`, and `reference_rir`; alignment is by `sample_id`. Both
files need at least two channels in the same order and at the same sample
rate. NumPy arrays require `sample_rate`.

## Output

The `rir_ipd_mae_score` trace reports lower-is-better error in radians.
Each `per_sample` entry includes `pair_count` and each pair's mean
predicted IPD, mean reference IPD, valid bin count, absolute circular error
sum, and mean circular spectral error. Pair errors are averaged per sample,
then sample scores are averaged equally.

## Versioned Computation

Node `scoring/rir_ipd_mae`, version `v1`, computes the FFT phase of
`ch0/chN` at bins with nonzero prediction and reference magnitudes.
It wraps the prediction-reference phase difference into [-pi, pi] and
averages its absolute value over valid bins.

## Runtime and Assets

The node runs in-process with NumPy and the optional audio loader. It has no
checkpoint, binary, or node-local environment.

## Source and References

Local SURE RIR implementation in `tasks/rir/metrics.py`.

## Limitations

Mono input is unsupported. The scorer does not unwrap phase or apply
frequency weighting; bins with zero magnitude in either RIR are excluded.
A pair with no valid bins is rejected.
