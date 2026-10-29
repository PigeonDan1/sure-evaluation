# RIR ILD MAE

## Purpose

Reports channel-0-relative interchannel level-difference (ILD) error. It
does not measure absolute RIR level.

## Task Scenarios

The `RIR estimation` two-channel and larger multichannel spatial routes
select `scoring/rir_ild_mae`, atomically or in a spatial bundle.

## Input

The `multichannel_rir_samples_jsonl` row contains `sample_id`,
`prediction_rir`, and `reference_rir`; alignment is by `sample_id`. Both
files need at least two channels in the same order and at the same sample
rate. NumPy arrays require `sample_rate`.

## Output

The `rir_ild_mae_score` trace reports lower-is-better error in dB. Each
`per_sample` entry includes `pair_count` and each channel pair's mean
predicted ILD, mean reference ILD, valid bin count, absolute error sum, and
mean absolute spectral error. Pair errors are averaged per sample, then
sample scores are averaged equally.

## Versioned Computation

Node `scoring/rir_ild_mae`, version `v1`, computes an FFT for each RIR
channel. At bins with nonzero prediction and reference magnitudes it uses
`20*log10(|ch0|/|chN|)` and averages the absolute binwise difference.

## Runtime and Assets

The node runs in-process with NumPy and the optional audio loader. It has no
checkpoint, binary, or node-local environment.

## Source and References

Local SURE RIR implementation in `tasks/rir/metrics.py`.

## Limitations

Mono input is unsupported. The scorer does not apply perceptual frequency
weighting or microphone-spacing correction; bins with zero magnitude in
either RIR are excluded. A pair with no valid bins is rejected.
