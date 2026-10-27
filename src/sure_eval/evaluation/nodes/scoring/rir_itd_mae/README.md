# RIR ITD MAE

## Purpose

Reports channel-0-relative interchannel time-difference (ITD) error. It does
not estimate a complete source trajectory.

## Task Scenarios

The `RIR estimation` two-channel and larger multichannel spatial routes
select `scoring/rir_itd_mae`, atomically or in a spatial bundle.

## Input

The `multichannel_rir_samples_jsonl` row contains `sample_id`,
`prediction_rir`, and `reference_rir`; alignment is by `sample_id`. Both
files need at least two channels with matching order and sample rate. NumPy
arrays require `sample_rate`.

## Output

The `rir_itd_mae_score` trace reports lower-is-better error in ms. Each
`per_sample` entry includes `pair_count` and predicted ITD, reference ITD,
and absolute error for every `chN` against `ch0`. Pair errors are averaged
within each sample; sample scores are then averaged equally.

## Versioned Computation

Node `scoring/rir_itd_mae`, version `v1`, selects the largest absolute
normalized broadband cross-correlation lag within +/-1 ms for each pair.
The lag is converted to milliseconds before comparing prediction and
reference.

## Runtime and Assets

The node runs in-process with NumPy and the optional audio loader. It has no
checkpoint, binary, or node-local environment.

## Source and References

Local SURE RIR implementation in `tasks/rir/metrics.py`.

## Limitations

Undefined correlation (silent or constant channels, or an empty selected
window) fails evaluation rather than receiving zero error.

Mono input is unsupported. The +/-1 ms search bounds measurable delay, and
the output is a broadband pairwise lag rather than a frequency-specific ITD.
