# RIR DOA MAE

## Purpose

Reports channel-pair direction-of-arrival (DOA) error derived from ITD. It
does not estimate a full spatial scene.

## Task Scenarios

The `RIR estimation` two-channel and larger multichannel spatial routes
select `scoring/rir_doa_mae`, atomically or in the spatial bundle without
IACC.

## Input

The `multichannel_rir_samples_jsonl` row contains `sample_id`,
`prediction_rir`, `reference_rir`, and `metadata.microphone_spacing_m`;
alignment is by `sample_id`. Both files need at least two ordered channels
at the same sample rate. Optional `channel_spacings_m` and
`sound_speed_mps` refine the geometry; NumPy arrays require `sample_rate`.

## Output

The `rir_doa_mae_score` trace reports lower-is-better error in degrees.
Each `per_sample` entry includes `pair_count` and each pair's predicted
and reference DOA, ITD in ms, and absolute angular error. Pair errors are
averaged per sample, then sample scores are averaged equally.

## Versioned Computation

Node `scoring/rir_doa_mae`, version `v1`, estimates broadband ITD for
each `chN` relative to `ch0` within +/-1 ms. It converts delay to angle
with `asin(clamp(delay * sound_speed / pair_spacing, -1, 1))`, using
`N * microphone_spacing_m` by default and 343 m/s unless specified.

## Runtime and Assets

The node runs in-process with NumPy and the optional audio loader. It has no
checkpoint, binary, or node-local environment.

## Source and References

Local SURE RIR implementation in `tasks/rir/metrics.py`.

## Limitations

Mono input and nonpositive microphone spacing are unsupported. The angle
assumes far-field pair geometry and cannot resolve full 3-D direction.
