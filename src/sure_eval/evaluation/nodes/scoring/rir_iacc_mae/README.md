# RIR IACC MAE

## Purpose

Reports interaural cross-correlation coefficient (IACC) error for a binaural
RIR pair. It does not describe arrays with more than two channels.

## Task Scenarios

The `RIR estimation` two-channel spatial route selects
`scoring/rir_iacc_mae`, atomically or in the binaural bundle. It is not in
the multichannel bundle for larger arrays.

## Input

The `binaural_rir_samples_jsonl` row contains `sample_id`, `prediction_rir`,
and `reference_rir`; alignment is by `sample_id`. Both files must have
exactly two channels and matching sample rates. NumPy arrays require
`sample_rate`.

## Output

The `rir_iacc_mae_score` trace reports lower-is-better coefficient error.
Each `per_sample` entry includes predicted IACC, reference IACC, and their
absolute difference. The metric averages sample errors equally.

## Versioned Computation

Node `scoring/rir_iacc_mae`, version `v1`, takes the largest absolute
normalized cross-correlation of the two channels within +/-1 ms. It
compares that coefficient between prediction and reference.

## Runtime and Assets

The node runs in-process with NumPy and the optional audio loader. It has no
checkpoint, binary, or node-local environment.

## Source and References

Local SURE RIR implementation in `tasks/rir/metrics.py`.

## Limitations

Only exactly two channels are accepted. The search is broadband and does
not use microphone spacing or frequency-dependent weighting.
