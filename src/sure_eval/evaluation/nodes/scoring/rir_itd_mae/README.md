# RIR ITD MAE

## Purpose

Reports binaural interaural time-difference error.

## Contract and computation

Input is `binaural_rir_samples_jsonl` with exactly two channels; output is a
lower-is-better mean in seconds. Version `v1` selects the peak normalized
cross-correlation lag within plus-or-minus 1 ms and compares predicted and
reference ITD values.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. It is not
defined for mono or three-channel input and reports broadband lag only.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
