# RIR ITD MAE

## Purpose

Reports binaural interaural time-difference error.

## Contract and computation

Input is `multichannel_rir_samples_jsonl` with at least two channels; output
is a lower-is-better mean in seconds. Version `v1` selects the peak normalized
cross-correlation lag within plus-or-minus 1 ms for every `chN` against `ch0`
and compares predicted and reference ITD values.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. It is not
defined for mono input and reports broadband lag pairs for each extra channel.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
