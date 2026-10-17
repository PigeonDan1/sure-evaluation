# RIR IACC MAE

## Purpose

Reports binaural interaural cross-correlation coefficient error.

## Contract and computation

Input is `binaural_rir_samples_jsonl` with exactly two channels; output is a
lower-is-better coefficient error. Version `v1` searches normalized absolute
cross-correlation over the plus-or-minus 1 ms lag window and compares predicted
and reference IACC values.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. It is not
defined for mono or three-channel input and does not use microphone spacing.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
