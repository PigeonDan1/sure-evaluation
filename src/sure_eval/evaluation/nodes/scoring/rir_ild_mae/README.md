# RIR ILD MAE

## Purpose

Reports binaural interaural level-difference error.

## Contract and computation

Input is `binaural_rir_samples_jsonl` with exactly two channels; output is a
lower-is-better mean in dB. Version `v1` compares the mean absolute difference
between left/right FFT magnitude ratios over valid frequency bins.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. It uses
the declared L/R order and rejects non-binaural input; no frequency weighting
or microphone spacing is applied.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
