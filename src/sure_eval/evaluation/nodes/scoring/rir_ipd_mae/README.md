# RIR IPD MAE

## Purpose

Reports binaural interaural phase-difference error.

## Contract and computation

Input is `binaural_rir_samples_jsonl` with exactly two channels; output is a
lower-is-better mean in radians. Version `v1` compares circular left/right FFT
phase differences over valid frequency bins.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. It uses
the declared L/R order and rejects non-binaural input; no phase unwrapping or
frequency weighting is applied.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
