# RIR IPD MAE

## Purpose

Reports binaural interaural phase-difference error.

## Contract and computation

Input is `multichannel_rir_samples_jsonl` with at least two channels; output is
a lower-is-better mean in radians. Version `v1` compares circular FFT phase
differences between each extra channel and channel 0 over valid frequency bins.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. It uses
channel 0 as the reference channel; no phase unwrapping or frequency weighting
is applied.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
