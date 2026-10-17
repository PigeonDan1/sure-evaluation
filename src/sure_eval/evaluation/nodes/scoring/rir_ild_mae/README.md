# RIR ILD MAE

## Purpose

Reports binaural interaural level-difference error.

## Contract and computation

Input is `multichannel_rir_samples_jsonl` with at least two channels; output is
a lower-is-better mean in dB. Version `v1` compares the mean absolute
difference between each extra channel and channel 0 FFT magnitude ratios over
valid frequency bins.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. It uses
channel 0 as the reference channel; no frequency weighting or microphone
spacing is applied.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
