# RIR Correlation

## Purpose

Reports channel-wise waveform correlation for predicted/reference RIR pairs.

## Contract and computation

Input is `rir_samples_jsonl`; output is a higher-is-better arithmetic mean and
per-channel values. Version `v1` centers each channel, computes normalized
dot products, averages channels, then averages samples equally.

## Runtime and limitations

The node is in-process and has no checkpoint or node-local environment. It
does not compensate for scale, delay, or channel permutation.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
