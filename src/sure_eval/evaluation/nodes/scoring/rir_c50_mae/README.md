# RIR C50 MAE

## Purpose

Reports 50 ms clarity error for predicted/reference RIR pairs.

## Contract and computation

Input is `rir_samples_jsonl`; output is a lower-is-better mean in dB. Version
`v1` compares early-to-late energy ratios using a 50 ms window after the
strongest impulse.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. It assumes
the two RIRs share source, time origin, and sample rate.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
