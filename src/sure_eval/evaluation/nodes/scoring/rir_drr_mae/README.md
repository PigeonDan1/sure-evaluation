# RIR DRR MAE

## Purpose

Reports direct-to-reverberant ratio error for predicted/reference RIR pairs.

## Contract and computation

Input is `rir_samples_jsonl`; output is a lower-is-better mean in dB. Version
`v1` uses a 2.5 ms early window after the strongest impulse and compares the
10-log early/late energy ratios.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. The
direct-arrival window and time origin are not inferred from external metadata.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
