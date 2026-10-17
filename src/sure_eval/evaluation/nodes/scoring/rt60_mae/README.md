# RT60 MAE

## Purpose

Reports absolute reverberation-time error for predicted/reference RIR pairs.

## Contract and computation

Input is `rir_samples_jsonl`; output is a lower-is-better mean in seconds.
Version `v1` fits the -5 to -35 dB Schroeder decay interval per channel,
averages valid channel RT60 values, and computes absolute prediction/reference
error with equal-weight sample aggregation.

## Runtime and limitations

The node is in-process with no checkpoint or node-local environment. RIRs must
contain a usable decay interval; no onset alignment or decay-window search is
performed outside the declared interval.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
