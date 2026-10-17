# RIR NMSE

## Purpose

Reports normalized mean squared error for predicted/reference RIR pairs in the
RIR signal-error evaluation branch.

## Contract and computation

Input is `rir_samples_jsonl` with `prediction_rir`, `reference_rir`, and
`sample_id`. Output is one lower-is-better mean score plus per-sample scores.
Version `v1` computes `mean((prediction-reference)^2) / mean(reference^2)`
after common sample-rate/channel validation and equal-weight aggregation.

## Runtime and limitations

The node runs in-process with NumPy and the optional audio loader; it has no
checkpoint or node-local model environment. No peak, delay, or time-origin
alignment is applied. Mono and multi-channel pairs are accepted.

Source: SURE RIR task implementation in `tasks/rir/metrics.py`.
