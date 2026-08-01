# scoring/vad_detection_duration

Scores VAD `speech_segments` as time-duration detection.

The node computes corpus-level duration totals:

- `TP_sec = duration(reference intersection prediction)`
- `FP_sec = duration(prediction minus reference)`
- `FN_sec = duration(reference minus prediction)`

It reports `f1`, `p_fa`, `p_miss`, and `dcf_nist` as primary metrics, with
`precision`, `recall`, `false_alarm_sec`, `miss_sec`, and scored durations in
the report details. `dcf_nist` is `0.25 * p_fa + 0.75 * p_miss`.
