# validation/vad_contract

Strictly validates VAD reference and prediction JSONL before scoring.

The node requires reference rows with `key`, `duration`, and `speech_segments`.
Prediction rows must use `key` plus `speech_segments` for duration routes or
`frame_scores` for AUC routes. Score aliases such as `scores`, `probs`, and
`speech_probabilities` are rejected so AUC routes only consume the documented
`frame_scores` contract. Prediction-side duration metadata such as
`audio_duration` is not accepted; reference `duration` defines the scored region.

The selected route declares the prediction fields it consumes, and missing
selected fields fail validation. All intervals must satisfy
`0 <= start < end <= duration`, and overlapping intervals are rejected within a
row.
