# validation/vad_contract

Strictly validates VAD reference and prediction JSONL before scoring.

The node requires reference rows with `key`, `duration`, and `speech_segments`.
Prediction rows must use `key` plus any of `speech_segments`, `frame_scores`,
and `audio_duration`. Score aliases such as `scores`, `probs`, and
`speech_probabilities` are rejected so AUC routes only consume the documented
`frame_scores` contract.

Missing prediction fields are preserved as missing. A row without
`speech_segments` skips duration-detection metrics, and a row without
`frame_scores` skips `auc_roc`; neither case is coerced into an empty prediction.
