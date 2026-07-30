# VAD Task

Voice activity detection is evaluated as a seconds-timebase task with aligned
reference and prediction JSONL rows.

Required roles:

- `reference_jsonl`
- `sample_output`

The reference row contract is:

```json
{"key": "utt1", "duration": 2.465, "speech_segments": [{"start": 0.3, "end": 0.838}]}
```

The prediction row contract is:

```json
{"key": "utt1", "speech_segments": [{"start": 0.3, "end": 0.9}], "frame_scores": [{"start": 0.0, "end": 0.01, "score": 0.03}], "audio_duration": 2.465}
```

Primary metrics:

- `f1`: duration-overlap speech detection F1, higher is better.
- `p_fa`: false alarm seconds divided by non-speech scored seconds, lower is better.
- `p_miss`: missed speech seconds divided by speech scored seconds, lower is better.
- `dcf_nist`: `0.25 * p_fa + 0.75 * p_miss`, lower is better.
- `auc_roc`: pooled frame-score ROC AUC, higher is better.

Default route:

`vad.any.f1.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1`

The evaluator does not run a VAD model, read audio, or resample audio. It only
scores already generated prediction JSONL on the reference seconds timebase.
