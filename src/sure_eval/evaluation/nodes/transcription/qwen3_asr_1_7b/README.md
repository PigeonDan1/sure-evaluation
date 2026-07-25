# Qwen3-ASR-1.7B Transcription Node

`transcription/qwen3_asr_1_7b` converts speech into text before TTS semantic error-rate scoring.

The first supported routes are TTS `zh` CER and TTS `en` WER alternatives selected by exact
`pipeline_id`. Existing Paraformer-ZH and Whisper-large-v3 defaults stay unchanged.

The node uses the official `qwen-asr` runtime with model `Qwen/Qwen3-ASR-1.7B`. Audio is passed
as a file path; the runtime manages decode, mono conversion, 16 kHz resampling, and float32
normalization internally. That preprocessing is recorded in trace details as
`audio_frontend_policy=runtime_managed`, not represented as a separate frontend node.

The node does not calculate CER or WER. Scoring is calculated by the ASR task route after
transcription and normalization.
