# Whisper-Large-V3 Transcription Node

`transcription/whisper_large_v3` converts English speech into text before semantic error-rate scoring.

It is used by the TTS and VC task routes before they call the canonical ASR WER pipeline.
The node accepts an injected `runner` for tests and smoke checks; without one, it lazily uses
`sure_eval.evaluation.nodes.transcription.WhisperLargeV3Transcriber`.

The node does not calculate WER. WER is calculated by the ASR task route after transcription.
