# Paraformer-ZH Transcription Node

`transcription/paraformer_zh` converts Mandarin-family speech into text before semantic error-rate scoring.

It is used by the TTS and VC task routes before they call the canonical ASR CER pipeline.
The node accepts an injected `runner` for tests and smoke checks; without one, it lazily uses
`sure_eval.evaluation.nodes.transcription.ParaformerZHTranscriber`.

The node does not calculate CER. CER is calculated by the ASR task route after transcription.
