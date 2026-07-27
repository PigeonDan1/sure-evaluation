# Whisper-Large-V3 Transcription Node

## Purpose

`transcription/whisper_large_v3` converts English speech into transcript text
before semantic WER scoring. It is an ASR transcription node, not a scorer.

The node does not calculate WER. WER is calculated by the task route after
transcription and Whisper text normalization.

## Task Scenarios

- TTS English semantic WER default route:
  `tts.en.wer.whisper_large_v3_v1.whisper_norm_english_v1.wenet_wer_v1`.
- VC English semantic WER default route.

## Input

- Schema: `audio_path`.
- Required role: generated/prediction audio path.
- Languages: `en`.

## Output

- Schema: `transcript_text`.
- Output is one transcript string per input audio sample.
- Trace records model runner, language, and transcription stage details.

## Versioned Computation

- Node id: `transcription/whisper_large_v3`.
- Version: `v1`.
- Default runner:
  `sure_eval.evaluation.nodes.transcription.WhisperLargeV3Transcriber`.
- Internal stages:
  - `audio_decode`
  - `asr_inference`
  - `text_extraction`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- GPU: optional.
- Required model env var: `WHISPER_LARGE_V3_CHECKPOINT`.
- Model id: `openai/whisper-large-v3`.
- Verify imports: `transformers`, `torch`.

Setup:

```bash
sure-eval env setup --node transcription/whisper_large_v3
```

## Source and References

- OpenAI Whisper repository: https://github.com/openai/whisper
- Hugging Face model card: https://huggingface.co/openai/whisper-large-v3

## Limitations

- The model checkpoint is a runtime asset and must not be committed.
- The node accepts an injected runner for tests and smoke checks; production
  scoring should use the prepared node-local runtime.
