# Nemotron-3.5-ASR-Streaming-0.6B Transcription Node

## Purpose

`transcription/nemotron_asr_streaming_06b` converts audio into transcript text
with the Hugging Face transformers model
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).

The node supports both offline (whole-file) inference and streaming (chunked)
inference for low-latency scenarios. It is an alternative transcription node
for semantic CER/WER routes.

The node does not calculate CER or WER. Scoring is performed by the task route
after transcription and normalization.

## Task Scenarios

- TTS English semantic WER alternative route:
  `tts.en.wer.nemotron_asr_streaming_06b_v1.whisper_norm_english_v1.wenet_wer_v1`
- TTS English semantic CER alternative route:
  `tts.en.cer.nemotron_asr_streaming_06b_v1.whisper_norm_english_v1.wenet_cer_v1`

## Input

- Schema: `audio_path`.
- Required role: generated/prediction audio path.
- Languages: `en`, `es`, `fr`, `de`, `it`, `pt`, etc. (see
  [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
  for the full list of supported language locales).
- Audio input mode: path.

## Output

- Schema: `transcript_text`.
- Output is one transcript string per input audio sample.
- Trace records `audio_frontend_policy=runtime_managed`, the resolved model
  checkpoint, backend, and streaming configuration.

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- GPU: optional (`--device` overrides).
- Required model env var: `NEMOTRON_ASR_STREAMING_06B_CHECKPOINT`.
- Verify imports: `torch`, `transformers`.

## Node Semantics

### Offline mode (default)

The whole audio file is loaded, resampled to the model's expected sample rate,
and passed to `model.generate()` in one shot.

### Streaming mode (`--stream`)

Audio is processed in non-overlapping chunks of `--chunk-s` seconds. Each chunk
is encoded and decoded incrementally using the model's streaming interface, and
partial transcripts are concatenated for the final output.

## CLI

```bash
python -m sure_eval.evaluation.nodes.transcription.nemotron_asr_streaming_06b.node \
  --audio-path /path/to/audio.wav \
  --language en-US \
  --device cuda \
  --json
```

Batch mode is supported via `--input-jsonl <in.jsonl>` and produces one JSONL
line per input (same order).

## Source and References

- Model: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- Transformers docs: https://huggingface.co/docs/transformers/en/model_doc/nemotron3_5_asr
- Open ASR Leaderboard (alignment reference): https://huggingface.co/open-asr-leaderboard

## Security and Compliance

- The model weights are downloaded from Hugging Face at first run unless
  `NEMOTRON_ASR_STREAMING_06B_CHECKPOINT` points to a local path.
- Verify the model's license before using it in production.
