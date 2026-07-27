# Qwen3-ASR-1.7B Transcription Node

## Purpose

`transcription/qwen3_asr_1_7b` converts speech into transcript text with the
official `qwen-asr` runtime and model `Qwen/Qwen3-ASR-1.7B`. It is an
alternative transcription node for semantic CER/WER routes.

The node does not calculate CER or WER. Scoring is performed by the task route
after transcription and normalization.

## Task Scenarios

- TTS Mandarin semantic CER alternative route:
  `tts.zh.cer.qwen3_asr_1_7b_v1.punctuation_strip_norm_v1.wenet_cer_v1`.
- TTS English semantic WER alternative route:
  `tts.en.wer.qwen3_asr_1_7b_v1.whisper_norm_english_v1.wenet_wer_v1`.

Existing Paraformer-ZH and Whisper-large-v3 default routes remain unchanged.

## Input

- Schema: `audio_path`.
- Required role: generated/prediction audio path.
- Languages: `zh`, `en`.
- Audio input mode: path.

## Output

- Schema: `transcript_text`.
- Output is one transcript string per input audio sample.
- Trace records `audio_frontend_policy=runtime_managed` and runtime
  normalization details.

## Versioned Computation

- Node id: `transcription/qwen3_asr_1_7b`.
- Version: `v1`.
- Package: `qwen-asr==0.0.6`.
- Model id: `Qwen/Qwen3-ASR-1.7B`.
- Backend: `transformers`.
- Internal stages:
  - `runtime_managed_audio_frontend`
  - `asr_inference`
  - `text_extraction`

Audio frontend policy:

- Audio is passed as a file path.
- The qwen-asr runtime manages decode, mono conversion, 16 kHz resampling, and
  float32 normalization internally.
- SURE-EVAL does not expose a separate downsampling/frontend node for this
  route.

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- GPU: optional.
- Required model env var: `QWEN3_ASR_1_7B_CHECKPOINT`.
- Verify imports: `qwen_asr`, `torch`, `transformers`.
- Inference defaults:
  - dtype: bfloat16 on CUDA, else float32.
  - device map: `cuda:0`.
  - max inference batch size: `32`.
  - max new tokens: `256`.
  - timestamps disabled.

Setup:

```bash
sure-eval env setup --node transcription/qwen3_asr_1_7b
```

## Source and References

- Qwen3-ASR repository: https://github.com/QwenLM/Qwen3-ASR
- Hugging Face model card: https://huggingface.co/Qwen/Qwen3-ASR-1.7B

## Limitations

- The runtime-managed frontend is part of this node version; adding an
  external frontend node would change the pipeline id.
- The model checkpoint is a runtime asset and must not be committed.
