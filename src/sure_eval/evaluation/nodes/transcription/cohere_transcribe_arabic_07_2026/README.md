# Cohere Transcribe Arabic 07-2026

## Purpose

`transcription/cohere_transcribe_arabic_07_2026` converts Arabic TTS audio into
transcript text. It does not normalize text or compute CER.

## Task Scenarios

- Default TTS Arabic semantic CER route:
  `tts.ar.cer.cohere_transcribe_arabic_07_2026_v1.nemo_norm_ar_tn_v1.wenet_cer_v1`.

## Input

- Schema: `audio_path`.
- Required role: `prediction_audio`.
- Language: `ar`.

## Output

- Schema: `transcript_text`.
- One transcript per input audio path.
- Trace records the model revision, runtime-managed frontend, language, and
  inference settings.

## Versioned Computation

- Node: `transcription/cohere_transcribe_arabic_07_2026@v1`.
- Model: `CohereLabs/cohere-transcribe-arabic-07-2026`.
- Revision: `c3e911b42149bf7a1e53d5cef9878aee87515a23`.
- Internal stages: audio decode/mono/resample, batching, ASR inference, and
  text extraction.
- The official Transformers processor/generate/decode flow runs with Arabic
  language selection, 256 generated tokens, sampling disabled, and beam size
  one.

## Runtime and Assets

- Node-local `uv` environment with Python 3.11.
- Transformers 5.4.0 and PyTorch 2.8.0.
- GPU is optional; CUDA uses bfloat16 when supported and otherwise falls back
  to float16. CPU uses float32.
- Local checkpoint directory:
  `checkpoints/cohere-transcribe-arabic-07-2026`.
- Override variable: `COHERE_TRANSCRIBE_ARABIC_07_2026_CHECKPOINT`.

```bash
sure-eval env setup --node transcription/cohere_transcribe_arabic_07_2026
```

Checkpoint files and node virtual environments are local assets and must not be
committed.

## Source and References

- Model card: https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026
- License: Apache-2.0.

## Limitations

- The model requires an explicit language and does not perform language
  detection.
- It does not provide timestamps or speaker diarization.
- This route accepts utterances up to the model's 35-second feature limit; it
  does not add a separate long-form chunker.
- Low-volume noise or silence can produce hallucinated text; VAD is not part of
  this versioned route.
