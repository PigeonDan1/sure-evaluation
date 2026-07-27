# Whisper Normalization

## Purpose

`normalization/whisper_norm` normalizes key-text ASR files with vendored
OpenAI Whisper text normalizers before WeNet edit-distance scoring.

Only the normalizer files are vendored; this node does not depend on the full
Whisper ASR package and does not transcribe audio.

## Task Scenarios

- Default English ASR WER:
  `asr.en.wer.whisper_norm_english_v1.wenet_wer_v1`.
- English TTS semantic WER after `transcription/whisper_large_v3`.
- English VC semantic WER after `transcription/whisper_large_v3`.

## Input

- Schema: `key_text_files`.
- Row format: `<key><TAB><text>`.
- Required roles: `ref`, `hyp`.

## Output

- Schema: `key_text_files`.
- Output files preserve keys and contain Whisper-normalized text.
- Trace records selected profile, row counts, empty-after-normalization counts,
  upstream package metadata, and local vendoring status.

## Versioned Computation

- Node id: `normalization/whisper_norm`.
- Version: `v1`.
- Profiles:
  - `english`: Whisper `EnglishTextNormalizer`; default for English WER.
  - `basic`: Whisper `BasicTextNormalizer`; available for future explicit
    multilingual routes.
- Upstream package snapshot recorded as `openai-whisper==20250625`.
- Local change: `more_itertools.windowed` was replaced with a private
  triple-window helper to avoid an additional root dependency.

## Runtime and Assets

- Runtime: `in_process`.
- No optional model checkpoint or full Whisper installation required.
- Vendored files live under `normalization/whisper_norm/normalization_impl/`.

## Source and References

- OpenAI Whisper normalizers:
  https://github.com/openai/whisper/tree/main/whisper/normalizers
- Vendored license:
  `normalization/whisper_norm/normalization_impl/LICENSE.openai-whisper`

## Limitations

- The English profile is only a default for English WER routes.
- Mandarin ASR CER defaults to `normalization/wetext_norm`.
- Code-switch MER defaults to `normalization/aispeech_norm`.
