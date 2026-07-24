# TTS Task Route

This route evaluates TTS metrics through task-level pipeline nodes.

Semantic intelligibility uses the shared ASR scoring path:

1. Transcribe `prediction_audio` with a language-specific transcription node.
2. Write the transcript and `reference_text` into temporary key-text files.
3. Call `sure_eval.evaluation.tasks.asr.pipeline.evaluate_asr_files`.
4. Return a trace containing transcription, ASR normalization, and ASR scoring nodes.

Semantic routes:

| Language | Canonical metric | Execution selector | Transcription node | Normalization node | Downstream ASR metric |
| --- | --- | --- | --- | --- | --- |
| `zh` / `cmn` / `yue` | `cer` | `tts_cer` | `transcription/paraformer_zh` | `normalization/punctuation_strip_norm` | `cer` |
| `en` | `wer` | `tts_wer` | `transcription/whisper_large_v3` | `normalization/whisper_norm` | `wer` |

Speaker similarity and MOS routes:

| Canonical metric | Method or selector | Scoring node |
| --- | --- | --- |
| `spk_sim` | `spk_sim` / `sim/wavlm-large` | `scoring/wavlm_large_sim` |
| `spk_sim` | `sim/ecapa-tdnn` | `scoring/ecapa_tdnn_sim` |
| `spk_sim` | `sim/eres2net` | `scoring/eres2net_sim` |
| `dnsmos` | `dnsmos` | `scoring/dnsmos` |
| `wv_mos` | `wv_mos` / `wv-mos` | `scoring/wv_mos` |
| `utmos` | `utmos` | `scoring/utmos` |

`metrics=None` defaults to the language-matched semantic metric. Speaker and MOS routes require explicit
metric selection plus injected providers. New `EvaluationReport` payloads use
canonical result keys; the legacy `TTSMetricPipeline` wrapper still exposes
`sim/...` and aggregate `sim` for compatibility.

`normalization/wetext_norm` can be selected explicitly with
`evaluate_tts_samples(..., semantic_normalizer="wetext:zh_tn")`. Mandarin defaults
use `normalization/punctuation_strip_norm`, not `normalization/aispeech_norm`, so
numbers, case, and non-punctuation text are preserved before WeNet CER scoring.
