# VC Task Route

This route evaluates VC metrics through task-level pipeline nodes.

Semantic content preservation uses the shared ASR scoring path:

1. Transcribe `converted_audio`.
2. Use `reference_text` when it is present.
3. If `reference_text` is missing, transcribe `reference_audio` and use that transcript as the reference.
4. Write reference and hypothesis text into temporary key-text files.
5. Call `sure_eval.evaluation.tasks.asr.pipeline.evaluate_asr_files`.

Semantic routes:

| Language | Canonical metric | Execution selector | Transcription node | Normalization node | Downstream ASR metric |
| --- | --- | --- | --- | --- | --- |
| `zh` | `cer` | `vc_cer` | `transcription/paraformer_zh` | `normalization/punctuation_strip_norm` | `cer` |
| `en` | `wer` | `vc_wer` | `transcription/whisper_large_v3` | `normalization/whisper_norm` | `wer` |

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
canonical result keys; the legacy `VCMetricPipeline` wrapper still exposes
`sim/...` and aggregate `sim` for compatibility.

Mandarin semantic CER uses punctuation-only normalization before WeNet CER and
does not use AISpeech number text normalization by default. `normalization/wetext_norm`
can be selected explicitly with `evaluate_vc_samples(..., semantic_normalizer="wetext:zh_tn")`.
