# VC Task Route

This route evaluates VC metrics through task-level pipeline nodes.

Semantic content preservation uses the shared ASR scoring path:

1. Transcribe `converted_audio`.
2. Use `reference_text` when it is present.
3. If `reference_text` is missing, transcribe `reference_audio` and use that transcript as the reference.
4. Write reference and hypothesis text into temporary key-text files.
5. Call `sure_eval.evaluation.tasks.asr.pipeline.evaluate_asr_files`.

Semantic routes:

| Language | Metric | Transcription node | Normalization node | Downstream ASR metric |
| --- | --- | --- | --- | --- |
| `zh` / `cmn` / `yue` | `vc_cer` | `transcription/paraformer_zh` | `normalization/aispeech_norm` | `cer` |
| `en` | `vc_wer` | `transcription/whisper_large_v3` | `normalization/whisper_norm` | `wer` |

Speaker similarity and MOS routes:

| Metric family | Metric names | Scoring node |
| --- | --- | --- |
| speaker similarity | `sim/wavlm-large` | `scoring/wavlm_large_sim` |
| speaker similarity | `sim/ecapa-tdnn` | `scoring/ecapa_tdnn_sim` |
| speaker similarity | `sim/eres2net` | `scoring/eres2net_sim` |
| MOS | `dnsmos` | `scoring/dnsmos` |
| MOS | `wv-mos` | `scoring/wv_mos` |
| MOS | `utmos` | `scoring/utmos` |

`metrics=None` defaults to the language-matched semantic metric. Speaker and MOS routes require explicit
metric selection plus injected providers. The report-level `sim` value is an aggregate over selected
named speaker backends, not a standalone scoring node.

`normalization/wetext_norm` can be selected explicitly with
`evaluate_vc_samples(..., semantic_normalizer="wetext:zh_tn")`. Defaults remain
unchanged.
