# ASR Evaluation Task

ASR reports public `cer`, `wer`, and `mer` metrics. Concrete pipelines are
selected by `pipeline_id` when a metric has multiple normalization or scorer
chains.

## Pipelines

### `cer`

- `asr.zh.cer.wetext_norm_zh_itn_v1.wenet_cer_v1`:
  `normalization/wetext_norm` (`zh_itn`) -> `scoring/wenet_cer`
- `asr.zh.cer.aispeech_norm_zh_v1.wenet_cer_v1`:
  `normalization/aispeech_norm` -> `scoring/wenet_cer`
- `asr.zh.cer.canonical_itn_zh_v1.token_cer_v1`:
  `normalization/canonical_itn` -> `scoring/token_cer`
- `asr.ja.cer.funasr_itn_ja_v1.wenet_cer_v1` and
  `asr.ko.cer.funasr_itn_ko_v1.wenet_cer_v1`:
  `normalization/funasr_itn` -> `scoring/wenet_cer`
- `asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1`:
  `normalization/nemo_norm` (`ar_tn`) -> `scoring/wenet_cer`

### `wer`

- `asr.en.wer.whisper_norm_english_v1.wenet_wer_v1`:
  `normalization/whisper_norm` -> `scoring/wenet_wer`
- `asr.en.wer.aispeech_norm_en_v1.wenet_wer_v1`:
  `normalization/aispeech_norm` -> `scoring/wenet_wer`
- `asr.en.wer.canonical_itn_en_v1.token_mer_v1`:
  `normalization/canonical_itn` -> `scoring/token_mer`
- `asr.<lang>.wer.funasr_itn_<lang>_v1.wenet_wer_v1` for `es`, `fr`, `de`,
  `ru`, `pt`, `vi`, `id`, and `tl`:
  `normalization/funasr_itn` -> `scoring/wenet_wer`

### `mer`

- `asr.cs.mer.aispeech_norm_cs_v1.wenet_mer_v1`:
  `normalization/aispeech_norm` -> `scoring/wenet_mer`
- `asr.cs.mer.canonical_itn_cs_v1.token_mer_v1`:
  `normalization/canonical_itn` -> `scoring/token_mer`

The script entrypoint is `sure_eval.evaluation.scripts.asr.run`; the task
executor is `sure_eval.evaluation.tasks.asr.pipeline.evaluate_asr_files`.
Route declarations live in `src/sure_eval/evaluation/tasks/asr/routes.yaml`.

`SUREEvaluator._eval_asr` and `SUREEvaluator._eval_asr_codeswitch` are kept as
legacy references for regression checks while non-ASR tasks are migrated.

Mandarin CER defaults to `normalization/wetext_norm` with profile `zh_itn`.
Other WeTextProcessing profiles remain available through lower-level task API
arguments such as `normalizer="wetext:zh_tn"` or `normalizer="wetext:en_itn"`.

Japanese/Korean CER and Spanish/French/German/Russian/Portuguese/Vietnamese/
Indonesian/Tagalog WER default to `normalization/funasr_itn`. Prepare its
optional node-local environment with
`sure-eval env setup --node normalization/funasr_itn` before scoring.

Arabic CER defaults to `normalization/nemo_norm` with profile `ar_tn`. It
converts written tokens to spoken Arabic on both sides before CER scoring.
Prepare its frozen node-local environment with
`sure-eval env setup --node normalization/nemo_norm` before scoring.

`scoring/sctk_sclite` is an optional binary-backed scorer wrapping NIST SCTK
`sclite`; default ASR pipelines continue to use WeNet-compatible scorers.
