# ASR Evaluation Task

ASR reports canonical `cer`, `wer`, and `mer` metrics. Concrete pipelines are
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

### `wer`

- `asr.en.wer.whisper_norm_english_v1.wenet_wer_v1`:
  `normalization/whisper_norm` -> `scoring/wenet_wer`
- `asr.en.wer.aispeech_norm_en_v1.wenet_wer_v1`:
  `normalization/aispeech_norm` -> `scoring/wenet_wer`
- `asr.en.wer.canonical_itn_en_v1.token_mer_v1`:
  `normalization/canonical_itn` -> `scoring/token_mer`

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

`scoring/sctk_sclite` is an optional binary-backed scorer wrapping NIST SCTK
`sclite`; default ASR pipelines continue to use WeNet-compatible scorers.
