# ASR Evaluation Task

ASR is routed by language and metric before selecting a concrete pipeline.
The canonical ASR entry is:

```python
sure_eval.evaluation.tasks.asr.pipeline.evaluate_asr_files
```

The first migrated pipelines are:

- `zh + cer`: `aispeech_norm` then `wenet_cer`
- `en + wer`: `whisper_norm` with profile `english`, then `wenet_wer`
- `cs + mer`: `aispeech_norm` with profile `cs`, then `wenet_mer`

The task layer owns route selection. The node layer owns executable
implementations and their isolated environment metadata.

`SUREEvaluator._eval_asr` and `SUREEvaluator._eval_asr_codeswitch` are kept as
legacy references for regression checks while non-ASR tasks are migrated.
The legacy English route remains available as `asr.en.wer.aispeech_norm.wenet_wer`
by calling `evaluate_asr_files(..., normalizer="aispeech")`.

`normalization/wetext_norm` is an optional tool node. It is selected only by
explicit arguments such as `normalizer="wetext:zh_tn"` or
`normalizer="wetext:en_itn"`; it is not part of any default ASR route.

`scoring/sctk_sclite` is an optional binary-backed scorer wrapping NIST SCTK
`sclite`. It is selected only by explicit arguments such as
`scorer="sctk_sclite"`; default ASR routes continue to use `wenet_wer` /
`wenet_cer`.
