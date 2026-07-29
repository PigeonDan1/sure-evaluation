# SA-ASR Evaluation

Speaker-attributed ASR uses a conversion bridge around a language-selected
normalization node, followed by the generic `scoring/meeteval` node. The task
reports cpWER as the main score and DER as a companion metric.

The English pipeline is
`sa_asr.en.cpwer.conversion_sa_asr_cpwer_v1.whisper_norm_english_v1.meeteval_v1`.
It uses `normalization/whisper_norm` with the `english` profile.

The Mandarin pipeline is
`sa_asr.zh.cpwer.conversion_sa_asr_cpwer_v1.gstar_norm_v1.meeteval_v1`.
It uses `normalization/gstar_norm` to preserve the G-STAR-compatible SA-ASR
text behavior used by `SUREEvaluator._eval_sa_asr`.

Both routes use `collar=0.5` for the DER companion metric and expect STM
six-field rows.

The task layer uses:

```text
src/sure_eval/evaluation/conversion/sa_asr__cpwer/
```

to convert STM to key-text before normalization and normalized key-text back to
STM before `meeteval.io.load`. This conversion is score-affecting and is part
of `computation_node_ids`.
