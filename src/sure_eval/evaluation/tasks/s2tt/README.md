# S2TT Evaluation Task

S2TT supports three scoring backends: SacreBLEU/chrF++ as the reproducible
anchor, XCOMET-XL as the primary semantic metric, and BLEURT-20 as the
complementary semantic metric.
The canonical S2TT entry is:

```python
sure_eval.evaluation.tasks.s2tt.pipeline.evaluate_s2tt_files
```

The SacreBLEU route preserves `SUREEvaluator._eval_s2tt` behavior. S2TT
currently does not add a separate normalization node; tokenizer selection is
part of the SacreBLEU scoring backend.

- scoring: `sacrebleu`
- scoring internal stages: tokenizer selection, corpus BLEU, corpus chrF2
- result fields: `bleu`, `bleu_char`, `chrf`, `score`
- semantic scoring: `xcomet_xl`, `bleurt_20`
- semantic aggregation: segment score arithmetic mean

`SUREEvaluator._eval_s2tt` is kept as a legacy reference for regression checks.

## Input Contracts

Main-report S2TT metrics use role-addressed key-text files:

| Metric backend | Required roles | Aggregation | Notes |
| --- | --- | --- | --- |
| `scoring/xcomet_xl` | `src`, `hyp`, `ref` | segment mean | `Unbabel/XCOMET-XL`; primary semantic quality |
| `scoring/bleurt_20` | `hyp`, `ref` | segment mean | `BLEURT-20`; complementary semantic quality |
| `scoring/sacrebleu` | `hyp`, `ref` | corpus metric | BLEU, `bleu_char`, and chrF2 reproducible anchor |

All roles use `key<TAB>text` rows aligned by `key`.

For full dataset evaluation, XCOMET-XL additionally requires the dataset JSONL
to provide source text through one of these fields: `source`, `src`,
`source_text`, `transcript`, or `speech_text`.

## Metric Environments

SacreBLEU runs with the uv project in its scoring node:

```text
src/sure_eval/evaluation/nodes/scoring/sacrebleu/pyproject.toml
src/sure_eval/evaluation/nodes/scoring/sacrebleu/.cache/uv
src/sure_eval/evaluation/nodes/scoring/sacrebleu/.venv
```

XCOMET-XL and BLEURT-20 have separate node-local uv environments:

| Backend | uv project | Runtime note |
| --- | --- | --- |
| `scoring/sacrebleu` | `src/sure_eval/evaluation/nodes/scoring/sacrebleu` | Uses node-local `.cache/uv` and `.venv` |
| `scoring/xcomet_xl` | `src/sure_eval/evaluation/nodes/scoring/xcomet_xl` | Loads `Unbabel/XCOMET-XL` from `checkpoints/xcomet_xl`; uses `checkpoints/xlm_roberta_xl` for tokenizer/config assets; keeps uv cache at `.cache/uv` |
| `scoring/bleurt_20` | `src/sure_eval/evaluation/nodes/scoring/bleurt_20` | Loads BLEURT-20 from `checkpoints/bleurt_20/saved_model`; uses `checkpoints/bleurt_source` for the local BLEURT Python package source; keeps uv cache at `.cache/uv` |

Do not use `/tmp` as the persistent cache location for these metric tools.

## Regression Checks

The SacreBLEU route was checked against the legacy
`SUREEvaluator._eval_s2tt` path:

| Input | BLEU | chrF2 | Check |
| --- | ---: | ---: | --- |
| inline two-row zh smoke | 100.00000000000004 | 89.80466888994758 | pipeline result equals legacy result |
| `src/sure_eval/models/asr_kimi_audio/artifacts/ref_s2tt.txt` and `hyp_s2tt.txt` | 36.72652624971295 | 26.50461077394935 | pipeline result equals legacy result |
