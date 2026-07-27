# SacreBLEU S2TT Scoring

## Purpose

`scoring/sacrebleu` wraps SacreBLEU corpus BLEU and chrF scoring for S2TT
routes. Tokenizer selection is part of this scoring backend and is recorded as
an internal stage rather than modeled as a separate normalization node.

## Task Scenarios

- S2TT BLEU:
  `s2tt.<language>.bleu.sacrebleu_<language>_v1`.
- S2TT character-oriented BLEU compatibility selector:
  `s2tt.zh.bleu_char.sacrebleu_zh_v1`.
- S2TT chrF:
  `s2tt.zh.chrf.sacrebleu_zh_v1`.

## Input

- Schema: `key_text_files`.
- Required roles:
  - `hyp`: `<key><TAB><hypothesis translation>`
  - `ref`: `<key><TAB><reference translation>`
- Keys must be aligned before corpus scoring.

## Output

- Schema: `bleu_chrf_result`.
- Metrics: `bleu`, `bleu_char`, `chrf`.
- Returns legacy-compatible fields `bleu`, `bleu_char`, `chrf`, and `score`.
- Higher scores are better.

## Versioned Computation

- Node id: `scoring/sacrebleu`.
- Version: `v1`.
- Library calls:
  - `sacrebleu.metrics.BLEU`
  - `sacrebleu.metrics.CHRF`
- Tokenizer by language:
  - `zh`: `zh`
  - `en`: `13a`
  - default: `none`
- Internal stages:
  - `tokenizer_selection`
  - `corpus_bleu`
  - `corpus_chrf2`

## Runtime and Assets

- Runtime: root package dependency and optional node-local `uv` project for
  isolated tests.
- No model checkpoint.

Node-local test command:

```bash
UV_CACHE_DIR=src/sure_eval/evaluation/nodes/scoring/sacrebleu/.cache/uv \
UV_PROJECT_ENVIRONMENT=src/sure_eval/evaluation/nodes/scoring/sacrebleu/.venv \
UV_LINK_MODE=copy \
PYTHONPATH=src uv run --project src/sure_eval/evaluation/nodes/scoring/sacrebleu \
  python -m pytest tests/test_s2tt_pipeline_nodes.py
```

## Source and References

- SacreBLEU: https://github.com/mjpost/sacrebleu

## Limitations

- BLEU/chrF are text-only automatic metrics and do not use the source sentence.
- For source-aware semantic scoring, use `scoring/xcomet_xl` when prepared.
