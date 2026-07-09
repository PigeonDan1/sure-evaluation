# SacreBLEU S2TT Scoring

This node wraps the S2TT scoring behavior previously embedded in
`SUREEvaluator._eval_s2tt`.

Tokenizer selection, corpus BLEU, and chrF2 are treated as internal stages of
the same SacreBLEU scoring backend. S2TT currently does not add a separate
normalization node.

## Runtime

The dependency project is this scoring node:

```text
src/sure_eval/evaluation/nodes/scoring/sacrebleu
```

Keep runtime files under the same scoring backend:

```bash
env \
  UV_CACHE_DIR=src/sure_eval/evaluation/nodes/scoring/sacrebleu/.cache/uv \
  UV_PROJECT_ENVIRONMENT=src/sure_eval/evaluation/nodes/scoring/sacrebleu/.venv \
  UV_LINK_MODE=copy \
  PYTHONPATH=src \
  uv run --project src/sure_eval/evaluation/nodes/scoring/sacrebleu \
  python -m pytest tests/test_s2tt_pipeline_nodes.py
```

`UV_CACHE_DIR` and `UV_PROJECT_ENVIRONMENT` are intentionally inside this node.
Do not keep the persistent SacreBLEU metric cache in `/tmp`.
