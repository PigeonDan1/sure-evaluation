# scoring/bleurt_20

S2TT semantic scoring node backed by `BLEURT-20`.

Input contract:

| Role | Format |
| --- | --- |
| `hyp` | `key<TAB>hypothesis translation` |
| `ref` | `key<TAB>reference translation` |

All files must contain the same keys. The node reports per-segment scores and
uses the arithmetic mean as `score`.

By default, the node expects the BLEURT-20 checkpoint at
`src/sure_eval/evaluation/nodes/scoring/bleurt_20/checkpoints/bleurt_20/saved_model`.
Set `BLEURT_20_CHECKPOINT` to override that path.

```bash
BLEURT_20_CHECKPOINT=src/sure_eval/evaluation/nodes/scoring/bleurt_20/checkpoints/bleurt_20/saved_model \
UV_CACHE_DIR=src/sure_eval/evaluation/nodes/scoring/bleurt_20/.cache/uv \
UV_LINK_MODE=copy \
PYTHONPATH=src uv run --project src/sure_eval/evaluation/nodes/scoring/bleurt_20 \
  python -m sure_eval.evaluation.nodes.scoring.bleurt_20.node \
  --ref-file /path/to/ref.txt \
  --hyp-file /path/to/hyp.txt \
  --language zh \
  --output /path/to/bleurt_20.json
```

The uv package cache is node-local at
`src/sure_eval/evaluation/nodes/scoring/bleurt_20/.cache/uv`.
