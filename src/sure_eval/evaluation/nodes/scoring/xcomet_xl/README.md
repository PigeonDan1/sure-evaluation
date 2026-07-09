# scoring/xcomet_xl

S2TT semantic scoring node backed by `Unbabel/XCOMET-XL`.

Input contract:

| Role | Format |
| --- | --- |
| `src` | `key<TAB>source text` |
| `hyp` | `key<TAB>hypothesis translation` |
| `ref` | `key<TAB>reference translation` |

All files must contain the same keys. The node reports per-segment scores and
uses the arithmetic mean as `score`.

Run in its own uv environment:

```bash
UV_CACHE_DIR=src/sure_eval/evaluation/nodes/scoring/xcomet_xl/.cache/uv \
UV_LINK_MODE=copy \
PYTHONPATH=src uv run --project src/sure_eval/evaluation/nodes/scoring/xcomet_xl \
  python -m sure_eval.evaluation.nodes.scoring.xcomet_xl.node \
  --src-file /path/to/src.txt \
  --ref-file /path/to/ref.txt \
  --hyp-file /path/to/hyp.txt \
  --language zh \
  --output /path/to/xcomet_xl.json
```

The local ModelScope checkpoint is stored under
`src/sure_eval/evaluation/nodes/scoring/xcomet_xl/checkpoints/xcomet_xl/modelscope`,
and the required XLM-Roberta-XL tokenizer/config cache is stored under
`src/sure_eval/evaluation/nodes/scoring/xcomet_xl/checkpoints/xlm_roberta_xl/huggingface`.
The uv package cache is node-local at
`src/sure_eval/evaluation/nodes/scoring/xcomet_xl/.cache/uv`.
When the local checkpoint is present, the node sets Hugging Face/Transformers
offline flags before loading the model. If no local checkpoint is found, set
`XCOMET_XL_CHECKPOINT_DIR` to control where COMET downloads `Unbabel/XCOMET-XL`.
