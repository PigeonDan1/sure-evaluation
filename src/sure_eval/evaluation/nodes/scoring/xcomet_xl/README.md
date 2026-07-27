# XCOMET-XL Scoring

## Purpose

`scoring/xcomet_xl` scores translation quality with the source-aware
`Unbabel/XCOMET-XL` model. It uses source, hypothesis, and reference text and
reports segment scores plus their arithmetic mean.

## Task Scenarios

- S2TT semantic metric route:
  `s2tt.<language>.xcomet_xl.xcomet_xl_v1`.
- Optional heavy S2TT metric when source-aware quality estimation is required.

## Input

- Schema: `key_text_src_hyp_ref`.
- Required roles:
  - `src`: `<key><TAB><source text>`
  - `hyp`: `<key><TAB><hypothesis translation>`
  - `ref`: `<key><TAB><reference translation>`
- All files must contain the same keys.

## Output

- Schema: `segment_mean_result`.
- Metric: `xcomet_xl`.
- Output includes per-segment scores and `score` as arithmetic mean.
- Higher scores are better.

## Versioned Computation

- Node id: `scoring/xcomet_xl`.
- Version: `v1`.
- Model: `Unbabel/XCOMET-XL`.
- Encoder cache: `facebook/xlm-roberta-xl`.
- Internal stages:
  - `aligned_input_loading`
  - `xcomet_xl_inference`
  - `segment_mean`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.11`.
- GPU: optional.
- Required checkpoint env vars:
  - `XCOMET_XL_CHECKPOINT_PATH`
  - `XCOMET_XL_ENCODER_CACHE`
- Local checkpoint targets:
  - `checkpoints/xcomet_xl/modelscope/evalscope/XCOMET-XL/checkpoints/model.ckpt`
  - `checkpoints/xlm_roberta_xl/huggingface`

Run the node-local project:

```bash
UV_CACHE_DIR=src/sure_eval/evaluation/nodes/scoring/xcomet_xl/.cache/uv \
UV_LINK_MODE=copy \
PYTHONPATH=src uv run --project src/sure_eval/evaluation/nodes/scoring/xcomet_xl \
  python -m sure_eval.evaluation.nodes.scoring.xcomet_xl.node \
  --src-file src.txt --ref-file ref.txt --hyp-file hyp.txt --language zh --output xcomet_xl.json
```

## Source and References

- COMET: https://github.com/Unbabel/COMET
- XCOMET-XL model card: https://huggingface.co/Unbabel/XCOMET-XL

## Limitations

- Requires prepared model and encoder caches for offline reproducible runs.
- Scores are only comparable with the same model checkpoint and node version.
