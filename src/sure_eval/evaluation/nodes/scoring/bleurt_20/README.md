# BLEURT-20 Scoring

## Purpose

`scoring/bleurt_20` scores machine translation or speech translation output
with the BLEURT-20 learned semantic metric. It reports segment-level scores
and the arithmetic mean as the route score.

The node is a scorer only. It does not normalize text and does not download
checkpoints during evaluation.

## Task Scenarios

- S2TT semantic metric route:
  `s2tt.<language>.bleurt_20.bleurt_20_v1`.
- Optional heavy S2TT metric when BLEU/chrF are not sufficient.

## Input

- Schema: `key_text_hyp_ref`.
- Required roles:
  - `hyp`: `<key><TAB><hypothesis translation>`
  - `ref`: `<key><TAB><reference translation>`
- All files must contain the same keys.

## Output

- Schema: `segment_mean_result`.
- Metric: `bleurt_20`.
- Output includes per-segment scores and `score` as arithmetic mean.
- Higher scores are better.

## Versioned Computation

- Node id: `scoring/bleurt_20`.
- Version: `v1`.
- Model: `BLEURT-20`.
- Internal stages:
  - `aligned_input_loading`
  - `bleurt_20_inference`
  - `segment_mean`

## Runtime and Assets

- Runtime: optional node-local `uv` project.
- Python: `3.10`.
- Required checkpoint env var: `BLEURT_20_CHECKPOINT`.
- Default checkpoint target:
  `checkpoints/bleurt_20/saved_model/saved_model.pb`.

Run the node-local project:

```bash
UV_CACHE_DIR=src/sure_eval/evaluation/nodes/scoring/bleurt_20/.cache/uv \
UV_LINK_MODE=copy \
PYTHONPATH=src uv run --project src/sure_eval/evaluation/nodes/scoring/bleurt_20 \
  python -m sure_eval.evaluation.nodes.scoring.bleurt_20.node \
  --ref-file ref.txt --hyp-file hyp.txt --language zh --output bleurt_20.json
```

## Source and References

- BLEURT: https://github.com/google-research/bleurt
- BLEURT-20 checkpoint is a runtime asset and is not repository content.

## Limitations

- Requires TensorFlow/BLEURT in the node-local environment.
- Scores are only comparable when the same checkpoint and node version are
  used.
