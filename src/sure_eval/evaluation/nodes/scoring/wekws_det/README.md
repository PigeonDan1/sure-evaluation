# KWS operating-point scoring, v2

`scoring/wekws_det` v2 evaluates keyword recall and trigger latency under one
system-wide false-alarm-per-hour budget (default **0.1**). Its historical node
name is retained; the v1 DET/accuracy scoring contract is no longer used.

The input is aligned `KWSSample` records with uncensored scalar/frame scores or
explicit candidate trigger lists. Negative recording durations are required.
The output contains equal-weight macro recall, per-keyword recall, the selected
threshold and actual FA/hour, and overall/per-keyword trigger latency in ms.
Optional clip ROC/AUC requires complete per-keyword scores for multiclass data.

See [the KWS task contract](../../../../../../docs/tasks/kws.md) for formulas,
JSON schemas, matching windows, frame timing, input requirements, and CLI examples.
No model inference, model weights, or additional plotting dependency is required.

Discover and prepare only the selected pipeline:

```bash
sure-eval metric routes kws --metric macro_recall --json
sure-eval metric describe kws \
  --pipeline-id kws.any.macro_recall.conversion_kws_sure_json_to_samples_v2.wekws_det_v2 \
  --output /tmp/kws.json
sure-eval env setup --pipeline /tmp/kws.json --dry-run
sure-eval env setup --pipeline /tmp/kws.json
sure-eval env check --pipeline /tmp/kws.json
```

Validation: `python -m pytest tests/test_kws_metrics.py tests/test_kws_pipeline_nodes.py`.
