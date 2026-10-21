# KWS evaluation task, v2

The default primary score is equal-weight keyword `macro_recall` at a shared
system budget of **0.1 FA/hour**. Reports also contain `per_keyword_recall` and
`trigger_latency`; `roc_curve` and `auc` are optional. Old accuracy, F1, precision,
FRR/FAR, and DET outputs have been removed.

- `manifest.yaml` and `routes.yaml` declare the three exact input routes.
- `loaders.py` aligns all reference/output keys and preserves raw scores and times.
- `pipeline.py` constructs versioned reports using `scoring/wekws_det` v2.
- `compat.py` exposes the `KWSMetricPipeline` Python convenience wrapper.
- `KWSEvent`, `KWSTrigger`, and `KWSSample` support clipped and multi-event input.

See [the public KWS contract](../../../../../docs/tasks/kws.md) for metric formulas,
input examples, matching rules, timestamp units, and migration requirements.

Use `sure-eval metric routes kws --metric macro_recall --json`, describe one exact
pipeline, prepare/check that pipeline's environments, and score through
`sure-eval metric run --pipeline ...`. The convenience script
`scripts/run_kws_metric_pipeline.py` follows this same JSON pipeline workflow.

All input conversions and the scoring node are v2. A representative identity is
`kws.any.macro_recall.conversion_kws_sure_json_to_samples_v2.wekws_det_v2`.
Raw scores are required even on rejected clips; event lists represent distinct
pre-threshold candidate events. Negative audio durations are mandatory.
