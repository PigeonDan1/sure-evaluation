# SURE Evaluation

`sure_eval.evaluation` is the deterministic evaluation package used by SURE.
It is designed to work in two modes:

- as an internal module called by the main-agent and tool-agent workflows;
- as a standalone evaluation package that contributors can extend with new tasks,
  metrics, and scoring backends.

The package favors explicit inputs, traceable pipelines, and reproducible metric
implementations. A metric result should always explain what was computed, which
pipeline nodes were used, and which files were evaluated.

## Package Layout

```text
src/sure_eval/evaluation/
├── conversion/ # Input conversion profiles and format adapters
├── core/       # Shared types, pipeline records, and report structures
├── scripts/    # Stable user/agent entrypoints and output_dir writers
├── tasks/      # Task-level routes, manifests, and pipeline composition
└── nodes/      # Reusable normalization, transcription, and scoring nodes
```

### `scripts/`

Use this layer when calling evaluation from an agent workflow, a smoke test, or a
higher-level harness. It selects the configured task pipeline and writes standard
artifacts to `output_dir`.

Recommended APIs:

```python
from sure_eval.evaluation.scripts import describe_pipeline, run_task

description = describe_pipeline("asr", language="zh", metric="cer")
print(description.node_ids)
# ("normalization/aispeech_norm", "scoring/wenet_cer")

report = run_task(
    "asr",
    ref_file="ref.txt",
    hyp_file="hyp.txt",
    language="zh",
    metric="cer",
    output_dir="/tmp/sure_eval/asr_eval",
)
```

The selected task is always a pipeline, even when the requested metric name
looks like a scoring metric. For the ASR example above, the script selects
AISpeech text normalization first, then WeNet-style CER scoring. The generated
`pipeline_description.json` records both nodes:

```json
{
  "pipeline_id": "asr.zh.cer.aispeech_norm.wenet_cer",
  "node_ids": ["normalization/aispeech_norm", "scoring/wenet_cer"],
  "nodes": [
    {
      "node_id": "normalization/aispeech_norm",
      "stage": "normalization",
      "version": "v1",
      "manifest_path": "src/sure_eval/evaluation/nodes/normalization/aispeech_norm/manifest.yaml"
    },
    {
      "node_id": "scoring/wenet_cer",
      "stage": "scoring",
      "version": "v1",
      "manifest_path": "src/sure_eval/evaluation/nodes/scoring/wenet_wer/manifest.yaml"
    }
  ]
}
```

Every `run_task(...)` call must provide `output_dir`. The script layer writes:

- `report.json`: metric result, score, input files, and pipeline trace;
- `pipeline_description.json`: selected task, metric, node IDs, node metadata,
  conversion profiles, input contracts, and config paths.

Route selection is an execution contract, not only documentation. Script
entrypoints load the executor declared in `tasks/<task>/routes.yaml`; after the
executor returns, the script rejects the run if `report.pipeline_id` differs
from the selected route's `pipeline_id`. This keeps `routes.yaml`,
`pipeline_description.json`, and the actual metric implementation aligned.

### CLI

The project CLI exposes the same script layer as a two-step user interface. It
does not change the execution path.

Use it when a human or agent wants to call the evaluation package directly
without writing Python code.

First describe the route-backed pipeline and save the editable pipeline file:

```bash
sure-eval metric describe asr \
  --language zh \
  --metric cer \
  --output /tmp/asr_pipeline.json \
  --json
```

The saved pipeline JSON is the contract between the describe and run phases. It
contains the selected route, legal route choices, node slots, input roles, and
empty run arguments. Some tasks also include `conversion_steps`; these are not
metric nodes, but they must be reviewed because they adapt model or annotation
formats before deterministic scoring. A shortened ASR example looks like this:

```json
{
  "task": "asr",
  "language": "zh",
  "metric": "cer",
  "pipeline_id": "asr.zh.cer.aispeech_norm.wenet_cer",
  "pipeline": [
    {
      "slot": "normalization",
      "selected": "default",
      "default": "normalization/aispeech_norm",
      "nullable": true,
      "choices": ["normalization/aispeech_norm"]
    },
    {
      "slot": "scoring",
      "selected": "default",
      "default": "scoring/wenet_cer",
      "nullable": false,
      "choices": ["scoring/wenet_cer", "scoring/wenet_wer", "scoring/wenet_mer"]
    }
  ],
  "run_args": {
    "ref_file": null,
    "hyp_file": null,
    "output_dir": null
  }
}
```

Node selection is intentionally strict:

- `selected: null` skips a nullable node;
- `selected: "default"` uses the described default node;
- `selected: "<node_id>"` must already appear in that slot's `choices`;
- choices not emitted by `describe` are rejected.

Then execute the described pipeline:

```bash
sure-eval metric run \
  --pipeline /tmp/asr_pipeline.json \
  --ref-file ref.txt \
  --hyp-file hyp.txt \
  --output-dir /tmp/sure_eval/asr_eval \
  --json
```

For TTS and VC, use a samples JSONL instead of many per-role flags:

```bash
sure-eval metric describe tts \
  --language zh \
  --metrics tts_cer,sim/wavlm-large \
  --output /tmp/tts_pipeline.json \
  --json

sure-eval metric run \
  --pipeline /tmp/tts_pipeline.json \
  --samples-jsonl /tmp/tts_samples.jsonl \
  --output-dir /tmp/sure_eval/tts_eval \
  --device cuda \
  --cache-dir "${SURE_EVAL_CACHE_DIR:-$HOME/.cache/sure-eval}/tts-metrics" \
  --validate-env \
  --json
```

For Mandarin TTS semantic CER, the default route is
`frontend/funasr_loader_16k_mono -> transcription/paraformer_zh ->
normalization/punctuation_strip_norm -> scoring/wenet_cer`. This strips
punctuation only and does not use `normalization/aispeech_norm` unless an
explicit semantic normalizer selects it.

TTS rows use explicit roles:

```json
{"sample_id":"tts_001","prediction_audio":"out.wav","reference_text":"你好世界","reference_audio":"speaker.wav","language":"zh"}
```

VC rows use the converted audio, source/reference audio, and optional text:

```json
{"sample_id":"vc_001","converted_audio":"converted.wav","source_audio":"source.wav","reference_audio":"speaker.wav","reference_text":"你好世界","language":"zh"}
```

`output_dir` is always required. The CLI calls
`sure_eval.evaluation.scripts.run_task(...)`, so route executor loading,
pipeline-id validation, and standard output files remain centralized in the
script layer.

The command writes:

- `/tmp/sure_eval/asr_eval/report.json`;
- `/tmp/sure_eval/asr_eval/pipeline_description.json`.

With `--json`, stdout remains machine-readable for agents:

```json
{
  "status": "ok",
  "task": "ASR",
  "metric": "cer",
  "score": 0.3,
  "pipeline_id": "asr.zh.cer.aispeech_norm.wenet_cer",
  "report_path": "/tmp/sure_eval/asr_eval/report.json",
  "pipeline_description_path": "/tmp/sure_eval/asr_eval/pipeline_description.json",
  "environment_note": "node-local environments are not validated unless --validate-env is set. Check selected node directories for pyproject.toml or uv.lock when preparing a run.",
  "node_config_paths": [
    "src/sure_eval/evaluation/nodes/normalization/aispeech_norm/manifest.yaml",
    "src/sure_eval/evaluation/nodes/scoring/wenet_wer/manifest.yaml"
  ]
}
```

By default, the CLI only reports environment hints. Add `--validate-env` to
fail before execution when a selected node declares a required local runtime
but its `.venv`, checkpoint environment variables, or other local prerequisites
are missing. The check does not create environments; it points users and agents
to the node directory where `pyproject.toml`, `uv.lock`, or setup notes live.

For a package-level prerequisite check, run:

```bash
sure-eval doctor --json
```

When a contributor adds a route under `tasks/<task>/routes.yaml` and a node
manifest under `nodes/`, `metric describe` exposes the new legal choices. The
CLI should remain a thin interface over the script layer, not a second place
where metrics are hard-coded.

### `conversion/`

Conversion modules prepare metric inputs without becoming pipeline nodes. Use
this layer when a model's default output format does not match the canonical
input expected by a task route. Conversion can affect metric results, so routes
may expose conversion profiles in `pipeline_description.json` and
`report.json`.

Reusable route profiles live under:

```text
src/sure_eval/evaluation/conversion/{task_slug}__{metric_slug}/
```

Each profile should contain:

```text
README.md
manifest.yaml
convert.py
<source_format>_to_<target_format>.py
```

For example, SA-ASR cpWER uses:

```text
src/sure_eval/evaluation/conversion/sa_asr__cpwer/
src/sure_eval/evaluation/conversion/sa_asr__cpwer/stm_to_txt.py
src/sure_eval/evaluation/conversion/sa_asr__cpwer/txt_to_stm.py
```

The package-level `conversion/` tree should only keep representative,
reusable conversions selected by a human or agent. Model-specific or
experiment-specific converters should stay with model artifacts or run
artifacts until they are general enough to promote here.

The default SA-ASR route converts STM to key-text before
`normalization/gstar_norm`, then converts normalized key-text back to STM before
`scoring/meeteval`. The metric trace remains:

```text
normalization/gstar_norm -> scoring/meeteval
```

The conversion path is recorded separately as `conversion_trace`, and run
scripts persist conversion artifacts under `<output_dir>/conversion/...` when
the conversion can affect scoring.

### `tasks/`

Task modules define how a benchmark task is evaluated. A task may combine
multiple nodes into one pipeline. For example:

- ASR: normalization -> WER/CER/MER scoring;
- S2TT: SacreBLEU, XCOMET-XL, or BLEURT scoring;
- SD: MeetEval annotation loading -> DER scoring;
- SA-ASR: STM-to-key-text conversion -> G-STAR key-text normalization ->
  key-text-to-STM conversion -> MeetEval cpWER scoring with DER as the default
  companion metric;
- TTS/VC: transcription -> ASR scoring for semantic metrics, plus optional
  speaker similarity or MOS scoring;
- KWS/classification/SLU: task-specific loaders or normalization followed by
  scoring nodes.

Each task should keep its manifest explicit about supported metrics, input
contracts, default routes, and legacy compatibility notes. When a task has
multiple metric routes, prefer declaring the route in `tasks/<task>/routes.yaml`
instead of hard-coding node selection in `scripts/`.

SD and SA-ASR use annotation-file inputs rather than key-tab text. SD delegates
parsing to `meeteval.io.load`, so accepted formats follow MeetEval's own
support policy, including STM, CTM, SegLST, and RTTM for DER. SA-ASR first
uses `conversion/sa_asr__cpwer` to expose transcript fields as key-text for
`normalization/gstar_norm`; after normalization, it converts the text back to
STM and calls MeetEval for cpWER and DER. Route hyperparameters such as
`collar` are recorded in `report.json` under both the task details and the
`scoring/meeteval` node trace.

### `nodes/`

Nodes are reusable pipeline stages. Current stages include:

- `normalization/`: text or prompt normalization;
- `transcription/`: audio-to-text nodes used by semantic audio metrics;
- `scoring/`: metric backends and scoring wrappers.

A node should do one thing well. If an external script combines several internal
steps, keep the original behavior intact and document those internal stages in
the node manifest and pipeline trace.

### `core/`

`core/` contains shared records such as `EvaluationReport`, `PipelineNodeResult`,
`MetricInputContract`, and role-addressed input files. Metric implementations
should use these structures instead of inventing ad hoc report formats.

## Output Contract

All standard script executions must be reproducible from their output metadata.
At minimum:

- input roles are declared before execution;
- `pipeline_description.json` records selected nodes and config files;
- `report.json` records score, metric, task, input files, and node trace;
- node traces include `node_id`, `version`, and relevant internal stages.

This is required for fair comparison, debugging, and later audit of model
results.

## Adding A New Metric

Use the smallest extension point that matches the metric.

1. **Decide whether this is a new task or a new metric for an existing task.**
   Prefer extending an existing task when the input/output semantics match.

2. **Add reusable backend logic under `nodes/`.**
   Use `nodes/scoring/<backend_name>/` for scoring backends,
   `nodes/normalization/<name>/` for normalization, and
   `nodes/transcription/<name>/` for transcription.

3. **Keep authoritative scripts stable.**
   If a metric comes from a known toolkit or institution, wrap it instead of
   rewriting it. Avoid changing the original scoring behavior unless the change
   is intentional, reviewed, and regression-tested.

4. **Create or update `manifest.yaml`.**
   A node manifest should state `id`, `version`, `stage`, implementation path,
   input/output schema, language sensitivity, and internal stages. A task
   manifest should state supported metrics, input contracts, default routes, and
   aggregation policy.

5. **Declare or extend task routes.**
   Prefer adding a route entry under `tasks/<task>/routes.yaml`. A route should
   name the `pipeline_id`, `metric`, `nodes`, `input_contract`, and executor.
   This keeps script entrypoints thin and lets contributors add metrics by
   adding nodes and route config instead of editing shared script code.
   The executor must return an `EvaluationReport` whose `pipeline_id` matches
   the selected route after any route placeholders are resolved.

6. **Declare input roles explicitly.**
   Do not rely on positional files. Use names such as `ref`, `hyp`, `src`,
   `prediction_audio`, `reference_audio`, `reference_text`, `prompt_jsonl`, or
   task-specific names when needed.

7. **Add conversion profiles when input format adaptation affects scoring.**
   Put reusable adapters under `conversion/` using the
   `{task_slug}__{metric_slug}` naming convention. Include a `manifest.yaml`, a
   `convert.py`, and focused tests. Keep model-specific converters with the run
   artifacts until they are general enough to promote here.

8. **Wire the task pipeline.**
   Add the task-level route in `tasks/<task>/pipeline.py`. The task layer is
   responsible for composing nodes and returning an `EvaluationReport`.

9. **Expose the route in `scripts/` only after the task route is stable.**
   The script layer should select configurations and write outputs. It should
   not contain core metric logic.

10. **Use node-local environments for heavy or conflicting dependencies.**
   Metrics such as model-based scorers may need their own `pyproject.toml`,
   `.venv`, cache, or checkpoint directory under the node. Local environments
   and downloaded weights should not be committed. When a node uses its own
   interpreter, keep host Python site-packages out of the child `PYTHONPATH`;
   see `nodes/common/README.md`.

11. **Add focused tests.**
   At minimum, add a node test, a task pipeline test, and a script-entry test.
   If replacing an old metric path, include an alignment test against the legacy
   result on a small fixture.

## Contribution Checklist

Before opening a PR for a new metric or backend:

- [ ] The metric has a clear task owner and aggregation policy.
- [ ] Input roles and row format are documented in the task manifest.
- [ ] Route selection is declared in `routes.yaml` when the task has multiple
      pipeline options.
- [ ] The pipeline trace exposes every meaningful node version.
- [ ] Any conversion that can affect metric results has a profile and appears
      in `pipeline_description.json` or `report.json`.
- [ ] Heavy dependencies are isolated or documented.
- [ ] `run_task(...)` writes both standard output files.
- [ ] Tests cover node behavior, task routing, and script-level execution.
- [ ] Any legacy behavior change is backed by a before/after comparison.

## Design Principles

- Keep metric computation deterministic.
- Keep wrappers thin and auditable.
- Prefer task manifests and node manifests over hidden conventions.
- Do not mix unrelated task logic in one scoring node.
- Preserve exact external scoring behavior unless intentionally changing it.
- Make every reported score traceable to inputs, pipeline config, and node
  versions.
