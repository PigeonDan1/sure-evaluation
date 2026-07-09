# Common Node Runtime Notes

This directory contains shared helpers for repository evaluation nodes.

## Node-Local Python Isolation

Node-local providers are isolated dependency surfaces. A provider may use a
different interpreter from the main evaluation process, for example UTMOS can
run from a Python 3.8 node-local `.venv` while `scripts/evaluate_predictions.py`
runs under the repository host Python.

When launching a node-local provider through an explicit interpreter such as:

```text
src/sure_eval/evaluation/nodes/scoring/utmos/.venv/bin/python
```

do not let the subprocess inherit host Python site-packages through the parent
`PYTHONPATH`. A parent value such as:

```text
.venv.hostbak/lib/python3.11/site-packages:src
```

can make a Python 3.8 provider import Python 3.11 compiled packages. Typical
symptoms include:

- `No module named numpy._core._multiarray_umath`
- compiled-extension ABI mismatches, such as a `cpython-311` shared object
  loaded by a Python 3.8 subprocess

The shared launcher should preserve only what is needed to import repository
code, and must not prepend the parent environment's site-packages when the node
has selected its own `.venv/bin/python`.

Before a full TTS/VC metric segment, verify the node-local runtime with one real
sample. If this fails, fix the evaluation execution surface; do not rerun model
inference and do not switch to the model inference image for metric
dependencies.

## TTS/VC Evaluation-Only Surface

For TTS and VC runs, prediction generation and audio metric evaluation are
separate execution surfaces. Model inference may run in a model Docker image or
through a scheduler, but evaluation-only retries should default to the
repository checkout and node-local evaluation environments:

```text
scripts/evaluate_predictions.py
  -> sure_eval.evaluation.scripts.run_task(...)
  -> tasks/<task>/routes.yaml
  -> nodes/<node_id>/.venv/bin/python
```

Do not submit an evaluation-only retry through `vc` merely because the
prediction run used `vc`. The model image is not the source of metric logic or
metric dependencies. Reuse validated prediction artifacts and repair the
node-local evaluation surface instead.

The exception is an explicit operator decision to host the same repository
evaluation command on cluster resources after local node-local execution is
blocked. In that case, the job must still call the repository evaluation router
and the same node-local providers; it must not replace metric nodes with ad hoc
image-specific scorers.

## Prediction-Complete TTS/VC Evaluation Lessons

When TTS/VC prediction artifacts already exist and validation passes, treat the
remaining work as evaluation-only work. The evaluation should use the host
repository checkout, the task route, and the node-local metric providers. Do not
resubmit through `vc` unless the operator explicitly asks for cluster hosting of
that same repository evaluation command.

Prediction manifests may contain absolute paths from the inference container,
for example `/workspace/sure-eval/...`. Host-side evaluation wrappers and
preflight checks must remap those paths to the current repository checkout before
declaring audio missing. The official evaluation path resolver may already know
how to resolve these files; wrapper checks must match that behavior instead of
failing early on container-only prefixes.

Keep wrapper preflight lightweight. The wrapper should verify repository imports,
task routing, input manifests, output directories, and the existence of
node-local interpreters/checkpoints. It should not import heavy node
dependencies such as `torch` from the parent interpreter, because each
node-local provider owns its own dependency surface.

For an evaluation-only failure, record the run as evaluation-incomplete rather
than prediction-incomplete when:

- prediction validation passed;
- audio files exist and are readable after path remapping;
- at least one unrelated metric family can score the same predictions.

This distinction prevents unnecessary model inference reruns and keeps blame on
the failing evaluation surface.

## Heavy Audio Nodes

Model-based audio metrics such as WV-MOS and UTMOS have two practical failure
modes that should be separated during triage:

- the node-local environment or checkpoint is missing or incompatible;
- the environment is present, but the selected CUDA path fails at runtime.

Use a one-sample probe before classifying a model or dataset as bad. For
WV-MOS, a CPU one-sample success proves that the repository, checkpoint, and
basic provider are usable. If the corresponding CUDA probe fails with a
low-level error such as:

```text
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
```

then the first suspect is the WV-MOS CUDA/cuDNN path, not the model output and
not the dataset. Keep the investigation scoped to the node-local provider,
torch/CUDA visibility, and batch/subprocess strategy.

When a node-local GPU command fails in a sandbox but `nvidia-smi` sees devices
outside the sandbox, verify with the same command on the real execution
surface before changing metric logic. A sandbox-only failure such as:

```text
RuntimeError: No CUDA GPUs are available
```

does not prove that the node cannot run on the local machine.

## Batch And Subprocess Strategy

`NodeLocal*Provider.score_batch(...)` calls a node-local subprocess. Batch size
therefore controls both memory behavior and process/model-load overhead.

For heavyweight MOS nodes, very small chunks can be slower than a full-dataset
subprocess because each chunk reloads the model. Use chunking only to isolate
OOM or per-sample failures. If a full-dataset subprocess succeeds, prefer it so
the model is loaded once and reused inside that process.

Do not assume that a chunked run is safer simply because it uses fewer samples
per subprocess. It may hide the original issue while making the run too slow to
finish.

Some nodes, especially speaker-similarity providers such as ERes2Net, launch
many short node-local subprocesses. The parent process may spend most of its
time polling while child PIDs appear and disappear. That is slow progress, not
proof of a hang.

Metric logs do not always update per sample. ERes2Net and DNSMOS may only emit
visible progress at dataset or segment boundaries. Before interrupting a quiet
run, check child processes, output artifact mtimes, and final segment logs.

If a heavyweight provider OOMs, first reduce the provider-specific batch size
or isolate the problematic dataset/metric segment. A recursive split or CPU
fallback can complete correctly but take much longer than the original GPU plan,
so classify it as runtime degradation unless the metric returns invalid output.

## Segment And Merge Discipline

Long TTS/VC evaluations may be segmented by metric family for runtime and
failure isolation. Each segment still has to call the same repository
evaluation route and must write the normal artifacts:

- `evaluation_payload.json`
- `report.jsonl`
- `protocol.yaml`
- `metrics/<dataset>/<metric_slug>/report.json`
- `metrics/<dataset>/<metric_slug>/pipeline_description.json`
- `sample_reports/<dataset>/<metric_slug>.jsonl`

The final merge should only merge verified segment payloads and refresh the
run-level report artifacts. It should not regenerate audio, rerun completed
metrics, or repeat expensive semantic probes unless the operator explicitly
needs to revalidate the node-local environment.

Segment names should describe the metric family or missing work, not the
execution host. A final run-level merge is only valid when every segment was
produced by the same repository evaluation contract and points at the same
validated prediction artifacts.

If prediction validation already passed, an evaluation failure should be
recorded as evaluation-incomplete, not prediction-incomplete. Do not rerun
model inference unless the prediction artifacts are missing, invalid, or known
stale.

## Attribution Checklist

When an audio evaluation fails, classify the failure from evidence:

- **Model issue**: prediction validation fails, required audio is missing,
  generated files are unreadable, or many metric nodes fail on the same sample
  for input-quality reasons.
- **Dataset issue**: annotations or role fields are missing or inconsistent,
  the metric is incompatible with the dataset language/task, or the failure
  reproduces with known-good predictions on the same dataset row.
- **Evaluation runtime issue**: one metric node fails while other metrics and
  prediction validation pass, a CPU probe succeeds but CUDA fails, a
  node-local `.venv` imports the wrong ABI, or a sandbox cannot expose GPU
  devices to the child process.

Do not generalize one metric's runtime failure into a model failure without
checking the other metric families and the prediction validation payload.
