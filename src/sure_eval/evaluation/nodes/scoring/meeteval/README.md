# MeetEval Scoring

This node wraps the `meeteval` toolkit for diarization and multi-speaker ASR
metrics. Input files are passed to `meeteval.io.load`, so the node keeps
MeetEval's own format support instead of implementing a local parser.

Supported routes currently use:

- `der`: diarization error rate via `meeteval.der.dscore`
- `cpwer`: concatenated minimum-permutation WER via `meeteval.wer.cpwer`

The report records the selected metric, `collar`, companion metrics, loader,
aggregation rule, and per-session results when MeetEval exposes them.

DER uses `md-eval-22.pl` through MeetEval. Keep a local copy at one of:

```text
src/sure_eval/evaluation/nodes/scoring/meeteval/md-eval-22.pl
src/sure_eval/evaluation/nodes/scoring/meeteval/.cache/md-eval-22.pl
```

The node adds that location to `PATH` before scoring. This avoids runtime
downloads during SD and SA-ASR evaluation.

## Runtime

Use this scoring node as the uv project:

```bash
env \
  UV_CACHE_DIR="$(pwd)/src/sure_eval/evaluation/nodes/scoring/meeteval/.cache/uv" \
  UV_PROJECT_ENVIRONMENT="$(pwd)/src/sure_eval/evaluation/nodes/scoring/meeteval/.venv" \
  UV_LINK_MODE=copy \
  CXX=/usr/bin/g++ \
  PYTHONPATH=src \
  uv sync --project src/sure_eval/evaluation/nodes/scoring/meeteval
```

Run a quick route check with the node-local Python:

```bash
PYTHONPATH=src \
src/sure_eval/evaluation/nodes/scoring/meeteval/.venv/bin/python - <<'PY'
from sure_eval.evaluation.scripts import describe_pipeline

print(describe_pipeline("sd").pipeline_id)
print(describe_pipeline("sa-asr").pipeline_id)
PY
```

`UV_CACHE_DIR` and `UV_PROJECT_ENVIRONMENT` are intentionally inside this node.
Do not keep the persistent MeetEval metric environment in `/tmp`.
