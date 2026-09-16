# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SURE-EVAL: a deterministic, route-based evaluation framework for speech/audio tasks (ASR, S2TT, SD, SA-ASR, TTS, VC, classification/SER/GR, SLU, KWS). Every metric is a declared pipeline of versioned nodes; every run writes `report.json` + `pipeline_description.json` to a required `output_dir`.

## Commands

```bash
# Install (base is lightweight; heavy deps are node-local, see below)
pip install -e ".[dev]"
# Optional extras: [diarization] (MeetEval), [audio], [download], [wetext]

# Tests
pytest -q                                    # full suite
pytest tests/test_evaluation_cli.py -q       # one file
pytest tests/test_evaluation_cli.py::test_name -q

# Lint / format / types (all configured in pyproject.toml, line-length 100)
black src tests
ruff check src tests
mypy src

# CLI smoke (what CI runs)
sure-eval doctor --json
sure-eval metric describe asr --language zh --metric cer --json

# Two-step evaluation flow
sure-eval metric describe asr --language zh --metric cer --output /tmp/asr.json
sure-eval metric run --pipeline /tmp/asr.json --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/asr_eval

# Node-local environments for heavy metrics (always --dry-run first)
sure-eval env list --json
sure-eval env setup --task tts --language zh --metrics tts_cer,dnsmos --dry-run

# Regenerate docs/pipeline_catalog.jsonl after adding/changing routes
python scripts/generate_pipeline_catalog.py
```

CI (`.github/workflows/core.yml`) runs the CLI smoke commands plus a test subset (`test_evaluation_cli`, `test_evaluation_env_check`, `test_rps_manager`, `test_whisper_normalization_node`, `test_sctk_sclite_scoring_node`) on Python 3.10/3.11, and a wheel-build smoke.

## Architecture

All real code lives under `src/sure_eval/evaluation/`; top-level `sure_eval/cli.py` just re-exports the Typer app from `evaluation/cli.py` (subcommands: `metric describe|run`, `env list|check|setup|download`, `doctor`).

The layers, from stable entrypoint down:

- **`scripts/`** — stable user/agent entrypoints: `describe_pipeline(...)` and `run_task(...)` (the recommended Python API), plus per-task modules. Loads the executor declared in the route, then **rejects the run if the returned `report.pipeline_id` differs from the selected route's `pipeline_id`** — route selection is an enforced execution contract, not documentation. Always writes `report.json` + `pipeline_description.json`. Keep metric logic out of this layer and out of the CLI.
- **`tasks/<task>/`** — `routes.yaml` declares each route: `pipeline_id`, `language`, `metric`, `nodes`, `input_contract`, and `executor` (a dotted path into that task's `pipeline.py`, which composes nodes and returns an `EvaluationReport`). Adding a metric to an existing task usually means adding a route entry, not editing script code.
- **`nodes/<stage>/<name>/`** — reusable stages: `normalization/`, `transcription/`, `scoring/`, `frontend/`. Each node dir owns `manifest.yaml` (id, version, stage, schemas, implementation) and, for heavyweight backends, `node_env.yaml` + its own `pyproject.toml`/`.venv`/checkpoints managed via `sure-eval env ...` with `uv`. When a node runs its own interpreter, keep host site-packages out of the child `PYTHONPATH` (see `nodes/common/README.md`).
- **`conversion/{task_slug}__{metric_slug}/`** — format adapters that run before/after nodes without being nodes (e.g. `sa_asr__cpwer` STM↔key-text). Recorded separately as `conversion_trace`; artifacts persist under `<output_dir>/conversion/` when they can affect scoring.
- **`core/`** — shared types: `EvaluationReport`, `PipelineNodeResult`, `MetricInputContract`, role-addressed input files. Use these instead of ad hoc report formats.

Layout invariants (e.g. normalization impls live under node packages, TTS/VC speaker routes use named backend nodes) are enforced by `tests/test_evaluation_architecture_contracts.py` — if you move code, run it.

### Input conventions

- Text tasks (ASR, S2TT, classification, SLU): tab-separated `<key>\t<text>` files with explicit roles (`--ref-file`, `--hyp-file`, `src`, ...). Never positional.
- TTS/VC: a samples JSONL with explicit role fields per row (`prediction_audio`, `reference_text`, `reference_audio`, `converted_audio`, ...).
- SD/SA-ASR: annotation files (STM/CTM/SegLST/RTTM via MeetEval), require the `[diarization]` extra.

Per-task input formats, pipeline IDs, and examples: `docs/tasks/<task>.md`; machine-readable metric→pipeline→node map: `docs/pipeline_catalog.jsonl`.

### Adding a metric

Full checklist in `docs/add_a_metric.md` and `src/sure_eval/evaluation/README.md`. Short version: add node under `nodes/<stage>/<name>/` with `manifest.yaml` (+ `node_env.yaml` if heavy) → add route in `tasks/<task>/routes.yaml` → add node/route/script tests → verify with `sure-eval metric describe` and `sure-eval env setup --dry-run` → regenerate the pipeline catalog. Wrap external toolkits (WeNet, MeetEval, SCTK, ...) without changing upstream scoring behavior unless the change is intentional, documented, and regression-tested.

## Notes

- Never commit runtime assets: `.venv`s, checkpoints (`*.ckpt/pt/onnx/safetensors/bin`), caches, generated reports, private absolute paths. `SURE_EVAL_CACHE_DIR` redirects caches to a large disk.
- `ARCHITECTURE.source.md` describes the larger SURE agent system (main_flow_agent / model_tool_agent, `fixtures/`, `models/`); several of those paths do not exist in this standalone repo — trust the actual tree and `src/sure_eval/evaluation/README.md` over it.
- `tests/conftest.py` stubs `structlog`, so most tests run without optional deps installed.
