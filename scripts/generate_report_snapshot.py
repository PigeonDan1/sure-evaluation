#!/usr/bin/env python3
"""Generate a user-facing report snapshot for a main-flow evaluation run.

Reads artifacts from ``<run_dir>/`` (execution_surface.json, evaluation_payload.json,
etc.) and renders ``docs/agents/main_flow_agent/templates/report_snapshot.md`` into
``results/<model_name>/<protocol_id>/report_snapshot.md``.

Fields that cannot be determined are filled with ``N/A``.

Usage::

    python scripts/generate_report_snapshot.py \
        --run-dir <selected_model_dir>/eval_runs/main_agent_asr_qwen3_002

    # Or with explicit output path
    python scripts/generate_report_snapshot.py \
        --run-dir <selected_model_dir>/eval_runs/main_agent_asr_qwen3_002 \
        --output <selected_model_dir>/eval_runs/main_agent_asr_qwen3_002/report_snapshot.md
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from sure_eval.core.logging import configure_logging, get_logger

configure_logging(level="INFO")
logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = (
    REPO_ROOT
    / "docs"
    / "agents"
    / "main_flow_agent"
    / "templates"
    / "report_snapshot.md"
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to load JSON: {path}", error=str(exc))
        return None


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if yaml is None or not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning(f"Failed to load YAML: {path}", error=str(exc))
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _na(value: Any) -> str:
    """Return N/A for missing/empty values, preserving bool/number 0."""
    if value is None:
        return "N/A"
    if isinstance(value, str) and not value.strip():
        return "N/A"
    return str(value)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _resolve_run_paths(run_dir: Path) -> dict[str, Any]:
    """Resolve model_name, run_id, model_dir from run_dir path."""
    run_dir = run_dir.resolve()
    parts = run_dir.parts
    result: dict[str, Any] = {
        "run_dir": _display_path(run_dir),
        "run_dir_absolute": str(run_dir),
    }
    try:
        if "eval_runs" in parts:
            idx = parts.index("eval_runs")
            if idx > 0:
                result["model_dir_absolute"] = str(Path(*parts[:idx]))
                result["model_name"] = parts[idx - 1]
            if len(parts) > idx + 1:
                result["run_id"] = parts[idx + 1]
    except Exception:
        pass
    return result


def _parse_run_evaluation_sh(run_evaluation_sh: Path) -> dict[str, str]:
    """Parse selected environment variables from run_evaluation.sh."""
    values: dict[str, str] = {}
    if not run_evaluation_sh.exists():
        return values
    text = run_evaluation_sh.read_text(encoding="utf-8")
    for key in ("RESULTS_DIR", "PROTOCOL_ID", "MODEL_NAME", "TOOL_NAME", "DEVICE"):
        pattern = rf'^{re.escape(key)}="([^"]+)"'
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            raw = m.group(1)
            # Skip shell default-value expressions like ${DEVICE:-} or ${VAR:-default}
            if re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-[^}]*\}", raw):
                continue
            values[key] = raw
    return values


def _build_dataset_scope_block(
    datasets: list[str],
    eval_payload: dict[str, Any] | None,
    pred_status: dict[str, Any] | None,
    report_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Build the 'Dataset Scope' markdown block."""
    if not datasets:
        return "- Evaluated datasets: N/A"

    lines: list[str] = []
    for ds in datasets:
        # Find num_samples from evaluation payload
        num_samples = "N/A"
        if report_rows:
            for row in report_rows:
                dataset = row.get("dataset", {})
                if dataset.get("name") == ds:
                    num_samples = dataset.get("num_samples", "N/A")
                    break
        if eval_payload:
            for r in eval_payload.get("results", []):
                if r.get("dataset") == ds:
                    result = r.get("result") or {}
                    num_samples = result.get("num_samples", r.get("num_samples", "N/A"))
                    break
        lines.append(f"- Dataset: `{ds}`")
        lines.append(f"  - Samples evaluated: {num_samples}")
    return "\n".join(lines)


def _build_result_summary_block(
    eval_payload: dict[str, Any] | None,
    results_dir: Path,
    run_dir: Path,
    report_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Build the 'Result Summary' markdown block."""
    if report_rows:
        lines = []
        lines.append(f"- Run-local result file: `{_display_path(run_dir / 'report.jsonl')}`")
        lines.append(f"- Run-local protocol file: `{_display_path(run_dir / 'protocol.yaml')}`")
        lines.append("")
        lines.append("| Dataset | Task | Lang | Samples | Metric | Score | SOTA | RPS | Status |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in report_rows:
            dataset = row.get("dataset", {})
            metric = row.get("metric", {})
            baseline = row.get("baseline") or {}
            rps = row.get("rps") or {}
            lines.append(
                "| "
                f"`{dataset.get('name', 'N/A')}` | "
                f"{dataset.get('task', 'N/A')} | "
                f"{dataset.get('language', 'N/A')} | "
                f"{dataset.get('num_samples', 'N/A')} | "
                f"{metric.get('name', 'N/A').upper()} | "
                f"{metric.get('display', metric.get('score', 'N/A'))} | "
                f"{baseline.get('score', 'N/A')} | "
                f"{rps.get('value', 'N/A')} | "
                f"{row.get('status', 'N/A')} |"
            )
        return "\n".join(lines)

    if not eval_payload or not eval_payload.get("results"):
        return "- No evaluation results available."

    lines: list[str] = []
    protocol_file = _display_path(results_dir / "protocol.yaml")
    report_file = _display_path(results_dir / "report.jsonl")

    lines.append(f"- Standard result file: `{report_file}`")
    lines.append(f"- Standard protocol file: `{protocol_file}`")

    for r in eval_payload["results"]:
        dataset = r.get("dataset", "N/A")
        metric = r.get("metric", "N/A")
        result = r.get("result") or {}
        score = result.get("score", r.get("score", "N/A"))
        rps = r.get("rps", "N/A")
        details = r.get("details", result)
        lines.append(f"- Dataset: `{dataset}`")
        lines.append(f"  - Metric selected by repository evaluation: {metric.upper()}")
        lines.append(f"  - {metric.upper()}: {score}")
        if isinstance(score, (int, float)):
            lines.append(f"  - {metric.upper()} percent: {score * 100:.6f}%")
        if isinstance(details, dict) and "all" in details:
            lines.append(
                f"  - Error counts: "
                f"`all={details.get('all', 'N/A')}`, "
                f"`cor={details.get('cor', 'N/A')}`, "
                f"`sub={details.get('sub', 'N/A')}`, "
                f"`ins={details.get('ins', 'N/A')}`, "
                f"`del={details.get('del', 'N/A')}`"
            )
        lines.append(f"  - RPS: {rps}")
    return "\n".join(lines)


def _build_evaluation_pipeline_block(eval_payload: dict[str, Any] | None, report_rows: list[dict[str, Any]] | None = None) -> str:
    """Build the 'Evaluation Pipeline' markdown block."""
    if report_rows:
        lines = []
        lines.append("| Dataset | Metric | Pipeline ID | Stage | Node | Node Version | Manifest |")
        lines.append("|---|---:|---|---|---|---|---|")
        for row in report_rows:
            dataset = row.get("dataset", {})
            metric = row.get("metric", {})
            pipeline = row.get("pipeline") or {}
            nodes = pipeline.get("nodes") or []
            if not nodes:
                lines.append(
                    f"| `{dataset.get('name', 'N/A')}` | {metric.get('name', 'N/A').upper()} | "
                    f"`{pipeline.get('pipeline_id', 'N/A')}` | N/A | N/A | N/A | N/A |"
                )
                continue
            for node in nodes:
                lines.append(
                    f"| `{dataset.get('name', 'N/A')}` | {metric.get('name', 'N/A').upper()} | "
                    f"`{pipeline.get('pipeline_id', 'N/A')}` | "
                    f"{node.get('stage', 'N/A')} | `{node.get('node_id', 'N/A')}` | "
                    f"{node.get('version', 'N/A')} | `{node.get('manifest_path', 'N/A')}` |"
                )
        return "\n".join(lines)

    if not eval_payload or not eval_payload.get("results"):
        return "- Evaluation pipeline details unavailable."

    lines: list[str] = []
    for r in eval_payload["results"]:
        dataset = r.get("dataset", "N/A")
        ctx = r.get("evaluation_context", {})
        metric = r.get("metric", "N/A")
        lines.append(f"- Dataset: `{dataset}`")
        lines.append(f"  - Metric: `{metric.upper()}`")
        if ctx.get("conversion"):
            lines.append(f"  - Conversion: `{ctx['conversion']}`")
        pipeline_parts = []
        if ctx.get("normalization"):
            pipeline_parts.append(f"normalization={ctx['normalization']}")
        if ctx.get("scoring"):
            pipeline_parts.append(f"scoring={ctx['scoring']}")
        if ctx.get("postprocessing"):
            pipeline_parts.append(f"postprocessing={ctx['postprocessing']}")
        if pipeline_parts:
            lines.append(f"  - Pipeline: `{', '.join(pipeline_parts)}`")
    return "\n".join(lines)


def _build_output_artifacts_block(
    run_dir: Path,
    results_dir: Path,
    datasets: list[str],
    report_rows: list[dict[str, Any]] | None = None,
) -> str:
    """Build the 'Output Artifacts' markdown block using repo-relative paths."""
    lines: list[str] = []
    lines.append(f"- Run-local protocol file: `{_display_path(run_dir / 'protocol.yaml')}`")
    lines.append(f"- Run-local dataset-metric report: `{_display_path(run_dir / 'report.jsonl')}`")
    lines.append(f"- Run-local markdown snapshot: `{_display_path(run_dir / 'report_snapshot.md')}`")
    lines.append(f"- Standard protocol mirror: `{_display_path(results_dir / 'protocol.yaml')}`")
    lines.append(f"- Standard result mirror: `{_display_path(results_dir / 'report.jsonl')}`")
    for ds in datasets:
        lines.append(
            f"- Standard sample prediction detail: `{_display_path(results_dir / 'predictions' / f'{ds}.txt')}`"
        )
        lines.append(
            f"- Run prediction file: `{_display_path(run_dir / 'predictions' / f'{ds}.txt')}`"
        )
    lines.append(f"- Prediction manifest: `{_display_path(run_dir / 'predictions' / 'manifest.json')}`")
    lines.append(f"- Prediction generation status: `{_display_path(run_dir / 'prediction_generation_status.json')}`")
    lines.append(f"- Validation payload: `{_display_path(run_dir / 'validation_payload.json')}`")
    lines.append(f"- Evaluation payload: `{_display_path(run_dir / 'evaluation_payload.json')}`")
    if report_rows:
        for row in report_rows:
            dataset = row.get("dataset", {}).get("name", "dataset")
            metric = row.get("metric", {}).get("name", "metric")
            pipeline = row.get("pipeline") or {}
            artifacts = row.get("artifacts") or {}
            if pipeline.get("report_path"):
                lines.append(f"- Metric report `{dataset}/{metric}`: `{pipeline['report_path']}`")
            if pipeline.get("description_path"):
                lines.append(f"- Pipeline description `{dataset}/{metric}`: `{pipeline['description_path']}`")
            if artifacts.get("sample_report"):
                lines.append(f"- Sample report `{dataset}/{metric}`: `{artifacts['sample_report']}`")
    return "\n".join(lines)


def _build_runtime_versions_block(report_rows: list[dict[str, Any]] | None) -> str:
    if not report_rows:
        return "- Runtime version details unavailable."
    first = report_rows[0]
    versions = first.get("versions") or {}
    python_info = versions.get("python") or {}
    sure_eval = versions.get("sure_eval") or {}
    git = sure_eval.get("git") or {}
    tools = versions.get("tools") or {}
    lines = [
        "| Component | Version / Path |",
        "|---|---|",
        f"| sure_eval package | `{sure_eval.get('package_version', 'N/A')}` |",
        f"| sure_eval git commit | `{git.get('commit', 'N/A')}` |",
        f"| sure_eval git dirty | `{git.get('dirty', 'N/A')}` |",
        f"| Python | `{python_info.get('implementation', 'N/A')} {python_info.get('version', 'N/A')}` |",
        f"| Python executable | `{python_info.get('executable', 'N/A')}` |",
        f"| ffmpeg | `{tools.get('ffmpeg', 'N/A')}` |",
        f"| CUDA_VISIBLE_DEVICES | `{tools.get('cuda_visible_devices', 'N/A')}` |",
    ]
    packages = versions.get("packages") or {}
    for name, version in sorted(packages.items()):
        if version:
            lines.append(f"| Python package `{name}` | `{version}` |")
    return "\n".join(lines)


def _build_validation_summary_block(report_rows: list[dict[str, Any]] | None, validation_payload: dict[str, Any] | None) -> str:
    rows = []
    if report_rows:
        seen = set()
        for row in report_rows:
            dataset_name = row.get("dataset", {}).get("name")
            if not dataset_name or dataset_name in seen:
                continue
            seen.add(dataset_name)
            validation = (row.get("prediction") or {}).get("validation") or {}
            rows.append(
                [
                    dataset_name,
                    validation.get("expected_samples", "N/A"),
                    validation.get("provided_predictions", "N/A"),
                    validation.get("missing", "N/A"),
                    validation.get("extra", "N/A"),
                    validation.get("duplicate", "N/A"),
                    validation.get("empty", "N/A"),
                    validation.get("is_valid", "N/A"),
                ]
            )
    elif validation_payload:
        for result in validation_payload.get("results") or []:
            rows.append(
                [
                    result.get("dataset", "N/A"),
                    result.get("expected_samples", "N/A"),
                    result.get("provided_predictions", "N/A"),
                    len(result.get("missing_keys") or []),
                    len(result.get("extra_keys") or []),
                    len(result.get("duplicate_keys") or []),
                    len(result.get("empty_prediction_keys") or []),
                    result.get("is_valid", "N/A"),
                ]
            )
    if not rows:
        return "- Validation details unavailable."
    lines = ["| Dataset | Expected | Provided | Missing | Extra | Duplicate | Empty | Valid |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _build_tool_signature(config: dict[str, Any] | None, tool_name: str) -> str:
    """Build a text representation of the tool call signature."""
    if not config:
        return f'{tool_name}(audio_path="<audio_path>", language="auto")'

    tools = config.get("tools", [])
    tool = next((t for t in tools if t.get("name") == tool_name), None)
    if tool is None and tools:
        tool = tools[0]
    if not tool:
        return f'{tool_name}(audio_path="<audio_path>", language="auto")'

    input_schema = tool.get("input_schema", {})
    props = input_schema.get("properties", {})
    args: list[str] = []
    for name, schema in props.items():
        default = schema.get("default")
        if default is not None:
            args.append(f'{name}="{default}"')
        else:
            args.append(f'{name}="<{name}>"')
    return f"{tool_name}({', '.join(args)})"


def _find_model_config(model_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load model config.yaml and model.spec.yaml if present."""
    config = _load_yaml(model_dir / "config.yaml")
    spec = _load_yaml(model_dir / "model.spec.yaml")
    return config, spec


def generate_snapshot(run_dir: Path, output_path: Path | None, template_path: Path) -> Path:
    """Generate the report snapshot and return the written path."""
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_paths = _resolve_run_paths(run_dir)
    model_name = run_paths.get("model_name", "N/A")
    run_id = run_paths.get("run_id", "N/A")

    # Load main-flow artifacts
    execution_surface = _load_json(run_dir / "execution_surface.json") or {}
    eval_payload = _load_json(run_dir / "evaluation_payload.json") or {}
    validation_payload = _load_json(run_dir / "validation_payload.json") or {}
    pred_status = _load_json(run_dir / "prediction_generation_status.json") or {}
    run_report = _load_json(run_dir / "main_agent_run_report.json") or {}
    readiness_report = _load_json(run_dir / "execution_readiness_report.json") or {}
    assessment_report = _load_json(run_dir / "assessment_report.json") or {}
    report_rows = _load_jsonl(run_dir / "report.jsonl")

    resolved = execution_surface.get("resolved_inputs", {})
    datasets = resolved.get("datasets") or [row.get("dataset", {}).get("name") for row in report_rows] or [r.get("dataset") for r in eval_payload.get("results", [])]
    datasets = [d for d in datasets if d]

    # Model directory and config
    model_dir_value = resolved.get("model_dir") or run_paths.get("model_dir_absolute") or f"src/sure_eval/models/{model_name}"
    model_dir = Path(model_dir_value)
    if not model_dir.is_absolute():
        model_dir = REPO_ROOT / model_dir
    model_dir_rel = _display_path(model_dir)
    config, spec = _find_model_config(model_dir)
    config = config or {}
    spec = spec or {}

    # Parse run_evaluation.sh for RESULTS_DIR / PROTOCOL_ID
    sh_values = _parse_run_evaluation_sh(run_dir / "run_evaluation.sh")
    protocol_id = sh_values.get("PROTOCOL_ID", "strict_core")
    results_dir_rel = sh_values.get(
        "RESULTS_DIR",
        f"results/{model_name}/{protocol_id}",
    )
    results_dir = REPO_ROOT / results_dir_rel

    # Determine output path
    if output_path is None:
        output_path = run_dir / "report_snapshot.md"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Basic fields
    model_full_name = (
        spec.get("runtime", {}).get("identity")
        or config.get("model", {}).get("id")
        or model_name
    )
    model_source = (
        f"`{model_full_name}` on {spec.get('weights', {}).get('source', 'N/A')}"
        if spec
        else f"`{model_full_name}`"
    )
    task = resolved.get("dataset_task") or (
        eval_payload.get("results", [{}])[0].get("task") if eval_payload else None
    ) or (config.get("task") if config else None) or "N/A"
    tool_name = resolved.get("tool_name") or sh_values.get("TOOL_NAME", "N/A")
    execution_path = (
        run_report.get("execution_path_actual")
        or resolved.get("execution_path")
        or pred_status.get("execution_path")
        or "N/A"
    )
    status = assessment_report.get("status") or run_report.get("final_status") or "N/A"
    device = (
        sh_values.get("DEVICE")
        or (config.get("server", {}).get("env", {}).get("DEVICE") if config else None)
        or readiness_report.get("gpu_preflight", {}).get("recommended_device")
        or "N/A"
    )
    model_cache_path = (
        (config.get("server", {}).get("env", {}).get("MODELSCOPE_CACHE") if config else None)
        or spec.get("weights", {}).get("local_path")
        or "N/A"
    )
    model_weights = spec.get("weights", {}).get("source", "N/A") if spec else "N/A"
    model_dir_source = resolved.get("model_dir_source") or pred_status.get("model_dir_source") or "N/A"

    # Environment fields
    eval_python_env = f"repository `.venv` (`{sys.executable}`)"
    server_python_env = f"model-local `.venv` under `{model_dir_rel}/.venv`"
    python_version = f"{platform.python_implementation()} {platform.python_version()}"

    # Blocks
    tool_signature = _build_tool_signature(config, tool_name)
    dataset_scope = _build_dataset_scope_block(datasets, eval_payload, pred_status, report_rows)
    result_summary = _build_result_summary_block(eval_payload, results_dir, run_dir, report_rows)
    evaluation_pipeline = _build_evaluation_pipeline_block(eval_payload, report_rows)
    output_artifacts = _build_output_artifacts_block(run_dir, results_dir, datasets, report_rows)
    runtime_versions = _build_runtime_versions_block(report_rows)
    validation_summary = _build_validation_summary_block(report_rows, validation_payload)

    test_notes = (
        "- `protocol.yaml` and `report.jsonl` were generated by "
        "`scripts/evaluate_predictions.py` into the model-local run directory.\n"
        "- `report_snapshot.md` is the user-facing evaluation snapshot for this run.\n"
        f"- Model directory source: `{model_dir_source}` (`{model_dir_rel}`).\n"
        "- The `results/` directory is a compatibility mirror; the model-local "
        "run directory is the source of truth."
    )

    # Render template
    template_text = template_path.read_text(encoding="utf-8")
    placeholders = {
        "model_name": _na(model_name),
        "model_full_name": _na(model_full_name),
        "model_source": _na(model_source),
        "task": _na(task),
        "datasets": _na(", ".join(datasets) if datasets else "N/A"),
        "run_id": _na(run_id),
        "run_dir": _na(_display_path(run_dir)),
        "results_dir": _na(_display_path(results_dir)),
        "status": _na(status),
        "dataset_scope": dataset_scope,
        "execution_path": _na(execution_path),
        "eval_python_env": _na(eval_python_env),
        "server_python_env": _na(server_python_env),
        "python_version": _na(python_version),
        "device": _na(device),
        "model_weights": _na(model_weights),
        "model_cache_path": _na(model_cache_path),
        "tool_signature": _na(tool_signature),
        "protocol_id": _na(protocol_id),
        "tool_name": _na(tool_name),
        "result_summary": result_summary,
        "evaluation_pipeline": evaluation_pipeline,
        "runtime_versions": runtime_versions,
        "validation_summary": validation_summary,
        "output_artifacts": output_artifacts,
        "test_notes": test_notes,
    }

    rendered = template_text
    for key, value in placeholders.items():
        rendered = rendered.replace(f"{{{key}}}", value)

    # Replace any remaining unknown placeholders with N/A
    rendered = re.sub(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", "N/A", rendered)

    output_path.write_text(rendered, encoding="utf-8")
    logger.info("Wrote report snapshot", path=str(output_path))
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a user-facing report snapshot for a SURE-EVAL run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the main-flow run directory (e.g., <selected_model_dir>/eval_runs/main_agent_asr_qwen3_002)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; defaults to results/<model>/<protocol>/report_snapshot.md",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"Template path; defaults to {DEFAULT_TEMPLATE}",
    )
    args = parser.parse_args()

    if not args.template.exists():
        logger.error("Template not found", path=str(args.template))
        return 1

    try:
        output_path = generate_snapshot(args.run_dir, args.output, args.template)
    except Exception as exc:
        logger.error("Failed to generate report snapshot", error=str(exc))
        return 1

    print(f"Report snapshot written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
