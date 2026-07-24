"""Shared manifest and input-contract helpers for task scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from sure_eval.evaluation.core.types import EvaluationReport, PipelineNodeResult
from sure_eval.evaluation.pipeline_identity import canonical_metric

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALUATION_ROOT.parents[2]
TASKS_ROOT = EVALUATION_ROOT / "tasks"
NODES_ROOT = EVALUATION_ROOT / "nodes"
CONVERSION_ROOT = EVALUATION_ROOT / "conversion"
REPORT_FILENAME = "report.json"
PIPELINE_DESCRIPTION_FILENAME = "pipeline_description.json"

NODE_MANIFEST_ALIASES = {
    "scoring/wenet_cer": "scoring/wenet_wer",
    "scoring/wenet_mer": "scoring/wenet_wer",
}


@dataclass(frozen=True)
class PipelineDescription:
    """Declarative view of one configured evaluation pipeline."""

    task: str
    pipeline_id: str
    metric: str
    language: str
    node_ids: tuple[str, ...]
    required_roles: tuple[str, ...]
    optional_roles: tuple[str, ...] = ()
    output_dir_required: bool = True
    contracts: tuple[dict[str, Any], ...] = ()
    task_config_path: str = ""
    route_config_path: str = ""
    node_config_paths: tuple[str, ...] = ()
    nodes: tuple[dict[str, Any], ...] = ()
    conversion_steps: tuple[dict[str, Any], ...] = ()
    pipeline_kind: str = "atomic"
    member_pipeline_ids: tuple[str, ...] = ()
    computation_node_ids: tuple[str, ...] = ()
    execution_metrics: tuple[str, ...] = ()
    describe_entrypoint: str = ""
    script_entrypoint: str = ""
    executor: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "pipeline_id": self.pipeline_id,
            "pipeline_kind": self.pipeline_kind,
            "metric": self.metric,
            "language": self.language,
            "node_ids": list(self.node_ids),
            "computation_node_ids": list(self.computation_node_ids),
            "member_pipeline_ids": list(self.member_pipeline_ids),
            "execution_metrics": list(self.execution_metrics),
            "required_roles": list(self.required_roles),
            "optional_roles": list(self.optional_roles),
            "output_dir_required": self.output_dir_required,
            "contracts": list(self.contracts),
            "task_config_path": self.task_config_path,
            "route_config_path": self.route_config_path,
            "node_config_paths": list(self.node_config_paths),
            "nodes": list(self.nodes),
            "conversion_steps": list(self.conversion_steps),
            "describe_entrypoint": self.describe_entrypoint,
            "script_entrypoint": self.script_entrypoint,
            "executor": self.executor,
        }


def load_task_manifest(task: str) -> tuple[dict[str, Any], Path]:
    path = TASKS_ROOT / task.lower() / "manifest.yaml"
    return load_yaml(path), path


def load_task_routes(task: str) -> tuple[dict[str, Any], Path]:
    path = TASKS_ROOT / task.lower() / "routes.yaml"
    return load_yaml(path), path


def find_task_route(
    routes: dict[str, Any],
    *,
    language: str | None = None,
    metric: str | None = None,
    **selectors: Any,
) -> dict[str, Any]:
    normalized_metric = metric.lower() if metric else None
    for route in routes.get("routes") or ():
        if language is not None and route.get("language") != language:
            continue
        if normalized_metric is not None and not _route_matches_metric(route, normalized_metric):
            continue
        if any(route.get(key) != value for key, value in selectors.items() if value is not None):
            continue
        return dict(route)
    details = []
    if language is not None:
        details.append(f"language={language}")
    if normalized_metric is not None:
        details.append(f"metric={normalized_metric}")
    for key, value in selectors.items():
        if value is not None:
            details.append(f"{key}={value}")
    suffix = ", ".join(details) if details else "default route"
    raise ValueError(f"No configured route found for {routes.get('task', 'task')} ({suffix})")


def find_metric_route(
    routes: dict[str, Any],
    *,
    metric: str,
    language: str | None = None,
    **selectors: Any,
) -> dict[str, Any]:
    try:
        return find_task_route(routes, language=language, metric=metric, **selectors)
    except ValueError:
        return find_task_route(routes, metric=metric, **selectors)


def find_pipeline_route(
    routes: dict[str, Any],
    *,
    pipeline_id: str,
    language: str | None = None,
) -> dict[str, Any]:
    """Return the configured route with a concrete pipeline id."""

    requested_pipeline_id = str(pipeline_id)
    for route in routes.get("routes") or ():
        route_language = language or route.get("language") or _language_from_pipeline_id(requested_pipeline_id)
        if route_pipeline_id(route, language=route_language) == requested_pipeline_id:
            resolved = dict(route)
            if route_language and not resolved.get("language"):
                resolved["language"] = route_language
            return resolved
    raise ValueError(f"No configured route found for {routes.get('task', 'task')} (pipeline_id={pipeline_id})")


def route_execution_metric(route: dict[str, Any]) -> str:
    """Return the executor-facing metric token declared by a route."""

    return str(route.get("internal_executor_metric") or route.get("executor_metric") or route.get("metric") or "")


def route_execution_metrics(routes: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    return tuple(route_execution_metric(route) for route in routes)


def route_pipeline_id(route: dict[str, Any], *, language: str | None = None) -> str:
    """Return the concrete pipeline id declared by a route."""

    pipeline_id = route.get("pipeline_id")
    if not pipeline_id:
        raise KeyError("Route is missing required field: pipeline_id")
    return str(pipeline_id).format(language=language or route.get("language", "n/a"))


def route_member_pipeline_ids(
    routes: tuple[dict[str, Any], ...],
    *,
    language: str | None = None,
) -> tuple[str, ...]:
    return tuple(route_pipeline_id(route, language=language) for route in routes)


def route_computation_node_ids(
    routes: tuple[dict[str, Any], ...],
    *,
    conversion_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    nodes: list[str] = [f"conversion/{conversion_id}" for conversion_id in conversion_ids]
    for route in routes:
        nodes.extend(str(node_id) for node_id in route.get("nodes") or ())
    return tuple(nodes)


def call_route_executor(route: dict[str, Any], **kwargs: Any) -> EvaluationReport:
    """Load and call the executor declared by a route.

    TTS and VC routes share task-level multi-metric executors. In those cases
    route selection describes and validates the requested metrics, while the
    executor receives the normalized metric list directly.
    """

    return _load_route_executor(route)(**kwargs)


def call_executor_path(executor_path: str, **kwargs: Any) -> EvaluationReport:
    """Load and call a task executor by dotted path."""

    return _load_executor_path(executor_path)(**kwargs)


def assert_report_matches_description(
    report: EvaluationReport,
    description: PipelineDescription,
) -> None:
    """Reject runs whose actual pipeline diverges from the selected description."""

    if report.pipeline_id != description.pipeline_id:
        raise ValueError(
            "pipeline_id mismatch: "
            f"route expected {description.pipeline_id!r}, executor returned {report.pipeline_id!r}"
        )
    if report.metric != description.metric:
        raise ValueError(
            "metric mismatch: "
            f"description expected {description.metric!r}, executor returned {report.metric!r}"
        )
    if report.pipeline_kind != description.pipeline_kind:
        raise ValueError(
            "pipeline_kind mismatch: "
            f"route expected {description.pipeline_kind!r}, executor returned {report.pipeline_kind!r}"
        )
    if tuple(report.member_pipeline_ids) != tuple(description.member_pipeline_ids):
        raise ValueError(
            "member_pipeline_ids mismatch: "
            f"route expected {tuple(description.member_pipeline_ids)!r}, "
            f"executor returned {tuple(report.member_pipeline_ids)!r}"
        )
    if tuple(report.computation_node_ids) != tuple(description.computation_node_ids):
        raise ValueError(
            "computation_node_ids mismatch: "
            f"description expected {tuple(description.computation_node_ids)!r}, "
            f"executor returned {tuple(report.computation_node_ids)!r}"
        )


def write_route_run_outputs(
    *,
    report: EvaluationReport,
    description: PipelineDescription,
    output_dir: str | Path,
) -> EvaluationReport:
    assert_report_matches_description(report, description)
    return write_run_outputs(report=report, description=description, output_dir=output_dir)


def _load_route_executor(route: dict[str, Any]):
    executor_path = route.get("executor")
    if not executor_path:
        raise KeyError("Route is missing required field: executor")
    return _load_executor_path(str(executor_path))


def _load_executor_path(executor_path: str):
    module_name, separator, function_name = str(executor_path).rpartition(".")
    if not separator or not module_name or not function_name:
        raise ValueError(f"Invalid route executor path: {executor_path!r}")
    module = import_module(module_name)
    executor = getattr(module, function_name)
    if not callable(executor):
        raise TypeError(f"Route executor is not callable: {executor_path!r}")
    return executor


def load_node_manifest(node_id: str) -> tuple[dict[str, Any], Path]:
    manifest_id = NODE_MANIFEST_ALIASES.get(node_id, node_id)
    stage, name = manifest_id.split("/", 1)
    path = NODES_ROOT / stage / name / "manifest.yaml"
    return load_yaml(path), path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def contract_from_manifest(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    contracts = manifest.get("input_contracts") or {}
    if key not in contracts:
        raise KeyError(f"Input contract {key!r} not found in task manifest")
    contract = dict(contracts[key] or {})
    contract.setdefault("metric_id", key)
    return contract


def describe_from_contracts(
    *,
    task: str,
    pipeline_id: str,
    metric: str,
    language: str,
    node_ids: tuple[str, ...],
    contracts: tuple[dict[str, Any], ...],
    task_config_path: Path,
    route_config_path: Path | None = None,
    conversion_steps: tuple[dict[str, Any], ...] = (),
    pipeline_kind: str = "atomic",
    member_pipeline_ids: tuple[str, ...] = (),
    computation_node_ids: tuple[str, ...] = (),
    execution_metrics: tuple[str, ...] = (),
    script_module: str = "",
    executor: str = "",
) -> PipelineDescription:
    required_roles, optional_roles = merge_input_roles(contracts)
    nodes = tuple(node_description(node_id) for node_id in node_ids)
    node_paths = tuple(node["manifest_path"] for node in nodes)
    return PipelineDescription(
        task=task,
        pipeline_id=pipeline_id,
        metric=canonical_metric(metric),
        language=language,
        node_ids=node_ids,
        required_roles=required_roles,
        optional_roles=optional_roles,
        contracts=contracts,
        task_config_path=_display_path(task_config_path),
        route_config_path=_display_path(route_config_path) if route_config_path else "",
        node_config_paths=node_paths,
        nodes=nodes,
        conversion_steps=conversion_steps,
        pipeline_kind=pipeline_kind,
        member_pipeline_ids=member_pipeline_ids,
        computation_node_ids=computation_node_ids or node_ids,
        execution_metrics=execution_metrics,
        describe_entrypoint=f"{script_module}.describe_pipeline" if script_module else "",
        script_entrypoint=f"{script_module}.run" if script_module else "",
        executor=executor,
    )


def node_description(node_id: str) -> dict[str, Any]:
    manifest, path = load_node_manifest(node_id)
    return {
        "node_id": node_id,
        "stage": manifest.get("stage", node_id.split("/", 1)[0]),
        "version": manifest.get("version", "unknown"),
        "manifest_path": _display_path(path),
    }


def conversion_description(conversion_id: str) -> dict[str, Any]:
    path = CONVERSION_ROOT / conversion_id / "manifest.yaml"
    manifest = load_yaml(path)
    script = CONVERSION_ROOT / conversion_id / "convert.py"
    description = {
        "id": conversion_id,
        "affects_metric": bool(manifest.get("affects_metric", True)),
        "manifest_path": _display_path(path),
        "script": _display_path(script),
    }
    for key in ("task", "metric", "source_format", "target_format"):
        if manifest.get(key) is not None:
            description[key] = manifest[key]
    if manifest.get("model"):
        description["model"] = manifest["model"]
    return description


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _language_from_pipeline_id(pipeline_id: str) -> str | None:
    parts = str(pipeline_id).split(".")
    if len(parts) > 1 and parts[1]:
        return parts[1]
    return None


def merge_input_roles(contracts: tuple[dict[str, Any], ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required: list[str] = []
    optional: list[str] = []
    for contract in contracts:
        for role in contract.get("required_roles") or ():
            if role not in required:
                required.append(role)
            if role in optional:
                optional.remove(role)
        for role in contract.get("optional_roles") or ():
            if role not in required and role not in optional:
                optional.append(role)
    return tuple(required), tuple(optional)


def normalize_metric_list(metrics: str | list[str] | tuple[str, ...] | None, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if metrics is None:
        return defaults
    if isinstance(metrics, str):
        return (metrics.lower(),)
    return tuple(metric.lower() for metric in metrics)


def write_run_outputs(
    *,
    report: EvaluationReport,
    description: PipelineDescription,
    output_dir: str | Path,
) -> EvaluationReport:
    if not output_dir:
        raise ValueError("output_dir is required")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(output_path / REPORT_FILENAME, evaluation_report_as_dict(report))
    _write_json(output_path / PIPELINE_DESCRIPTION_FILENAME, description.as_dict())
    return report


def evaluation_report_as_dict(report: EvaluationReport) -> dict[str, Any]:
    return {
        "task": report.task,
        "language": report.language,
        "metric": report.metric,
        "score": _json_safe(report.score),
        "pipeline_id": report.pipeline_id,
        "pipeline_kind": report.pipeline_kind,
        "member_pipeline_ids": list(report.member_pipeline_ids),
        "computation_node_ids": list(report.computation_node_ids),
        "pipeline_trace": [_pipeline_node_result_as_dict(node) for node in report.pipeline_trace],
        "input_contract": report.input_contract.as_dict() if report.input_contract else None,
        "input_files": report.input_files.as_dict() if report.input_files else None,
        "details": _json_safe(report.details),
    }


def _pipeline_node_result_as_dict(result: PipelineNodeResult) -> dict[str, Any]:
    return {
        "stage": result.stage,
        "node_id": result.node_id,
        "version": result.version,
        "details": _json_safe(result.details),
        "internal_stages": list(result.internal_stages),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _route_matches_metric(route: dict[str, Any], metric: str) -> bool:
    tokens = [str(route.get("metric", "")).lower()]
    tokens.extend(str(alias).lower() for alias in route.get("aliases") or ())
    executor_metric = route.get("executor_metric")
    if executor_metric:
        tokens.append(str(executor_metric).lower())
    return metric in tokens
