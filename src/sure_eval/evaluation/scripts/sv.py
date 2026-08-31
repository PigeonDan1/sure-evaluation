"""Speaker-verification configured script route descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sure_eval.evaluation.pipeline_identity import build_bundle_pipeline_id, canonical_metric
from sure_eval.evaluation.scripts.contracts import (
    call_executor_path,
    contract_from_manifest,
    describe_from_contracts,
    find_metric_route,
    find_pipeline_route,
    load_task_manifest,
    load_task_routes,
    normalize_metric_list,
    route_computation_node_ids,
    route_execution_metrics,
    route_member_pipeline_ids,
    write_route_run_outputs,
)

DEFAULT_METRICS = ("eer", "min_dcf")


def describe_pipeline(
    *, metrics: str | list[str] | tuple[str, ...] | None = None, pipeline_id: str | None = None
):
    manifest, manifest_path, routes_path, routes, requested_metrics = _select_routes(
        metrics=metrics, pipeline_id=pipeline_id
    )
    return _describe_from_routes(
        manifest=manifest,
        manifest_path=manifest_path,
        routes_path=routes_path,
        selected_routes=routes,
        requested_metrics=requested_metrics,
    )


def run(
    sample_output: str,
    trial_manifest: str,
    *,
    output_dir: str,
    metrics: tuple[str, ...] | list[str] | None = None,
    pipeline_id: str | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    manifest, manifest_path, routes_path, routes, requested_metrics = _select_routes(
        metrics=metrics, pipeline_id=pipeline_id
    )
    description = _describe_from_routes(
        manifest=manifest,
        manifest_path=manifest_path,
        routes_path=routes_path,
        selected_routes=routes,
        requested_metrics=requested_metrics,
    )
    report = call_executor_path(
        _shared_executor_path(routes),
        sample_output=sample_output,
        trial_manifest=trial_manifest,
        metrics=requested_metrics,
        work_dir=Path(output_dir),
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_routes(*, metrics: str | list[str] | tuple[str, ...] | None, pipeline_id: str | None):
    if metrics is not None and pipeline_id:
        raise ValueError("Use either metrics or pipeline_id, not both")
    manifest, manifest_path = load_task_manifest("sv")
    routes_config, routes_path = load_task_routes("sv")
    if pipeline_id:
        route = find_pipeline_route(routes_config, pipeline_id=pipeline_id)
        return manifest, manifest_path, routes_path, (route,), route_execution_metrics((route,))
    if isinstance(metrics, str):
        metrics = tuple(item.strip() for item in metrics.split(",") if item.strip())
    normalized_metrics = tuple(
        _normalize_metric(metric) for metric in normalize_metric_list(metrics, DEFAULT_METRICS)
    )
    selected_routes = tuple(
        find_metric_route(routes_config, metric=metric) for metric in normalized_metrics
    )
    return (
        manifest,
        manifest_path,
        routes_path,
        selected_routes,
        route_execution_metrics(selected_routes),
    )


def _describe_from_routes(
    *,
    manifest: dict[str, Any],
    manifest_path,
    routes_path,
    selected_routes: tuple[dict[str, Any], ...],
    requested_metrics: tuple[str, ...],
):
    node_ids: list[str] = []
    contracts: list[dict[str, Any]] = []
    for route in selected_routes:
        for node_id in route["nodes"]:
            if node_id not in node_ids:
                node_ids.append(node_id)
        contracts.append(contract_from_manifest(manifest, route["input_contract"]))
    member_pipeline_ids = route_member_pipeline_ids(selected_routes)
    pipeline_kind = "atomic" if len(selected_routes) == 1 else "bundle"
    pipeline_id = (
        member_pipeline_ids[0]
        if pipeline_kind == "atomic"
        else build_bundle_pipeline_id("sv", "any", member_pipeline_ids)
    )
    return describe_from_contracts(
        task="SV",
        pipeline_id=pipeline_id,
        metric=canonical_metric(requested_metrics[0]) if pipeline_kind == "atomic" else "multi",
        language="n/a",
        node_ids=tuple(node_ids),
        contracts=tuple(contracts),
        task_config_path=manifest_path,
        route_config_path=routes_path,
        pipeline_kind=pipeline_kind,
        member_pipeline_ids=() if pipeline_kind == "atomic" else member_pipeline_ids,
        computation_node_ids=_dedupe(route_computation_node_ids(selected_routes)),
        execution_metrics=requested_metrics,
        script_module=__name__,
        executor=_shared_executor_path(selected_routes),
    )


def _shared_executor_path(selected_routes: tuple[dict[str, Any], ...]) -> str:
    paths = {str(route["executor"]) for route in selected_routes}
    if len(paths) != 1:
        raise ValueError("SV selected routes must share one task-level executor")
    return paths.pop()


def _normalize_metric(metric: str) -> str:
    normalized = str(metric).strip().lower().replace("-", "_")
    return {"mindcf": "min_dcf"}.get(normalized, normalized)


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)
