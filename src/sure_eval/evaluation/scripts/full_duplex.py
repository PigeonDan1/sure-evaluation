"""Configured full-duplex timing route descriptions and execution."""

from __future__ import annotations

from sure_eval.evaluation.pipeline_identity import canonical_metric
from sure_eval.evaluation.scripts.contracts import (
    call_route_executor,
    contract_from_manifest,
    describe_from_contracts,
    find_pipeline_route,
    find_task_route,
    load_task_manifest,
    load_task_routes,
    route_execution_metric,
    write_route_run_outputs,
)


def describe_pipeline(*, metric: str = "ird", pipeline_id: str | None = None):
    manifest, manifest_path, routes_path, route, normalized_metric = _select_route(
        metric=metric, pipeline_id=pipeline_id
    )
    return describe_from_contracts(
        task="FULL_DUPLEX",
        pipeline_id=route["pipeline_id"],
        metric=normalized_metric,
        language="n/a",
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
        route_config_path=routes_path,
        computation_node_ids=tuple(route.get("computation_nodes") or route["nodes"]),
        execution_metrics=(normalized_metric,),
        script_module=__name__,
        executor=str(route.get("executor") or ""),
    )


def run(*, output_dir: str, **kwargs):
    if not output_dir:
        raise ValueError("output_dir is required")
    metric = kwargs.pop("metric", "ird")
    pipeline_id = kwargs.pop("pipeline_id", None)
    description = describe_pipeline(metric=metric, pipeline_id=pipeline_id)
    _, _, _, route, normalized_metric = _select_route(metric=metric, pipeline_id=pipeline_id)
    report = call_route_executor(
        route,
        metric=normalized_metric,
        protocol=str(route["protocol"]),
        **kwargs,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(*, metric: str = "ird", pipeline_id: str | None = None):
    manifest, manifest_path = load_task_manifest("full_duplex")
    routes, routes_path = load_task_routes("full_duplex")
    if pipeline_id:
        route = find_pipeline_route(routes, pipeline_id=pipeline_id)
        return manifest, manifest_path, routes_path, route, route_execution_metric(route)
    requested_metric = canonical_metric(metric)
    if requested_metric not in set(manifest["metrics"]):
        raise ValueError(f"Unsupported FULL_DUPLEX metric: {metric}")
    route = find_task_route(routes, metric=requested_metric)
    return manifest, manifest_path, routes_path, route, route_execution_metric(route)
