"""RIR estimation task script route descriptors."""

from __future__ import annotations

from collections.abc import Iterable

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
from sure_eval.evaluation.tasks.rir.types import RIRSample

DEFAULT_METRICS = ("rir_npm", "rt60_mae")


def describe_pipeline(*, metrics: str | list[str] | tuple[str, ...] | None = None, metric: str | None = None, pipeline_id: str | None = None, language: str = "n/a"):
    if metric is not None:
        if metrics is not None:
            raise ValueError("Use either metric or metrics, not both")
        metrics = metric
    manifest, manifest_path = load_task_manifest("rir")
    routes, routes_path = load_task_routes("rir")
    if pipeline_id:
        selected = (find_pipeline_route(routes, pipeline_id=pipeline_id),)
        requested = route_execution_metrics(selected)
    else:
        if isinstance(metrics, str):
            metrics = tuple(item.strip() for item in metrics.split(",") if item.strip())
        requested = normalize_metric_list(metrics, DEFAULT_METRICS)
        selected = tuple(find_metric_route(routes, metric=metric) for metric in requested)
        requested = route_execution_metrics(selected)
    contracts = tuple(contract_from_manifest(manifest, route["input_contract"]) for route in selected)
    members = route_member_pipeline_ids(selected)
    bundle = len(members) > 1
    pipeline = members[0] if not bundle else build_bundle_pipeline_id("rir", "any", members)
    nodes = tuple(dict.fromkeys(node for route in selected for node in route["nodes"]))
    return describe_from_contracts(task="RIR estimation", pipeline_id=pipeline, metric=canonical_metric(requested[0]) if not bundle else "multi", language="n/a", node_ids=nodes, contracts=contracts, task_config_path=manifest_path, route_config_path=routes_path, pipeline_kind="bundle" if bundle else "atomic", member_pipeline_ids=members if bundle else (), computation_node_ids=route_computation_node_ids(selected), execution_metrics=requested, script_module=__name__, executor=str(selected[0]["executor"]))


def run(samples: list[RIRSample], *, output_dir: str, metrics: Iterable[str] | None = None, pipeline_id: str | None = None):
    routes, _routes_path = load_task_routes("rir")
    if pipeline_id:
        selected = (find_pipeline_route(routes, pipeline_id=pipeline_id),)
        requested = route_execution_metrics(selected)
    else:
        if isinstance(metrics, str):
            metrics = tuple(item.strip() for item in metrics.split(",") if item.strip())
        requested = normalize_metric_list(metrics, DEFAULT_METRICS)
        selected = tuple(find_metric_route(routes, metric=metric) for metric in requested)
        requested = route_execution_metrics(selected)
    description = describe_pipeline(
        pipeline_id=pipeline_id if pipeline_id else None,
        metrics=None if pipeline_id else requested,
    )
    report = call_executor_path(str(selected[0]["executor"]), samples=samples, metrics=requested)
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)
