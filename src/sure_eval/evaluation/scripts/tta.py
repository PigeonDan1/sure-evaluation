"""TTA configured script route descriptors."""

from __future__ import annotations
from collections.abc import Iterable, Mapping
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
from sure_eval.evaluation.tasks.tta.pipeline import DEFAULT_METRICS
from sure_eval.evaluation.tasks.tta.types import TTASample


def describe_pipeline(
    *,
    metrics: str | list[str] | tuple[str, ...] | None = None,
    language: str = "n/a",
    pipeline_id: str | None = None,
):
    manifest, manifest_path, routes_path, routes, requested = _select(
        metrics=metrics, pipeline_id=pipeline_id
    )
    return _describe(manifest, manifest_path, routes_path, routes, requested)


def _describe(manifest, manifest_path, routes_path, routes, requested):
    node_ids, contracts = [], []
    for route in routes:
        node_ids.extend(route["nodes"])
        contracts.append(contract_from_manifest(manifest, route["input_contract"]))
    members = route_member_pipeline_ids(routes)
    kind = "atomic" if len(routes) == 1 else "bundle"
    pipeline_id = (
        members[0] if kind == "atomic" else build_bundle_pipeline_id("tta", "any", members)
    )
    return describe_from_contracts(
        task="TTA",
        pipeline_id=pipeline_id,
        metric=canonical_metric(requested[0]) if len(requested) == 1 else "multi",
        language="n/a",
        node_ids=tuple(dict.fromkeys(node_ids)),
        contracts=tuple(contracts),
        task_config_path=manifest_path,
        route_config_path=routes_path,
        pipeline_kind=kind,
        member_pipeline_ids=() if kind == "atomic" else members,
        computation_node_ids=route_computation_node_ids(routes),
        execution_metrics=requested,
        script_module=__name__,
        executor=_executor(routes),
    )


def _select(*, metrics=None, pipeline_id=None):
    if metrics is not None and pipeline_id:
        raise ValueError("Use either metrics or pipeline_id, not both")
    manifest, manifest_path = load_task_manifest("tta")
    config, routes_path = load_task_routes("tta")
    if pipeline_id:
        route = find_pipeline_route(config, pipeline_id=pipeline_id)
        return manifest, manifest_path, routes_path, (route,), route_execution_metrics((route,))
    if isinstance(metrics, str):
        metrics = tuple(x.strip() for x in metrics.split(",") if x.strip())
    requested = tuple(_norm(x) for x in normalize_metric_list(metrics, DEFAULT_METRICS))
    routes = tuple(find_metric_route(config, metric=x) for x in requested)
    return manifest, manifest_path, routes_path, routes, route_execution_metrics(routes)


def run(
    samples: list[TTASample],
    *,
    output_dir: str,
    metrics: Iterable[str] | None = None,
    providers: Mapping[str, Any] | None = None,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
    pipeline_id: str | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    requested = tuple(_norm(x) for x in metrics) if metrics else None
    manifest, manifest_path, routes_path, routes, normalized = _select(
        metrics=requested, pipeline_id=pipeline_id
    )
    description = _describe(manifest, manifest_path, routes_path, routes, normalized)
    if providers is None:
        from sure_eval.evaluation.audio_runtime import build_tta_runtime

        providers = build_tta_runtime(metrics=normalized, device=device, cache_dir=cache_dir)
    report = call_executor_path(
        _executor(routes), samples=samples, metrics=normalized, providers=providers
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _executor(routes):
    values = {str(route["executor"]) for route in routes}
    if len(values) != 1:
        raise ValueError("TTA selected routes must share one task-level executor")
    return values.pop()


def _norm(metric):
    value = str(metric).strip().lower().replace("-", "_")
    return {
        "inception_score": "is",
        "clap": "clap_score",
        "ce": "audiobox_ce",
        "cu": "audiobox_cu",
        "pc": "audiobox_pc",
        "pq": "audiobox_pq",
    }.get(value, value)
