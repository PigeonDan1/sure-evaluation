"""SE configured script route descriptors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from sure_eval.evaluation.scripts.contracts import (
    call_executor_path,
    contract_from_manifest,
    describe_from_contracts,
    find_metric_route,
    load_task_manifest,
    load_task_routes,
    normalize_metric_list,
    route_pipeline_id,
    write_route_run_outputs,
)
from sure_eval.evaluation.tasks.se.types import SESample

DEFAULT_METRICS = ("si-sdr", "stoi", "pesq", "dnsmos", "wv-mos", "utmos")


def describe_pipeline(
    *, metrics: str | list[str] | tuple[str, ...] | None = None, language: str = "n/a"
):
    manifest, manifest_path, routes, requested_metrics = _select_routes(metrics=metrics)
    return _describe_from_routes(
        manifest=manifest,
        manifest_path=manifest_path,
        selected_routes=routes,
        requested_metrics=requested_metrics,
    )


def _describe_from_routes(
    *,
    manifest: dict[str, Any],
    manifest_path,
    selected_routes: tuple[dict[str, Any], ...],
    requested_metrics: tuple[str, ...],
):
    node_ids: list[str] = []
    contracts: list[dict[str, Any]] = []
    for route in selected_routes:
        node_ids.extend(route["nodes"])
        contracts.append(contract_from_manifest(manifest, route["input_contract"]))
    pipeline_metric = requested_metrics[0] if len(requested_metrics) == 1 else "multi"
    pipeline_id = (
        route_pipeline_id(selected_routes[0])
        if len(selected_routes) == 1
        else "se.multi.enhancement_quality_nodes"
    )
    return describe_from_contracts(
        task="SE",
        pipeline_id=pipeline_id,
        metric=pipeline_metric,
        language="n/a",
        node_ids=_dedupe(node_ids),
        contracts=tuple(contracts),
        task_config_path=manifest_path,
    )


def _select_routes(*, metrics: str | list[str] | tuple[str, ...] | None = None):
    manifest, manifest_path = load_task_manifest("se")
    routes_config, _ = load_task_routes("se")
    if isinstance(metrics, str):
        metrics = tuple(item.strip() for item in metrics.split(",") if item.strip())
    requested_metrics = normalize_metric_list(metrics, DEFAULT_METRICS)
    requested_metrics = tuple(_normalize_metric(metric) for metric in requested_metrics)
    selected_routes = [
        find_metric_route(routes_config, metric=metric) for metric in requested_metrics
    ]
    return manifest, manifest_path, tuple(selected_routes), requested_metrics


def run(
    samples: list[SESample],
    *,
    output_dir: str,
    metrics: Iterable[str] | None = None,
    mos_providers: Mapping[str, Any] | None = None,
    reference_providers: Mapping[str, Any] | None = None,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    requested_metrics = (
        tuple(_normalize_metric(metric) for metric in metrics) if metrics is not None else None
    )
    if not requested_metrics:
        requested_metrics = None
    manifest, manifest_path, selected_routes, normalized_metrics = _select_routes(
        metrics=requested_metrics
    )
    description = _describe_from_routes(
        manifest=manifest,
        manifest_path=manifest_path,
        selected_routes=selected_routes,
        requested_metrics=normalized_metrics,
    )
    report = call_executor_path(
        _shared_executor_path(selected_routes),
        samples=samples,
        metrics=normalized_metrics,
        mos_providers=mos_providers,
        reference_providers=reference_providers,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _shared_executor_path(selected_routes: tuple[dict[str, Any], ...]) -> str:
    executor_paths = {str(route["executor"]) for route in selected_routes}
    if len(executor_paths) != 1:
        raise ValueError("SE selected routes must share one task-level executor")
    return executor_paths.pop()


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _normalize_metric(metric: str) -> str:
    normalized = str(metric).strip().lower().replace("_", "-")
    return {
        "sisdr": "si-sdr",
        "si-sdr": "si-sdr",
        "wvmos": "wv-mos",
        "wv-mos": "wv-mos",
    }.get(normalized, normalized)
