"""VC configured script route descriptors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
from sure_eval.evaluation.tasks.vc.types import VCSample


def describe_pipeline(
    *,
    language: str,
    metrics: str | list[str] | tuple[str, ...] | None = None,
    reference_mode: str = "text",
):
    manifest, manifest_path, routes, requested_metrics = _select_routes(
        language=language,
        metrics=metrics,
        reference_mode=reference_mode,
    )
    return _describe_from_routes(
        language=language,
        manifest=manifest,
        manifest_path=manifest_path,
        selected_routes=routes,
        requested_metrics=requested_metrics,
    )


def _describe_from_routes(
    *,
    language: str,
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
        route_pipeline_id(selected_routes[0], language=language)
        if len(selected_routes) == 1
        else f"vc.{language}.multi.audio_metric_nodes"
    )
    return describe_from_contracts(
        task="VC",
        pipeline_id=pipeline_id,
        metric=pipeline_metric,
        language=language,
        node_ids=_dedupe(node_ids),
        contracts=tuple(contracts),
        task_config_path=manifest_path,
    )


def _select_routes(
    *,
    language: str,
    metrics: str | list[str] | tuple[str, ...] | None = None,
    reference_mode: str = "text",
):
    manifest, manifest_path = load_task_manifest("vc")
    routes_config, _ = load_task_routes("vc")
    requested_metrics = normalize_metric_list(metrics, (manifest["default_metrics"][language],))
    selected_routes: list[dict[str, Any]] = []

    for metric in requested_metrics:
        selectors = {"reference_mode": reference_mode} if metric in {"vc_wer", "vc_cer"} else {}
        selected_routes.append(find_metric_route(routes_config, metric=metric, language=language, **selectors))

    return manifest, manifest_path, tuple(selected_routes), requested_metrics


def run(
    samples: list[VCSample],
    *,
    output_dir: str,
    metrics: Iterable[str] | None = None,
    transcribers: Mapping[str, Any] | None = None,
    speaker_providers: Mapping[str, Any] | None = None,
    mos_providers: Mapping[str, Any] | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    language = _common_language([sample.language for sample in samples])
    requested_metrics = tuple(metric.lower() for metric in metrics) if metrics is not None else None
    reference_mode = "text" if all(bool(sample.reference_text) for sample in samples) else "audio"
    manifest, manifest_path, selected_routes, normalized_metrics = _select_routes(
        language=language,
        metrics=requested_metrics,
        reference_mode=reference_mode,
    )
    description = _describe_from_routes(
        language=language,
        manifest=manifest,
        manifest_path=manifest_path,
        selected_routes=selected_routes,
        requested_metrics=normalized_metrics,
    )
    report = call_executor_path(
        _shared_executor_path(selected_routes),
        samples=samples,
        metrics=normalized_metrics,
        transcribers=transcribers,
        speaker_providers=speaker_providers,
        mos_providers=mos_providers,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _shared_executor_path(selected_routes: tuple[dict[str, Any], ...]) -> str:
    executor_paths = {str(route["executor"]) for route in selected_routes}
    if len(executor_paths) != 1:
        raise ValueError("VC selected routes must share one task-level executor")
    return executor_paths.pop()


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _common_language(languages: list[str]) -> str:
    unique = {language for language in languages if language}
    if len(unique) != 1:
        raise ValueError("VC script route requires one language per call")
    return unique.pop()
