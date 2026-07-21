"""TSE configured script route descriptors."""

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
from sure_eval.evaluation.tasks.tse.types import TSESample


def _default_metric(language: str) -> str:
    return "si_sdr"


def describe_pipeline(*, language: str, metrics: str | list[str] | tuple[str, ...] | None = None):
    manifest, manifest_path, routes, requested_metrics = _select_routes(language=language, metrics=metrics)
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
        else f"tse.{language}.multi.audio_metric_nodes"
    )
    return describe_from_contracts(
        task="TSE",
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
):
    manifest, manifest_path = load_task_manifest("tse")
    routes_config, _ = load_task_routes("tse")
    requested_metrics = normalize_metric_list(
        metrics,
        (manifest.get("default_metrics", {}).get(language) or _default_metric(language),),
    )
    selected_routes: list[dict[str, Any]] = []

    for metric in requested_metrics:
        selected_routes.append(find_metric_route(routes_config, metric=metric, language=language))

    return manifest, manifest_path, tuple(selected_routes), requested_metrics


def run(
    samples: list[TSESample],
    *,
    output_dir: str,
    metrics: Iterable[str] | None = None,
    transcribers: Mapping[str, Any] | None = None,
    speaker_providers: Mapping[str, Any] | None = None,
    mos_providers: Mapping[str, Any] | None = None,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    language = _common_language([sample.language for sample in samples])
    requested_metrics = tuple(metric.lower() for metric in metrics) if metrics is not None else None
    manifest, manifest_path, selected_routes, normalized_metrics = _select_routes(
        language=language,
        metrics=requested_metrics,
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
        transcribers=_default_transcribers(
            language=language,
            metrics=normalized_metrics,
            transcribers=transcribers,
            device=device,
            cache_dir=cache_dir,
        ),
        speaker_providers=speaker_providers,
        mos_providers=mos_providers,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _shared_executor_path(selected_routes: tuple[dict[str, Any], ...]) -> str:
    executor_paths = {str(route["executor"]) for route in selected_routes}
    if len(executor_paths) != 1:
        raise ValueError("TSE selected routes must share one task-level executor")
    return executor_paths.pop()


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _default_transcribers(
    *,
    language: str,
    metrics: tuple[str, ...],
    transcribers: Mapping[str, Any] | None,
    device: str,
    cache_dir: str | Path | None,
) -> Mapping[str, Any] | None:
    if transcribers is not None or not (set(metrics) & {"tse_wer", "tse_cer"}):
        return transcribers

    if language.lower().startswith(("zh", "cmn", "yue")):
        from sure_eval.evaluation.nodes.transcription.paraformer_zh.node import DEFAULT_CACHE_DIR as DEFAULT_PARAFORMER_CACHE_DIR
        from sure_eval.evaluation.nodes.transcription.common.providers import ParaformerZHTranscriber

        cache_path = Path(cache_dir) if cache_dir is not None else DEFAULT_PARAFORMER_CACHE_DIR
        semantic_cache = cache_path / "semantic" if cache_dir is not None else cache_path
        return {"zh": ParaformerZHTranscriber(device=device, cache_dir=semantic_cache)}

    from sure_eval.evaluation.nodes.transcription.whisper_large_v3.node import DEFAULT_CACHE_DIR as DEFAULT_WHISPER_CACHE_DIR
    from sure_eval.evaluation.nodes.transcription.common.providers import WhisperLargeV3Transcriber

    cache_path = Path(cache_dir) if cache_dir is not None else DEFAULT_WHISPER_CACHE_DIR
    semantic_cache = cache_path / "semantic" if cache_dir is not None else cache_path
    return {"en": WhisperLargeV3Transcriber(device=device, cache_dir=semantic_cache)}


def _common_language(languages: list[str]) -> str:
    unique = {language for language in languages if language}
    if len(unique) != 1:
        raise ValueError("TSE script route requires one language per call")
    return unique.pop()