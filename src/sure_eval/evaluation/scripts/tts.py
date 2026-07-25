"""TTS configured script route descriptors."""

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
from sure_eval.evaluation.tasks.tts.types import TTSSample


def _semantic_metric_for_language(language: str) -> str:
    return "tts_cer" if language.lower().startswith(("zh", "cmn", "yue")) else "tts_wer"


def describe_pipeline(
    *,
    language: str | None = None,
    metrics: str | list[str] | tuple[str, ...] | None = None,
    pipeline_id: str | None = None,
):
    manifest, manifest_path, routes_path, resolved_language, routes, requested_metrics = _select_routes(
        language=language, metrics=metrics, pipeline_id=pipeline_id
    )
    return _describe_from_routes(
        language=resolved_language,
        manifest=manifest,
        manifest_path=manifest_path,
        routes_path=routes_path,
        selected_routes=routes,
        requested_metrics=requested_metrics,
    )


def _describe_from_routes(
    *,
    language: str,
    manifest: dict[str, Any],
    manifest_path,
    routes_path,
    selected_routes: tuple[dict[str, Any], ...],
    requested_metrics: tuple[str, ...],
):
    node_ids: list[str] = []
    contracts: list[dict[str, Any]] = []

    for route in selected_routes:
        node_ids.extend(route["nodes"])
        contracts.append(contract_from_manifest(manifest, route["input_contract"]))

    pipeline_metric = canonical_metric(requested_metrics[0]) if len(requested_metrics) == 1 else "multi"
    member_pipeline_ids = route_member_pipeline_ids(selected_routes, language=language)
    pipeline_kind = "atomic" if len(selected_routes) == 1 else "bundle"
    pipeline_id = (
        member_pipeline_ids[0]
        if pipeline_kind == "atomic"
        else build_bundle_pipeline_id("tts", language, member_pipeline_ids)
    )
    return describe_from_contracts(
        task="TTS",
        pipeline_id=pipeline_id,
        metric=pipeline_metric,
        language=language,
        node_ids=_dedupe(node_ids),
        contracts=tuple(contracts),
        task_config_path=manifest_path,
        route_config_path=routes_path,
        pipeline_kind=pipeline_kind,
        member_pipeline_ids=() if pipeline_kind == "atomic" else member_pipeline_ids,
        computation_node_ids=route_computation_node_ids(selected_routes),
        execution_metrics=requested_metrics,
        script_module=__name__,
        executor=_shared_executor_path(selected_routes),
    )


def _select_routes(
    *,
    language: str | None,
    metrics: str | list[str] | tuple[str, ...] | None = None,
    pipeline_id: str | None = None,
):
    if metrics is not None and pipeline_id:
        raise ValueError("Use either metrics or pipeline_id, not both")
    manifest, manifest_path = load_task_manifest("tts")
    routes_config, routes_path = load_task_routes("tts")
    if pipeline_id:
        route = find_pipeline_route(routes_config, pipeline_id=pipeline_id, language=language)
        resolved_language = str(route.get("language") or _language_from_pipeline_id(pipeline_id) or language or "zh")
        selected_routes = (route,)
        return (
            manifest,
            manifest_path,
            routes_path,
            resolved_language,
            selected_routes,
            route_execution_metrics(selected_routes),
        )

    resolved_language = language or "zh"
    requested_metrics = normalize_metric_list(
        metrics,
        (
            manifest.get("default_metrics", {}).get(resolved_language)
            or _semantic_metric_for_language(resolved_language),
        ),
    )
    selected_routes: list[dict[str, Any]] = []

    for metric in requested_metrics:
        selected_routes.append(find_metric_route(routes_config, metric=metric, language=resolved_language))

    return (
        manifest,
        manifest_path,
        routes_path,
        resolved_language,
        tuple(selected_routes),
        route_execution_metrics(tuple(selected_routes)),
    )


def run(
    samples: list[TTSSample],
    *,
    output_dir: str,
    metrics: Iterable[str] | None = None,
    transcribers: Mapping[str, Any] | None = None,
    speaker_providers: Mapping[str, Any] | None = None,
    mos_providers: Mapping[str, Any] | None = None,
    device: str = "cuda",
    cache_dir: str | Path | None = None,
    pipeline_id: str | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    language = _common_language([sample.language for sample in samples])
    requested_metrics = tuple(metric.lower() for metric in metrics) if metrics is not None else None
    manifest, manifest_path, routes_path, resolved_language, selected_routes, normalized_metrics = _select_routes(
        language=language,
        metrics=requested_metrics,
        pipeline_id=pipeline_id,
    )
    description = _describe_from_routes(
        language=resolved_language,
        manifest=manifest,
        manifest_path=manifest_path,
        routes_path=routes_path,
        selected_routes=selected_routes,
        requested_metrics=normalized_metrics,
    )
    report = call_executor_path(
        _shared_executor_path(selected_routes),
        samples=samples,
        metrics=normalized_metrics,
        semantic_transcription_node=_semantic_transcription_node(selected_routes),
        transcribers=_default_transcribers(
            language=resolved_language,
            metrics=normalized_metrics,
            semantic_transcription_node=_semantic_transcription_node(selected_routes),
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
        raise ValueError("TTS selected routes must share one task-level executor")
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
    semantic_transcription_node: str | None,
    transcribers: Mapping[str, Any] | None,
    device: str,
    cache_dir: str | Path | None,
) -> Mapping[str, Any] | None:
    if transcribers is not None or not (set(metrics) & {"tts_wer", "tts_cer"}):
        return transcribers

    if semantic_transcription_node == "transcription/qwen3_asr_1_7b":
        from sure_eval.evaluation.nodes.transcription.qwen3_asr_1_7b.node import DEFAULT_CACHE_DIR as DEFAULT_QWEN_CACHE_DIR
        from sure_eval.evaluation.nodes.transcription.common.providers import Qwen3ASR17BTranscriber

        cache_path = Path(cache_dir) if cache_dir is not None else DEFAULT_QWEN_CACHE_DIR
        semantic_cache = cache_path / "semantic" if cache_dir is not None else cache_path
        family = "zh" if language.lower().startswith(("zh", "cmn", "yue")) else "en"
        return {family: Qwen3ASR17BTranscriber(device=device, cache_dir=semantic_cache)}

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
        raise ValueError("TTS script route requires one language per call")
    return unique.pop()


def _semantic_transcription_node(selected_routes: tuple[dict[str, Any], ...]) -> str | None:
    for route in selected_routes:
        if route.get("family") != "semantic_error_rate":
            continue
        for node_id in route.get("nodes") or ():
            if str(node_id).startswith("transcription/"):
                return str(node_id)
    return None


def _language_from_pipeline_id(pipeline_id: str) -> str | None:
    parts = str(pipeline_id).split(".")
    if len(parts) > 1 and parts[1]:
        return parts[1]
    return None
