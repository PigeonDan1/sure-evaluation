"""ASR configured script route descriptors."""

from __future__ import annotations

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


def describe_pipeline(
    *, language: str | None = None, metric: str | None = None, pipeline_id: str | None = None
):
    manifest, manifest_path, routes_path, route, executor_metric = _select_route(
        language=language, metric=metric, pipeline_id=pipeline_id
    )
    route_language = route.get("language") or language or "n/a"
    return describe_from_contracts(
        task="ASR",
        pipeline_id=route["pipeline_id"],
        metric=route["metric"],
        language=route_language,
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
        route_config_path=routes_path,
        computation_node_ids=tuple(route["nodes"]),
        execution_metrics=(str(route["metric"]),),
        script_module=__name__,
        executor=str(route.get("executor") or ""),
    )


def run(
    ref_file: str,
    hyp_file: str,
    *,
    language: str | None = None,
    metric: str | None = None,
    pipeline_id: str | None = None,
    output_dir: str,
):
    """Execute the configured ASR task route.

    The ``output_dir`` argument is required by the script contract even though
    the current in-process pipeline returns its report directly.
    """

    if not output_dir:
        raise ValueError("output_dir is required")
    description = describe_pipeline(language=language, metric=metric, pipeline_id=pipeline_id)
    _, _, _, route, executor_metric = _select_route(
        language=language, metric=metric, pipeline_id=pipeline_id
    )
    report = call_route_executor(
        route,
        ref_file=ref_file,
        hyp_file=hyp_file,
        language=route.get("language") or language or description.language,
        metric=executor_metric,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(
    *, language: str | None = None, metric: str | None = None, pipeline_id: str | None = None
):
    manifest, manifest_path = load_task_manifest("asr")
    routes, routes_path = load_task_routes("asr")
    if pipeline_id:
        route = find_pipeline_route(routes, pipeline_id=pipeline_id, language=language)
        return manifest, manifest_path, routes_path, route, route_execution_metric(route)
    route_language = language or "zh"
    normalized_metric = _normalize_metric(
        language=route_language, metric=metric or _default_metric(manifest, route_language)
    )
    route = find_task_route(routes, language=route_language, metric=normalized_metric)
    return manifest, manifest_path, routes_path, route, route_execution_metric(route) or normalized_metric


def _default_metric(manifest: dict, language: str) -> str:
    try:
        return manifest["default_metrics"][language]
    except KeyError as exc:
        raise ValueError(f"Unsupported ASR language: {language}") from exc


def _normalize_metric(*, language: str, metric: str) -> str:
    normalized = metric.lower()
    if language == "zh" and normalized == "wer":
        return "cer"
    if language == "cs" and normalized in {"wer", "cer"}:
        return "mer"
    return normalized
