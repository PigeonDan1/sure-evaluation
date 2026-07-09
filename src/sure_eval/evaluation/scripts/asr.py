"""ASR configured script route descriptors."""

from __future__ import annotations

from sure_eval.evaluation.scripts.contracts import (
    call_route_executor,
    contract_from_manifest,
    describe_from_contracts,
    find_task_route,
    load_task_manifest,
    load_task_routes,
    write_route_run_outputs,
)


def describe_pipeline(*, language: str, metric: str | None = None):
    manifest, manifest_path, route, normalized_metric = _select_route(language=language, metric=metric)
    return describe_from_contracts(
        task="ASR",
        pipeline_id=route["pipeline_id"],
        metric=normalized_metric,
        language=language,
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
    )


def run(ref_file: str, hyp_file: str, *, language: str, metric: str | None = None, output_dir: str):
    """Execute the configured ASR task route.

    The ``output_dir`` argument is required by the script contract even though
    the current in-process pipeline returns its report directly.
    """

    if not output_dir:
        raise ValueError("output_dir is required")
    description = describe_pipeline(language=language, metric=metric)
    _, _, route, _ = _select_route(language=language, metric=metric)
    report = call_route_executor(
        route,
        ref_file=ref_file,
        hyp_file=hyp_file,
        language=language,
        metric=description.metric,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(*, language: str, metric: str | None = None):
    manifest, manifest_path = load_task_manifest("asr")
    normalized_metric = _normalize_metric(language=language, metric=metric or _default_metric(manifest, language))
    routes, _ = load_task_routes("asr")
    route = find_task_route(routes, language=language, metric=normalized_metric)
    return manifest, manifest_path, route, normalized_metric


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
