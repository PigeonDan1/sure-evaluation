"""S2TT configured script route descriptors."""

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


def describe_pipeline(*, language: str = "zh", metric: str = "bleu"):
    manifest, manifest_path, route, normalized_metric = _select_route(language=language, metric=metric)
    return describe_from_contracts(
        task="S2TT",
        pipeline_id=route["pipeline_id"],
        metric=normalized_metric,
        language=language,
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
    )


def run(
    ref_file: str,
    hyp_file: str,
    *,
    language: str,
    metric: str = "bleu",
    output_dir: str,
    src_file: str | None = None,
):
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
        src_file=src_file,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(*, language: str, metric: str):
    manifest, manifest_path = load_task_manifest("s2tt")
    normalized_metric = metric.lower()
    routes, _ = load_task_routes("s2tt")
    route = find_task_route(routes, language=language, metric=normalized_metric)
    return manifest, manifest_path, route, normalized_metric
