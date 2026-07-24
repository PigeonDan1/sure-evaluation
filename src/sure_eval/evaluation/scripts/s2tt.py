"""S2TT configured script route descriptors."""

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
    *, language: str = "zh", metric: str = "bleu", pipeline_id: str | None = None
):
    manifest, manifest_path, routes_path, route, normalized_metric = _select_route(
        language=language, metric=metric, pipeline_id=pipeline_id
    )
    return describe_from_contracts(
        task="S2TT",
        pipeline_id=route["pipeline_id"],
        metric=normalized_metric,
        language=route.get("language") or language,
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
        route_config_path=routes_path,
        computation_node_ids=tuple(route["nodes"]),
        execution_metrics=(normalized_metric,),
        script_module=__name__,
        executor=str(route.get("executor") or ""),
    )


def run(
    ref_file: str,
    hyp_file: str,
    *,
    language: str,
    metric: str = "bleu",
    pipeline_id: str | None = None,
    output_dir: str,
    src_file: str | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    description = describe_pipeline(language=language, metric=metric, pipeline_id=pipeline_id)
    _, _, _, route, normalized_metric = _select_route(
        language=language, metric=metric, pipeline_id=pipeline_id
    )
    report = call_route_executor(
        route,
        ref_file=ref_file,
        hyp_file=hyp_file,
        language=language,
        metric=normalized_metric,
        src_file=src_file,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(*, language: str, metric: str, pipeline_id: str | None = None):
    manifest, manifest_path = load_task_manifest("s2tt")
    normalized_metric = metric.lower()
    routes, routes_path = load_task_routes("s2tt")
    if pipeline_id:
        route = find_pipeline_route(routes, pipeline_id=pipeline_id, language=language)
    else:
        route = find_task_route(routes, language=language, metric=normalized_metric)
    return manifest, manifest_path, routes_path, route, route_execution_metric(route) or normalized_metric
