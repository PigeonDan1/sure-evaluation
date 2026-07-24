"""SA-ASR configured script route descriptors."""

from __future__ import annotations

from pathlib import Path

from sure_eval.evaluation.scripts.contracts import (
    call_route_executor,
    contract_from_manifest,
    conversion_description,
    describe_from_contracts,
    find_pipeline_route,
    find_task_route,
    load_task_manifest,
    load_task_routes,
    route_execution_metric,
    write_route_run_outputs,
)


def describe_pipeline(
    *, metric: str = "cpwer", language: str = "en", pipeline_id: str | None = None
):
    manifest, manifest_path, routes_path, route, normalized_metric = _select_route(
        metric=metric, pipeline_id=pipeline_id
    )
    return describe_from_contracts(
        task="SA-ASR",
        pipeline_id=route["pipeline_id"],
        metric=normalized_metric,
        language=language,
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
        route_config_path=routes_path,
        conversion_steps=(conversion_description("sa_asr__cpwer"),),
        computation_node_ids=("conversion/sa_asr__cpwer", *tuple(route["nodes"])),
        execution_metrics=(normalized_metric,),
        script_module=__name__,
        executor=str(route.get("executor") or ""),
    )


def run(
    ref_file: str,
    hyp_file: str,
    *,
    output_dir: str,
    metric: str = "cpwer",
    language: str = "en",
    pipeline_id: str | None = None,
    collar: float | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    description = describe_pipeline(metric=metric, language=language, pipeline_id=pipeline_id)
    _, _, _, route, normalized_metric = _select_route(metric=metric, pipeline_id=pipeline_id)
    params = dict(route.get("params") or {})
    if collar is not None:
        params["collar"] = collar
    report = call_route_executor(
        route,
        ref_file=ref_file,
        hyp_file=hyp_file,
        metric=normalized_metric,
        language=language,
        conversion_output_dir=str(Path(output_dir) / "conversion" / "sa_asr__cpwer"),
        **params,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(*, metric: str = "cpwer", pipeline_id: str | None = None):
    manifest, manifest_path = load_task_manifest("sa_asr")
    normalized_metric = metric.lower().replace("-", "_")
    routes, routes_path = load_task_routes("sa_asr")
    if pipeline_id:
        route = find_pipeline_route(routes, pipeline_id=pipeline_id)
    else:
        route = find_task_route(routes, metric=normalized_metric)
    return manifest, manifest_path, routes_path, route, route_execution_metric(route) or normalized_metric
