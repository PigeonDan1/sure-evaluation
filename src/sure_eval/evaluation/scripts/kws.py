"""KWS configured script route descriptors."""

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
    *, metric: str = "accuracy", input_mode: str = "sure_json", pipeline_id: str | None = None
):
    manifest, manifest_path, routes_path, route, normalized_metric = _select_route(
        metric=metric, input_mode=input_mode, pipeline_id=pipeline_id
    )
    return describe_from_contracts(
        task="KWS",
        pipeline_id=route["pipeline_id"],
        metric=normalized_metric,
        language="n/a",
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
        route_config_path=routes_path,
        computation_node_ids=tuple(route.get("computation_nodes") or route["nodes"]),
        execution_metrics=(normalized_metric,),
        script_module=__name__,
        executor=str(route.get("executor") or ""),
    )


def run(*, output_dir: str, **kwargs):
    if not output_dir:
        raise ValueError("output_dir is required")
    input_mode = _infer_input_mode(kwargs)
    metric = kwargs.pop("metric", "accuracy")
    pipeline_id = kwargs.pop("pipeline_id", None)
    description = describe_pipeline(metric=metric, input_mode=input_mode, pipeline_id=pipeline_id)
    _, _, _, route, normalized_metric = _select_route(
        metric=metric, input_mode=input_mode, pipeline_id=pipeline_id
    )
    report = call_route_executor(route, metric=normalized_metric, **kwargs)
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(
    *, metric: str = "accuracy", input_mode: str = "sure_json", pipeline_id: str | None = None
):
    manifest, manifest_path = load_task_manifest("kws")
    requested_metric = metric.lower().strip()
    canonical_metric = requested_metric.replace("-", "_")
    routes, routes_path = load_task_routes("kws")
    if pipeline_id:
        route = find_pipeline_route(routes, pipeline_id=pipeline_id)
        return manifest, manifest_path, routes_path, route, route_execution_metric(route)
    metric_aliases = {
        str(alias).lower()
        for route in (manifest.get("routes") or {}).values()
        for alias in (route or {}).get("aliases", ())
    }
    if canonical_metric not in set(manifest["metrics"]) and requested_metric not in metric_aliases:
        raise ValueError(f"Unsupported KWS metric: {metric}")
    route = find_task_route(routes, metric=canonical_metric, input_mode=input_mode)
    return manifest, manifest_path, routes_path, route, route_execution_metric(route) or canonical_metric


def _infer_input_mode(kwargs: dict) -> str:
    if kwargs.get("reference_jsonl") and kwargs.get("sample_output"):
        return "sure_json"
    if kwargs.get("wekws_label_file") and kwargs.get("wekws_score_file") and kwargs.get("keyword"):
        return "wekws_score_ctc"
    if kwargs.get("wekws_label_file") and kwargs.get("wekws_frame_score_file") and kwargs.get("keyword"):
        return "wekws_frame_score"
    return "sure_json"
