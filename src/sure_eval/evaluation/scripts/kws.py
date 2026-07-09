"""KWS configured script route descriptors."""

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


def describe_pipeline(*, metric: str = "accuracy", input_mode: str = "sure_json"):
    manifest, manifest_path, route, normalized_metric = _select_route(metric=metric, input_mode=input_mode)
    return describe_from_contracts(
        task="KWS",
        pipeline_id=route["pipeline_id"],
        metric=normalized_metric,
        language="n/a",
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
    )


def run(*, output_dir: str, **kwargs):
    if not output_dir:
        raise ValueError("output_dir is required")
    input_mode = _infer_input_mode(kwargs)
    metric = kwargs.pop("metric", "accuracy")
    description = describe_pipeline(metric=metric, input_mode=input_mode)
    _, _, route, _ = _select_route(metric=metric, input_mode=input_mode)
    report = call_route_executor(route, **kwargs)
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(*, metric: str = "accuracy", input_mode: str = "sure_json"):
    manifest, manifest_path = load_task_manifest("kws")
    normalized_metric = metric.lower()
    if normalized_metric not in set(manifest["metrics"]):
        raise ValueError(f"Unsupported KWS metric: {metric}")
    routes, _ = load_task_routes("kws")
    route = find_task_route(routes, metric=normalized_metric, input_mode=input_mode)
    return manifest, manifest_path, route, normalized_metric


def _infer_input_mode(kwargs: dict) -> str:
    if kwargs.get("reference_jsonl") and kwargs.get("sample_output"):
        return "sure_json"
    if kwargs.get("wekws_label_file") and kwargs.get("wekws_score_file") and kwargs.get("keyword"):
        return "wekws_score_ctc"
    if kwargs.get("wekws_label_file") and kwargs.get("wekws_frame_score_file") and kwargs.get("keyword"):
        return "wekws_frame_score"
    return "sure_json"
