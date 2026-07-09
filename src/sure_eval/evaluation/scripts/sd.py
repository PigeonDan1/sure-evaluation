"""SD configured script route descriptors."""

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


def describe_pipeline(*, metric: str = "der"):
    manifest, manifest_path, route, normalized_metric = _select_route(metric=metric)
    return describe_from_contracts(
        task="SD",
        pipeline_id=route["pipeline_id"],
        metric=normalized_metric,
        language="n/a",
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
    )


def run(ref_file: str, hyp_file: str, *, output_dir: str, metric: str = "der", collar: float | None = None):
    if not output_dir:
        raise ValueError("output_dir is required")
    description = describe_pipeline(metric=metric)
    _, _, route, _ = _select_route(metric=metric)
    params = dict(route.get("params") or {})
    if collar is not None:
        params["collar"] = collar
    report = call_route_executor(
        route,
        ref_file=ref_file,
        hyp_file=hyp_file,
        metric=description.metric,
        **params,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(*, metric: str = "der"):
    manifest, manifest_path = load_task_manifest("sd")
    normalized_metric = metric.lower()
    routes, _ = load_task_routes("sd")
    route = find_task_route(routes, metric=normalized_metric)
    return manifest, manifest_path, route, normalized_metric
