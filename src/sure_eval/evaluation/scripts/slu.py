"""SLU configured script route descriptors."""

from __future__ import annotations

from sure_eval.evaluation.scripts.contracts import (
    call_route_executor,
    contract_from_manifest,
    describe_from_contracts,
    find_pipeline_route,
    find_task_route,
    load_task_manifest,
    load_task_routes,
    write_route_run_outputs,
)


def describe_pipeline(
    *, metric: str = "accuracy", output_mode: str = "choice_id", pipeline_id: str | None = None
):
    manifest, manifest_path, routes_path, route = _select_route(
        metric=metric, output_mode=output_mode, pipeline_id=pipeline_id
    )
    return describe_from_contracts(
        task="SLU",
        pipeline_id=route["pipeline_id"],
        metric=route["metric"],
        language="n/a",
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
        route_config_path=routes_path,
        computation_node_ids=tuple(route["nodes"]),
        execution_metrics=(route["metric"],),
        script_module=__name__,
        executor=str(route.get("executor") or ""),
    )


def run(
    ref_file: str | None = None,
    hyp_file: str | None = None,
    *,
    prompt_jsonl: str,
    output_dir: str,
    output_mode: str = "choice_id",
    pipeline_id: str | None = None,
    metric: str = "accuracy",
    llm_config: str | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    _, _, _, route = _select_route(metric=metric, output_mode=output_mode, pipeline_id=pipeline_id)
    resolved_output_mode = str(route.get("output_mode") or output_mode)
    description = describe_pipeline(
        metric=route["metric"], output_mode=resolved_output_mode, pipeline_id=pipeline_id
    )
    if not hyp_file:
        raise ValueError("hyp_file is required for SLU evaluation")
    if route["metric"] == "llm_judge":
        if not llm_config:
            raise ValueError("SLU llm_judge requires llm_config (JSON file); set run_args.llm_config in the pipeline")
    else:
        if llm_config is not None:
            raise ValueError("llm_config is only accepted by the llm_judge route")
        if not ref_file:
            raise ValueError("ref_file is required for SLU accuracy")
    report = call_route_executor(
        route, ref_file=ref_file, hyp_file=hyp_file,
        prompt_jsonl=prompt_jsonl, output_mode=resolved_output_mode,
        metric=route["metric"], llm_config=llm_config,
    )
    write_route_run_outputs(report=report, description=description, output_dir=output_dir)
    if report.details.get("scoring_result", {}).get("status") == "failed":
        raise RuntimeError("judge_batch_failed; inspect report.json for sample error codes")
    return report


def _select_route(*, metric: str = "accuracy", output_mode: str = "choice_id", pipeline_id: str | None = None):
    manifest, manifest_path = load_task_manifest("slu")
    routes, routes_path = load_task_routes("slu")
    if pipeline_id:
        route = find_pipeline_route(routes, pipeline_id=pipeline_id)
    else:
        selectors = {"output_mode": output_mode} if metric.lower() == "accuracy" else {}
        route = find_task_route(routes, metric=metric.lower(), **selectors)
    return manifest, manifest_path, routes_path, route
