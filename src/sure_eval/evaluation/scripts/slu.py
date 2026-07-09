"""SLU configured script route descriptors."""

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


def describe_pipeline(*, metric: str = "accuracy", output_mode: str = "choice_id"):
    if metric.lower() != "accuracy":
        raise ValueError(f"Unsupported SLU metric: {metric}")
    manifest, manifest_path, route = _select_route(output_mode=output_mode)
    return describe_from_contracts(
        task="SLU",
        pipeline_id=route["pipeline_id"],
        metric="accuracy",
        language="n/a",
        node_ids=tuple(route["nodes"]),
        contracts=(contract_from_manifest(manifest, route["input_contract"]),),
        task_config_path=manifest_path,
    )


def run(
    ref_file: str,
    hyp_file: str,
    *,
    prompt_jsonl: str,
    output_dir: str,
    output_mode: str = "choice_id",
):
    if not output_dir:
        raise ValueError("output_dir is required")
    description = describe_pipeline(output_mode=output_mode)
    _, _, route = _select_route(output_mode=output_mode)
    report = call_route_executor(
        route,
        ref_file=ref_file,
        hyp_file=hyp_file,
        prompt_jsonl=prompt_jsonl,
        output_mode=output_mode,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(*, output_mode: str = "choice_id"):
    manifest, manifest_path = load_task_manifest("slu")
    routes, _ = load_task_routes("slu")
    route = find_task_route(routes, metric="accuracy", output_mode=output_mode)
    return manifest, manifest_path, route
