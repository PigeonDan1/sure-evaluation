"""Classification configured script route descriptors."""

from __future__ import annotations

from pathlib import Path

from sure_eval.evaluation.scripts.contracts import (
    call_route_executor,
    contract_from_manifest,
    describe_from_contracts,
    find_task_route,
    load_task_manifest,
    load_task_routes,
    write_route_run_outputs,
)


def describe_pipeline(*, task: str = "classification", metric: str = "accuracy"):
    if metric.lower() != "accuracy":
        raise ValueError(f"Unsupported classification metric: {metric}")
    manifest, manifest_path, route, normalized_task = _select_route(task=task)
    return describe_from_contracts(
        task=normalized_task,
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
    output_dir: str,
    task: str = "classification",
    label_spec: str | Path | dict | None = None,
):
    if not output_dir:
        raise ValueError("output_dir is required")
    description = describe_pipeline(task=task, metric="accuracy")
    _, _, route, _ = _select_route(task=task)
    report = call_route_executor(
        route,
        ref_file=ref_file,
        hyp_file=hyp_file,
        task=task,
        label_spec=label_spec,
    )
    return write_route_run_outputs(report=report, description=description, output_dir=output_dir)


def _select_route(*, task: str = "classification"):
    manifest, manifest_path = load_task_manifest("classification")
    normalized_task = task.upper() if task.upper() in {"SER", "GR"} else task
    routes, _ = load_task_routes("classification")
    route = find_task_route(routes, metric="accuracy", task_alias=normalized_task)
    return manifest, manifest_path, route, normalized_task
