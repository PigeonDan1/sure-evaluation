"""Typer commands for deterministic metric evaluation."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import sure_eval
from sure_eval.evaluation.cli_adapters import (
    build_pipeline_spec,
    read_json,
    run_pipeline_spec,
)
from sure_eval.evaluation.agent_plan import build_agent_plan, write_agent_plan
from sure_eval.evaluation.cache import get_cache_root
from sure_eval.evaluation.env_check import (
    EnvironmentCheckError,
    NodeEnvChecker,
    doctor_payload,
    check_pipeline_environment,
    iter_known_node_ids,
    package_install_specs,
    raise_if_environment_failed,
)

metric_app = typer.Typer(help="Describe and run deterministic evaluation pipelines")
env_app = typer.Typer(help="Inspect and prepare optional node-local environments")
agent_app = typer.Typer(help="Plan route selection and environment readiness for agents")
app = typer.Typer(help="SURE-EVAL deterministic speech and audio evaluation")
console = Console()


@metric_app.command("describe")
def describe_metric_pipeline(
    task: str = typer.Argument(..., help="Task name, e.g. asr, s2tt, kws, classification, slu"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Task language/profile"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Metric name"),
    metrics: Optional[str] = typer.Option(
        None, "--metrics", help="Comma-separated metric names for multi-metric tasks"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write pipeline JSON to this path"
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Describe a route-backed metric pipeline without executing it."""

    try:
        selected_metric = metrics or metric
        payload = build_pipeline_spec(
            task, language=language, metric=selected_metric, output_path=output
        )
    except Exception as exc:
        _print_error(exc, json_output=json_output)
        raise typer.Exit(1) from exc
    if json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return
    table = Table(title=f"Metric Pipeline: {payload['task']}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("pipeline_id", payload["pipeline_id"])
    table.add_row("metric", payload["metric"])
    table.add_row("language", payload["language"])
    table.add_row("required_roles", ", ".join(payload["required_roles"]))
    table.add_row("nodes", " -> ".join(slot["default"] for slot in payload["pipeline"]))
    if output:
        table.add_row("pipeline_json", str(output))
    console.print(table)


@metric_app.command("run")
def run_metric_pipeline(
    pipeline: Path = typer.Option(
        ..., "--pipeline", "-p", help="Pipeline JSON emitted by metric describe"
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="Expected output directory"),
    ref_file: Optional[str] = typer.Option(None, "--ref-file", help="Reference key-text file"),
    hyp_file: Optional[str] = typer.Option(None, "--hyp-file", help="Hypothesis key-text file"),
    src_file: Optional[str] = typer.Option(None, "--src-file", help="Source key-text file"),
    prompt_jsonl: Optional[str] = typer.Option(
        None, "--prompt-jsonl", help="SLU prompt JSONL file"
    ),
    label_spec: Optional[str] = typer.Option(
        None, "--label-spec", help="Classification label spec path or id"
    ),
    reference_jsonl: Optional[str] = typer.Option(
        None, "--reference-jsonl", help="KWS reference JSONL"
    ),
    sample_output: Optional[str] = typer.Option(
        None, "--sample-output", help="KWS model output JSONL"
    ),
    wekws_label_file: Optional[str] = typer.Option(
        None, "--wekws-label-file", help="WeKWS label file"
    ),
    wekws_score_file: Optional[str] = typer.Option(
        None, "--wekws-score-file", help="WeKWS CTC score file"
    ),
    wekws_frame_score_file: Optional[str] = typer.Option(
        None, "--wekws-frame-score-file", help="WeKWS frame score file"
    ),
    keyword: Optional[str] = typer.Option(None, "--keyword", help="KWS keyword"),
    macro_recall_false_alarms: int = typer.Option(
        0, "--macro-recall-false-alarms", help="False alarm count budget for KWS macro-recall"
    ),
    samples_jsonl: Optional[str] = typer.Option(
        None, "--samples-jsonl", help="TTS/VC/SE/TSE samples JSONL file"
    ),
    device: str = typer.Option(
        "cuda", "--device", help="Device passed to audio metric runtime builders"
    ),
    cache_dir: Optional[str] = typer.Option(
        None, "--cache-dir", help="Cache directory for audio metric runtime builders"
    ),
    validate_env: bool = typer.Option(
        False, "--validate-env", help="Validate selected node-local environments before running"
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Run a previously described metric pipeline."""

    try:
        payload = read_json(pipeline)
        if validate_env:
            env_results = check_pipeline_environment(payload)
            raise_if_environment_failed(env_results)
        summary = run_pipeline_spec(
            payload,
            output_dir=str(output_dir),
            ref_file=ref_file,
            hyp_file=hyp_file,
            src_file=src_file,
            prompt_jsonl=prompt_jsonl,
            label_spec=label_spec,
            reference_jsonl=reference_jsonl,
            sample_output=sample_output,
            wekws_label_file=wekws_label_file,
            wekws_score_file=wekws_score_file,
            wekws_frame_score_file=wekws_frame_score_file,
            keyword=keyword,
            macro_recall_false_alarms=macro_recall_false_alarms,
            samples_jsonl=samples_jsonl,
            device=device,
            cache_dir=cache_dir,
        )
    except EnvironmentCheckError as exc:
        _print_env_error(exc, json_output=json_output)
        raise typer.Exit(1) from exc
    except Exception as exc:
        _print_error(exc, json_output=json_output)
        raise typer.Exit(1) from exc
    if json_output:
        sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return
    console.print(
        "[yellow]Environment note:[/yellow] use --validate-env for checks; "
        "check pyproject.toml or uv.lock in selected node directories."
    )
    table = Table(title="Metric Run")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("status", summary["status"])
    table.add_row("task", summary["task"])
    table.add_row("metric", summary["metric"])
    table.add_row("score", f"{summary['score']:.6f}")
    table.add_row("pipeline_id", summary["pipeline_id"])
    table.add_row("report", summary["report_path"])
    table.add_row("pipeline_description", summary["pipeline_description_path"])
    console.print(table)


def _print_error(exc: Exception, *, json_output: bool) -> None:
    if json_output:
        sys.stdout.write(
            json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False) + "\n"
        )
    else:
        console.print(f"[bold red]Error:[/bold red] {exc}")


def _print_env_error(exc: EnvironmentCheckError, *, json_output: bool) -> None:
    payload = {
        "status": "error",
        "kind": "environment",
        "message": str(exc),
        "checks": [result.as_dict() for result in exc.results],
    }
    if json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return
    console.print(f"[bold red]Environment check failed:[/bold red] {exc}")
    for result in exc.results:
        if result.status == "failed":
            console.print(f"  [red]FAIL[/red] {result.node_id or result.name}: {result.message}")
            if result.fix:
                console.print(f"       Fix: {result.fix}")


@agent_app.command("plan")
def agent_plan(
    task_arg: Optional[str] = typer.Argument(None, help="Task name, e.g. asr, tts, vc"),
    task_opt: Optional[str] = typer.Option(
        None, "--task", help="Task name for non-positional callers"
    ),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Task language/profile"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Single metric name"),
    metrics: Optional[str] = typer.Option(None, "--metrics", help="Comma-separated metric names"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write plan JSON to this path"
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Describe selected routes and required node environments without scoring."""

    selected_task = task_opt or task_arg
    if not selected_task:
        raise typer.BadParameter("Task is required as an argument or --task")
    if metric and metrics:
        raise typer.BadParameter("Use only one of --metric or --metrics")

    try:
        payload = build_agent_plan(
            selected_task,
            language=language,
            metric=metric,
            metrics=metrics,
        )
    except Exception as exc:
        _print_error(exc, json_output=json_output)
        raise typer.Exit(1) from exc

    if output:
        write_agent_plan(output, payload)
    if json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return
    _print_agent_plan_table(payload, output=output)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print package version and exit"),
) -> None:
    """SURE-EVAL command line interface."""

    if version:
        console.print(sure_eval.__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON")
) -> None:
    """Check root package and optional node environment status."""

    payload = doctor_payload()
    if json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    else:
        _print_check_table("SURE-EVAL Doctor", payload["checks"])
    if payload["status"] == "failed":
        raise typer.Exit(1)


@env_app.command("list")
def env_list(
    group: Optional[str] = typer.Option(None, "--group", help="Filter by node_env.yaml group"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """List known node environments."""

    checker = NodeEnvChecker()
    node_ids = _node_ids_for_group(group) if group else list(iter_known_node_ids())
    checks = [_check_with_metadata(checker, node_id) for node_id in node_ids]
    payload = {"status": "ok", "cache_root": str(get_cache_root(create=False)), "nodes": checks}
    if json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return
    table = Table(title="SURE-EVAL Node Environments")
    table.add_column("Node", style="cyan")
    table.add_column("Group")
    table.add_column("Runtime")
    table.add_column("Status")
    table.add_column("Message")
    for item in checks:
        table.add_row(
            item.get("node_id", item["name"]),
            item.get("group", ""),
            item.get("runtime", ""),
            item["status"],
            item["message"],
        )
    console.print(table)


@env_app.command("check")
def env_check(
    node: Optional[str] = typer.Option(
        None, "--node", help="Check one node id, e.g. scoring/dnsmos"
    ),
    pipeline: Optional[Path] = typer.Option(
        None, "--pipeline", "-p", help="Check nodes selected by a pipeline JSON"
    ),
    task: Optional[str] = typer.Option(None, "--task", help="Check nodes selected by a task route"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Task language/profile"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Single metric name"),
    metrics: Optional[str] = typer.Option(None, "--metrics", help="Comma-separated metric names"),
    group: Optional[str] = typer.Option(
        None, "--group", help="Check nodes in one node_env.yaml group"
    ),
    all_nodes: bool = typer.Option(False, "--all", help="Check all known nodes"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Validate optional node-local environments without creating them."""

    if sum(bool(value) for value in (node, pipeline, task, group, all_nodes)) > 1:
        raise typer.BadParameter("Use only one of --node, --pipeline, --task, --group, or --all")
    checker = NodeEnvChecker()
    node_ids = _resolve_env_node_ids(
        node=node,
        pipeline=pipeline,
        task=task,
        language=language,
        metric=metric,
        metrics=metrics,
        group=group,
        all_nodes=all_nodes,
    )
    checks = [_check_with_metadata(checker, node_id) for node_id in node_ids]
    failed = [item for item in checks if item["status"] == "failed"]
    payload = {"status": "failed" if failed else "ok", "checks": checks}
    if json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    else:
        _print_check_table("SURE-EVAL Environment Check", checks)
    if failed:
        raise typer.Exit(1)


@env_app.command("setup")
def env_setup(
    node: Optional[str] = typer.Option(None, "--node", help="Prepare one node id"),
    task: Optional[str] = typer.Option(
        None, "--task", help="Prepare nodes selected by a task route"
    ),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Task language/profile"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Single metric name"),
    metrics: Optional[str] = typer.Option(None, "--metrics", help="Comma-separated metric names"),
    group: Optional[str] = typer.Option(
        None, "--group", help="Prepare nodes in one node_env.yaml group"
    ),
    all_nodes: bool = typer.Option(False, "--all", help="Prepare all known optional nodes"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print planned setup actions without running them"
    ),
    force: bool = typer.Option(
        False, "--force", help="Recreate or refresh environments even if they already exist"
    ),
    no_download: bool = typer.Option(
        False, "--no-download", help="Skip checkpoint/model downloads"
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Prepare node-local environments from node_env.yaml metadata."""

    if sum(bool(value) for value in (node, task, group, all_nodes)) > 1:
        raise typer.BadParameter("Use only one of --node, --task, --group, or --all")
    node_ids = _resolve_env_node_ids(
        node=node,
        pipeline=None,
        task=task,
        language=language,
        metric=metric,
        metrics=metrics,
        group=group,
        all_nodes=all_nodes,
    )
    actions = [_setup_plan_for_node(node_id, no_download=no_download) for node_id in node_ids]
    if dry_run:
        payload = {"status": "planned", "dry_run": True, "actions": actions}
    else:
        results = [_execute_setup_action(action, force=force) for action in actions]
        failed = [item for item in results if item["status"] == "failed"]
        payload = {"status": "failed" if failed else "ok", "dry_run": False, "actions": results}
    if json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        if payload["status"] == "failed":
            raise typer.Exit(1)
        return
    table = Table(title="SURE-EVAL Environment Setup Plan")
    table.add_column("Node", style="cyan")
    table.add_column("Command")
    table.add_column("Status")
    for action in payload["actions"]:
        table.add_row(action["node_id"], action.get("command", ""), action["status"])
    console.print(table)
    if payload["status"] == "failed":
        raise typer.Exit(1)


@env_app.command("download")
def env_download(
    node: Optional[str] = typer.Option(None, "--node", help="Download assets for one node id"),
    task: Optional[str] = typer.Option(
        None, "--task", help="Download assets for nodes selected by a task route"
    ),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Task language/profile"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Single metric name"),
    metrics: Optional[str] = typer.Option(None, "--metrics", help="Comma-separated metric names"),
    group: Optional[str] = typer.Option(
        None, "--group", help="Download assets for one node_env.yaml group"
    ),
    all_nodes: bool = typer.Option(
        False, "--all", help="Download assets for all known optional nodes"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print planned downloads without running them"
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Download model/tool assets declared by node_env.yaml when supported."""

    if sum(bool(value) for value in (node, task, group, all_nodes)) > 1:
        raise typer.BadParameter("Use only one of --node, --task, --group, or --all")
    node_ids = _resolve_env_node_ids(
        node=node,
        pipeline=None,
        task=task,
        language=language,
        metric=metric,
        metrics=metrics,
        group=group,
        all_nodes=all_nodes,
    )
    plans = [_download_plan_for_node(node_id) for node_id in node_ids]
    if dry_run:
        payload = {"status": "planned", "dry_run": True, "downloads": plans}
    else:
        results = [_execute_download_plan(plan) for plan in plans]
        failed = [
            asset
            for plan in results
            for asset in plan.get("assets", [])
            if asset.get("status") == "failed"
        ]
        payload = {"status": "failed" if failed else "ok", "dry_run": False, "downloads": results}
    if json_output:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        if payload["status"] == "failed":
            raise typer.Exit(1)
        return
    table = Table(title="SURE-EVAL Asset Downloads")
    table.add_column("Node", style="cyan")
    table.add_column("Asset")
    table.add_column("Provider")
    table.add_column("Status")
    for plan in payload["downloads"]:
        assets = plan.get("assets", []) if isinstance(plan, dict) else []
        if not assets:
            table.add_row(str(plan.get("node_id", "")), "", "", "none")
        for asset in assets:
            table.add_row(
                str(plan.get("node_id", "")),
                str(asset.get("id", "")),
                str(asset.get("provider", "")),
                str(asset.get("status", "planned")),
            )
    console.print(table)
    if payload["status"] == "failed":
        raise typer.Exit(1)


def _setup_plan_for_node(node_id: str, *, no_download: bool) -> dict[str, object]:
    checker = NodeEnvChecker()
    node_path = checker.node_path(node_id)
    node_env = checker.load_node_env(node_id) or {}
    runtime = node_env.get("runtime") if isinstance(node_env.get("runtime"), dict) else {}
    runtime_type = str(runtime.get("type", "uv"))
    project = str(runtime.get("project", "pyproject.toml"))
    python = runtime.get("python")
    group = str(node_env.get("group") or "")
    command_parts = [f"cd {node_path}"]
    if runtime_type == "uv":
        if python:
            command_parts.append(f"uv venv --python {python}")
        command_parts.append(
            "uv sync" if project == "pyproject.toml" else f"uv sync --project {project}"
        )
    elif runtime_type == "pip":
        specs = package_install_specs(node_env)
        if specs:
            command_parts.append(
                "python -m pip install " + " ".join(shlex.quote(spec) for spec in specs)
            )
        else:
            command_parts.append("# no pip packages declared")
    elif runtime_type == "binary" and runtime.get("build_script"):
        command_parts.append(f"bash {runtime['build_script']}")
    else:
        command_parts.append(f"# setup runtime type {runtime_type}")
    models = node_env.get("models") if isinstance(node_env.get("models"), list) else []
    downloads = []
    for model in models:
        if isinstance(model, dict):
            downloads.append(
                {
                    "id": model.get("id"),
                    "provider": model.get("provider"),
                    "target": model.get("target"),
                    "env": model.get("env"),
                }
            )
    return {
        "node_id": node_id,
        "node_path": str(node_path),
        "group": group,
        "runtime": runtime_type,
        "python": python,
        "project": project,
        "build_script": runtime.get("build_script"),
        "command": " && ".join(command_parts),
        "packages": package_install_specs(node_env),
        "no_download": no_download,
        "downloads": [] if no_download else downloads,
        "node_env": str(checker.node_env_path(node_id)) if node_env else "",
        "status": "planned",
        "note": "Checkpoint downloads are declared but not executed in this command yet.",
    }


def _download_plan_for_node(node_id: str) -> dict[str, object]:
    checker = NodeEnvChecker()
    node_path = checker.node_path(node_id)
    node_env = checker.load_node_env(node_id) or {}
    assets: list[dict[str, object]] = []
    for collection in ("models", "tools", "packages"):
        values = node_env.get(collection) if isinstance(node_env.get(collection), list) else []
        for item in values:
            if not isinstance(item, dict):
                continue
            asset = dict(item)
            asset.setdefault("kind", collection[:-1])
            asset.setdefault("status", "planned")
            if asset.get("target") and not str(asset["target"]).startswith("${"):
                asset["target_path"] = str(node_path / str(asset["target"]))
            if asset.get("env"):
                asset["env_override"] = str(asset["env"])
            asset.setdefault("license", "see upstream provider")
            asset.setdefault("citation", "see upstream provider")
            assets.append(asset)
    return {
        "node_id": node_id,
        "node_path": str(node_path),
        "group": node_env.get("group", ""),
        "node_env": str(checker.node_env_path(node_id)) if node_env else "",
        "assets": assets,
    }


def _execute_download_plan(plan: dict[str, object]) -> dict[str, object]:
    result = dict(plan)
    assets = []
    for asset in plan.get("assets", []):
        if not isinstance(asset, dict):
            continue
        executed = dict(asset)
        try:
            _download_asset(executed)
        except Exception as exc:
            executed["status"] = "failed"
            executed["message"] = str(exc)
        else:
            executed["status"] = "ok"
        assets.append(executed)
    result["assets"] = assets
    return result


def _download_asset(asset: dict[str, object]) -> None:
    provider = str(asset.get("provider") or "").lower()
    model_id = str(asset.get("id") or "")
    target_path = asset.get("target_path")
    if provider == "huggingface":
        from huggingface_hub import snapshot_download

        kwargs = {"repo_id": model_id}
        if target_path:
            kwargs["local_dir"] = str(Path(str(target_path)).parent)
        snapshot_download(**kwargs)
        return
    if provider == "modelscope":
        from modelscope import snapshot_download

        kwargs = {"model_id": model_id}
        if target_path:
            kwargs["cache_dir"] = str(Path(str(target_path)).parents[1])
        snapshot_download(**kwargs)
        return
    raise RuntimeError(
        f"provider {provider!r} is manual or unsupported for automated download; "
        "use the target/env fields from node_env.yaml."
    )


def _resolve_env_node_ids(
    *,
    node: str | None,
    pipeline: Path | None,
    task: str | None,
    language: str | None,
    metric: str | None,
    metrics: str | None,
    group: str | None,
    all_nodes: bool,
) -> list[str]:
    if node:
        return [node]
    if pipeline is not None:
        return _node_ids_from_pipeline(read_json(pipeline))
    if task:
        return _node_ids_from_task(task, language=language, metric=metric, metrics=metrics)
    if group:
        return _node_ids_for_group(group)
    return list(iter_known_node_ids())


def _node_ids_from_pipeline(pipeline: dict[str, object]) -> list[str]:
    known = set(iter_known_node_ids())
    node_ids = []
    for item in pipeline.get("nodes") or ():
        if isinstance(item, dict):
            node_id = str(item.get("node_id") or "")
            if node_id in known and node_id not in node_ids:
                node_ids.append(node_id)
    return node_ids


def _node_ids_from_task(
    task: str,
    *,
    language: str | None,
    metric: str | None,
    metrics: str | None,
) -> list[str]:
    selected_metric = metrics or metric
    payload = build_pipeline_spec(task, language=language, metric=selected_metric)
    return _node_ids_from_pipeline(payload)


def _node_ids_for_group(group: str) -> list[str]:
    checker = NodeEnvChecker()
    node_ids = []
    for node_id in iter_known_node_ids():
        node_env = checker.load_node_env(node_id) or {}
        if str(node_env.get("group") or "") == group:
            node_ids.append(node_id)
    return node_ids


def _check_with_metadata(checker: NodeEnvChecker, node_id: str) -> dict[str, object]:
    payload = checker.check_node(node_id).as_dict()
    node_env = checker.load_node_env(node_id) or {}
    if node_env.get("group"):
        payload["group"] = node_env["group"]
    if node_env:
        payload["node_env"] = str(checker.node_env_path(node_id))
    return payload


def _execute_setup_action(action: dict[str, object], *, force: bool) -> dict[str, object]:
    runtime = str(action.get("runtime") or "")
    node_path = Path(str(action["node_path"]))
    result = dict(action)
    log_file = _setup_log_file(str(action["node_id"]))
    result["log_file"] = str(log_file)
    try:
        if runtime == "uv":
            _execute_uv_setup(action, node_path=node_path, log_file=log_file, force=force)
        elif runtime == "binary":
            _execute_binary_setup(action, node_path=node_path, log_file=log_file)
        else:
            result["status"] = "skipped"
            result["message"] = f"unsupported runtime type: {runtime}"
            return result
    except Exception as exc:
        result["status"] = "failed"
        result["message"] = str(exc)
        return result
    result["status"] = "ok"
    result["message"] = "environment setup command completed"
    return result


def _execute_uv_setup(
    action: dict[str, object],
    *,
    node_path: Path,
    log_file: Path,
    force: bool,
) -> None:
    commands: list[list[str]] = []
    venv_dir = node_path / ".venv"
    python = action.get("python")
    if force or not venv_dir.exists():
        command = ["uv", "venv"]
        if python:
            command.extend(["--python", str(python)])
        commands.append(command)
    commands.append(["uv", "sync"])
    for command in commands:
        _run_logged(command, cwd=node_path, log_file=log_file)


def _execute_binary_setup(action: dict[str, object], *, node_path: Path, log_file: Path) -> None:
    build_script = action.get("build_script")
    if not build_script:
        raise RuntimeError("binary node has no build_script in node_env.yaml")
    _run_logged(["bash", str(build_script)], cwd=node_path, log_file=log_file)


def _run_logged(command: list[str], *, cwd: Path, log_file: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.write(completed.stdout or "")
        handle.write(f"\n[returncode] {completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}; see {log_file}")


def _setup_log_file(node_id: str) -> Path:
    log_dir = get_cache_root() / "logs" / "env-setup"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_node = node_id.replace("/", "__")
    return log_dir / f"{timestamp}-{safe_node}.log"


def _print_check_table(title: str, checks: list[dict[str, object]]) -> None:
    table = Table(title=title)
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Message")
    table.add_column("Fix")
    for check in checks:
        table.add_row(
            str(check.get("node_id") or check["name"]),
            str(check["status"]),
            str(check["message"]),
            str(check.get("fix", "")),
        )
    console.print(table)


def _print_agent_plan_table(payload: dict[str, object], *, output: Path | None) -> None:
    table = Table(title="SURE-EVAL Agent Plan")
    table.add_column("Metric", style="cyan")
    table.add_column("Pipeline")
    table.add_column("Env")
    table.add_column("Ready")
    table.add_column("Setup")
    for route in payload.get("selected_routes") or []:
        if not isinstance(route, dict):
            continue
        setup_nodes = [
            str(check.get("node_id"))
            for check in route.get("env_checks") or []
            if isinstance(check, dict) and check.get("blocking")
        ]
        table.add_row(
            str(route.get("metric")),
            str(route.get("pipeline_id")),
            str(route.get("environment_status")),
            "yes" if route.get("can_run_now") else "no",
            ", ".join(setup_nodes),
        )
    console.print(table)
    if output:
        console.print(f"plan_json: {output}")
    if payload.get("blocking_issues"):
        console.print("[yellow]Blocking issues:[/yellow]")
        for issue in payload["blocking_issues"]:
            console.print(f"  - {issue}")


app.add_typer(metric_app, name="metric")
app.add_typer(env_app, name="env")
app.add_typer(agent_app, name="agent")
