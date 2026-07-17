"""Agent-facing route and environment planning helpers."""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any

from sure_eval.evaluation.cache import CACHE_ENV_VAR, get_cache_root
from sure_eval.evaluation.cli_adapters import build_pipeline_spec, normalize_task, split_metric_csv
from sure_eval.evaluation.env_check import NodeEnvChecker, package_install_specs

SCHEMA = "sure.eval.agent_plan.v1"


def build_agent_plan(
    task: str,
    *,
    language: str | None = None,
    metric: str | None = None,
    metrics: str | list[str] | tuple[str, ...] | None = None,
    include_root_env: bool = True,
) -> dict[str, Any]:
    """Build a deterministic route/env plan without executing metrics.

    The payload is intended for TUI agents and harnesses that need to know which
    configured routes will run, which node environments are required, and which
    setup command should be shown before a scoring run.
    """

    if metric and metrics:
        raise ValueError("Use only one of metric or metrics")

    selected_metrics = _requested_metrics(
        task=task,
        language=language,
        metric=metric,
        metrics=metrics,
    )
    checker = NodeEnvChecker()
    routes = [_route_plan(task, language=language, metric=item, checker=checker) for item in selected_metrics]
    root_env = _root_env_payload() if include_root_env else {"status": "skipped", "checks": []}
    blocking_issues = _blocking_issues(root_env=root_env, routes=routes)

    return {
        "schema": SCHEMA,
        "status": "blocked" if blocking_issues else "ready",
        "task": normalize_task(task),
        "language": language,
        "metrics": selected_metrics,
        "root_env": root_env,
        "selected_routes": routes,
        "can_run_now": not blocking_issues,
        "blocking_issues": blocking_issues,
        "next_steps": _next_steps(blocking_issues, routes),
    }


def write_agent_plan(path: Path, payload: dict[str, Any]) -> None:
    """Write an agent plan with stable JSON formatting."""

    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _requested_metrics(
    *,
    task: str,
    language: str | None,
    metric: str | None,
    metrics: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    if metric:
        values = list(split_metric_csv(metric))
    elif isinstance(metrics, str):
        values = list(split_metric_csv(metrics))
    elif metrics:
        values = [str(item).strip().lower() for item in metrics if str(item).strip()]
    else:
        default_spec = build_pipeline_spec(task, language=language)
        values = [str(item).lower() for item in default_spec.get("metrics") or ()]
        if not values and default_spec.get("metric") and default_spec["metric"] != "multi":
            values = [str(default_spec["metric"]).lower()]
    return _dedupe(values)


def _route_plan(
    task: str,
    *,
    language: str | None,
    metric: str,
    checker: NodeEnvChecker,
) -> dict[str, Any]:
    spec = build_pipeline_spec(task, language=language, metric=metric)
    checks = [_node_env_payload(checker, str(node["node_id"])) for node in spec.get("nodes") or ()]
    blocking = [item for item in checks if item.get("blocking")]
    warnings = [item for item in checks if item.get("status") == "warning"]
    environment_status = "failed" if blocking else "warning" if warnings else "ok"
    nodes = []
    checks_by_node = {item["node_id"]: item for item in checks}
    for node in spec.get("nodes") or ():
        node_id = str(node["node_id"])
        check = checks_by_node.get(node_id, {})
        nodes.append(
            {
                "node_id": node_id,
                "stage": node_id.split("/", 1)[0],
                "runtime": check.get("runtime", ""),
                "manifest_path": node.get("manifest_path"),
                "status": check.get("status"),
            }
        )

    return {
        "metric": metric,
        "resolved_metric": spec.get("metric"),
        "route_id": spec.get("route_id"),
        "pipeline_id": spec.get("pipeline_id"),
        "task": spec.get("task"),
        "language": spec.get("language"),
        "nodes": nodes,
        "required_roles": spec.get("required_roles") or [],
        "optional_roles": spec.get("optional_roles") or [],
        "run_args": spec.get("run_args") or {},
        "conversion_steps": spec.get("conversion_steps") or [],
        "task_config_path": spec.get("task_config_path"),
        "env_checks": checks,
        "environment_status": environment_status,
        "setup_required": bool(blocking),
        "can_run_now": not blocking,
    }


def _node_env_payload(checker: NodeEnvChecker, node_id: str) -> dict[str, Any]:
    result = checker.check_node(node_id)
    payload = result.as_dict()
    payload["node_id"] = node_id
    payload["required_for_selected_route"] = result.required
    payload["blocking"] = bool(result.required and result.status != "ok")
    node_env = checker.load_node_env(node_id) or {}
    if node_env:
        payload["node_env"] = str(checker.node_env_path(node_id))
        payload["group"] = node_env.get("group", "")
    if payload["blocking"]:
        payload["setup"] = _setup_hint(checker, node_id, node_env, fallback=result.fix)
    return payload


def _setup_hint(
    checker: NodeEnvChecker,
    node_id: str,
    node_env: dict[str, Any],
    *,
    fallback: str,
) -> dict[str, Any]:
    node_path = checker.node_path(node_id)
    runtime = node_env.get("runtime") if isinstance(node_env.get("runtime"), dict) else {}
    runtime_type = str(runtime.get("type") or "")
    packages = package_install_specs(node_env)
    command = fallback

    if runtime_type == "pip":
        if packages:
            command = "python -m pip install " + " ".join(shlex.quote(spec) for spec in packages)
        else:
            command = "# no pip packages declared"
    elif runtime_type == "uv":
        command = f"cd {node_path} && uv sync"
    elif runtime_type == "binary":
        build_script = runtime.get("build_script")
        if build_script:
            command = f"cd {node_path} && bash {build_script}"

    return {
        "runtime": runtime_type or "inferred",
        "command": command,
        "packages": packages,
        "assets": _asset_hints(node_path, node_env),
        "node_path": str(node_path),
    }


def _asset_hints(node_path: Path, node_env: dict[str, Any]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for collection in ("models", "tools"):
        values = node_env.get(collection) if isinstance(node_env.get(collection), list) else []
        for item in values:
            if not isinstance(item, dict):
                continue
            target = item.get("target")
            asset = {
                "kind": collection[:-1],
                "id": item.get("id"),
                "provider": item.get("provider"),
                "target": target,
                "env": item.get("env"),
            }
            if target and not str(target).startswith("${"):
                asset["target_path"] = str(node_path / str(target))
            assets.append(asset)
    return assets


def _root_env_payload() -> dict[str, Any]:
    checks = [
        _check(
            name="python",
            status="ok" if sys.version_info >= (3, 10) else "failed",
            message=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            fix="Use Python >= 3.10",
            required=True,
        ),
        _check(
            name="sure_eval",
            status="ok" if importlib.util.find_spec("sure_eval") else "failed",
            message="importable" if importlib.util.find_spec("sure_eval") else "not importable",
            fix="Set PYTHONPATH=src or install the package",
            required=True,
        ),
        _check(
            name="uv",
            status="ok" if shutil.which("uv") else "warning",
            message=shutil.which("uv") or "uv not found on PATH",
            fix="Install uv before preparing uv-backed node-local environments",
            required=False,
        ),
        {
            "name": "cache_root",
            "status": "ok",
            "message": str(get_cache_root(create=False)),
            "required": False,
            "details": {"env_var": CACHE_ENV_VAR},
        },
    ]
    failed = [item for item in checks if item["status"] == "failed"]
    warnings = [item for item in checks if item["status"] == "warning"]
    return {
        "status": "failed" if failed else "warning" if warnings else "ok",
        "checks": checks,
    }


def _check(
    *,
    name: str,
    status: str,
    message: str,
    fix: str,
    required: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "message": message,
        "fix": fix,
        "required": required,
        "blocking": bool(required and status == "failed"),
    }


def _blocking_issues(*, root_env: dict[str, Any], routes: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for check in root_env.get("checks") or []:
        if check.get("blocking"):
            issues.append(f"root_env/{check.get('name')}: {check.get('message')}")
    for route in routes:
        for check in route.get("env_checks") or []:
            if check.get("blocking"):
                issues.append(
                    f"{route.get('metric')}:{check.get('node_id')}: {check.get('message')}"
                )
    return issues


def _next_steps(blocking_issues: list[str], routes: list[dict[str, Any]]) -> list[str]:
    if not blocking_issues:
        return ["Run `sure-eval metric describe`, then `sure-eval metric run` with the declared inputs."]
    commands = []
    for route in routes:
        for check in route.get("env_checks") or []:
            setup = check.get("setup") if check.get("blocking") else None
            command = setup.get("command") if isinstance(setup, dict) else None
            if command and command not in commands:
                commands.append(command)
    return commands or ["Resolve the blocking environment checks before running metrics."]


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
