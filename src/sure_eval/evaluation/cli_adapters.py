"""Adapters between the metric CLI and configured evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sure_eval.evaluation.scripts import describe_pipeline, run_task
from sure_eval.evaluation.scripts.contracts import (
    NODES_ROOT,
    load_task_routes,
    load_yaml,
)
from sure_eval.evaluation.pipeline_identity import canonical_metric

ROLE_TO_CLI_ARG = {
    "ref": "ref_file",
    "hyp": "hyp_file",
    "src": "src_file",
    "prompt_jsonl": "prompt_jsonl",
    "label_spec": "label_spec",
    "reference_jsonl": "reference_jsonl",
    "sample_output": "sample_output",
    "trial_manifest": "trial_manifest",
    "wekws_label_file": "wekws_label_file",
    "wekws_score_file": "wekws_score_file",
    "wekws_frame_score_file": "wekws_frame_score_file",
    "keyword": "keyword",
    "samples_jsonl": "samples_jsonl",
}

TASK_ALIASES = {
    "ser": "classification",
    "gr": "classification",
    "sa-asr": "sa_asr",
    "speech_enhancement": "se",
}

ENVIRONMENT_NOTE = (
    "node-local environments are not validated unless --validate-env is set. "
    "Check selected node directories for pyproject.toml or uv.lock when preparing a run."
)

AUDIO_SAMPLE_TASKS = {"tts", "vc", "se", "speech_enhancement", "tse"}
SE_TASKS = {"se", "speech_enhancement"}
SE_DEFAULT_METRICS = ("si-sdr", "stoi", "pesq", "dnsmos", "wv-mos", "utmos")
SV_DEFAULT_METRICS = ("eer", "min_dcf")
PIPELINE_SCHEMA = "sure.metric.pipeline.v1"
ROUTE_LIST_SCHEMA = "sure.metric.routes.v1"
OPTIONAL_RUNTIME_TYPES = {"binary", "pip", "uv"}


def build_pipeline_spec(
    task: str,
    *,
    language: str | None = None,
    metric: str | None = None,
    pipeline_id: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Describe a configured script route as a user-editable pipeline spec."""

    normalized_task = normalize_task(task)
    if metric and pipeline_id:
        raise ValueError("Use either metric/metrics or pipeline_id, not both")
    describe_kwargs = _describe_kwargs(
        normalized_task,
        original_task=task,
        language=language,
        metric=metric,
        pipeline_id=pipeline_id,
    )
    description = describe_pipeline(normalized_task, **describe_kwargs)
    routes, _ = load_task_routes(TASK_ALIASES.get(normalized_task, normalized_task))
    route_choices = _route_choices(routes, language=description.language)
    selected_routes = _match_selected_routes(route_choices, description)
    selected_route = selected_routes[0]
    node_slots = _node_slots(
        description.node_ids, selected_route=selected_route, route_choices=route_choices
    )
    run_args = {ROLE_TO_CLI_ARG.get(role, role): None for role in description.required_roles}
    required_roles = list(description.required_roles)
    if normalized_task in AUDIO_SAMPLE_TASKS:
        required_roles = ["samples_jsonl"]
        run_args.setdefault("samples_jsonl", None)
    run_args["output_dir"] = None
    execution_metrics = tuple(description.execution_metrics) or _requested_metrics(
        normalized_task,
        metric=metric,
        description_metric=description.metric,
    )

    payload = {
        "schema": PIPELINE_SCHEMA,
        "task": normalized_task,
        "task_alias": task,
        "language": description.language,
        "metric": description.metric,
        "requested_metric": execution_metrics[0] if len(execution_metrics) == 1 else None,
        "metrics": list(execution_metrics),
        "pipeline_id": description.pipeline_id,
        "pipeline_kind": description.pipeline_kind,
        "member_pipeline_ids": list(description.member_pipeline_ids),
        "computation_node_ids": list(description.computation_node_ids),
        "pipeline": node_slots,
        "required_roles": required_roles,
        "optional_roles": list(description.optional_roles),
        "run_args": run_args,
        "route_choices": route_choices,
        "selected_pipeline_ids": [route.get("pipeline_id") for route in selected_routes],
        "task_config_path": description.task_config_path,
        "route_config_path": description.route_config_path,
        "describe_entrypoint": description.describe_entrypoint,
        "script_entrypoint": description.script_entrypoint,
        "executor": description.executor,
        "nodes": list(description.nodes),
        "conversion_steps": list(description.conversion_steps),
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload


def list_metric_routes(
    task: str,
    *,
    language: str | None = None,
    metric: str | None = None,
) -> dict[str, Any]:
    """List exact atomic routes registered for one task selection."""

    requested_task = normalize_task(task)
    route_task = TASK_ALIASES.get(requested_task, requested_task)
    routes, _ = load_task_routes(route_task)
    requested_language = _normalize_language_filter(language)
    requested_metric = canonical_metric(metric) if metric else None
    language_sensitive = bool(routes.get("language_sensitive", False))
    if not language_sensitive and requested_language not in {None, "n/a"}:
        raise ValueError(f"Task {task!r} is language-independent; omit --language or use any")
    matched: list[tuple[dict[str, Any], str]] = []
    unresolved_template = False

    for configured_route in routes.get("routes") or ():
        route = dict(configured_route)
        if not _route_matches_task_alias(route, requested_task):
            continue
        if requested_metric and canonical_metric(str(route.get("metric") or "")) != requested_metric:
            continue
        route_language = _concrete_route_language(
            route,
            requested_language=requested_language,
            language_sensitive=language_sensitive,
        )
        if route_language is None:
            unresolved_template = True
            continue
        if not _route_matches_language(route, requested_language):
            continue
        pipeline_id = str(route["pipeline_id"]).format(language=route_language)
        matched.append((route, pipeline_id))

    if unresolved_template and requested_language is None:
        raise ValueError(
            f"Task {task!r} has language-templated routes; pass --language to obtain exact "
            "pipeline IDs"
        )
    if not matched:
        filters = []
        if language:
            filters.append(f"language={language}")
        if metric:
            filters.append(f"metric={metric}")
        suffix = f" ({', '.join(filters)})" if filters else ""
        raise ValueError(f"No configured routes found for task {task!r}{suffix}")

    entries: list[dict[str, Any]] = []
    for route, pipeline_id in matched:
        spec = build_pipeline_spec(task, pipeline_id=pipeline_id)
        default_pipeline_id = _default_pipeline_id(
            task,
            language=str(spec.get("language") or "n/a"),
            metric=str(spec["metric"]),
        )
        environments = _declared_route_environments(spec["computation_node_ids"])
        entries.append(
            {
                "default": pipeline_id == default_pipeline_id,
                "pipeline_id": pipeline_id,
                "language": spec["language"],
                "metric": spec["metric"],
                "computation_node_ids": list(spec["computation_node_ids"]),
                "required_roles": list(spec["required_roles"]),
                "optional_roles": list(spec["optional_roles"]),
                "input_contract": route.get("input_contract"),
                "selectors": _route_selectors(route),
                "environments": environments,
                "setup_node_ids": [
                    item["node_id"] for item in environments if item["setup_required"]
                ],
                "route_config_path": spec["route_config_path"],
            }
        )

    default_ids = [entry["pipeline_id"] for entry in entries if entry["default"]]
    return {
        "schema": ROUTE_LIST_SCHEMA,
        "task": requested_task,
        "language": requested_language,
        "metric": requested_metric,
        "count": len(entries),
        "default_pipeline_id": default_ids[0] if len(default_ids) == 1 else None,
        "default_pipeline_ids": default_ids,
        "routes": entries,
    }


def validate_pipeline_identity(pipeline: dict[str, Any]) -> dict[str, Any]:
    """Rebuild and validate the route identity declared by a pipeline JSON."""

    if pipeline.get("schema") != PIPELINE_SCHEMA:
        raise ValueError(f"Unsupported pipeline schema: {pipeline.get('schema')!r}")
    task = str(pipeline.get("task_alias") or pipeline.get("task") or "").strip()
    pipeline_id = str(pipeline.get("pipeline_id") or "").strip()
    if not task or not pipeline_id:
        raise ValueError("Pipeline JSON requires task and pipeline_id")

    if pipeline.get("pipeline_kind") == "bundle":
        metrics = [str(item) for item in pipeline.get("metrics") or ()]
        if len(metrics) < 2:
            raise ValueError("Bundle pipeline JSON requires at least two metrics")
        expected = build_pipeline_spec(
            task,
            language=str(pipeline.get("language") or "n/a"),
            metric=",".join(metrics),
        )
    else:
        expected = build_pipeline_spec(task, pipeline_id=pipeline_id)

    comparisons = {
        "pipeline_id": str(pipeline.get("pipeline_id") or ""),
        "pipeline_kind": str(pipeline.get("pipeline_kind") or ""),
        "member_pipeline_ids": list(pipeline.get("member_pipeline_ids") or ()),
        "computation_node_ids": list(pipeline.get("computation_node_ids") or ()),
    }
    for field, actual in comparisons.items():
        expected_value = expected[field]
        if actual != expected_value:
            raise ValueError(
                f"Pipeline identity mismatch for {field}: expected {expected_value!r}, "
                f"found {actual!r}"
            )

    actual_node_ids = [
        str(item.get("node_id") or "")
        for item in pipeline.get("nodes") or ()
        if isinstance(item, dict)
    ]
    expected_node_ids = [str(item["node_id"]) for item in expected["nodes"]]
    if actual_node_ids != expected_node_ids:
        raise ValueError(
            f"Pipeline identity mismatch for nodes: expected {expected_node_ids!r}, "
            f"found {actual_node_ids!r}"
        )
    for slot in pipeline.get("pipeline") or ():
        selected = slot.get("selected")
        selected_node = slot.get("default") if selected == "default" else selected
        if selected_node != slot.get("default"):
            raise ValueError(
                "Exact pipeline JSON cannot switch nodes in place; select the registered "
                "pipeline_id for the required node chain"
            )
    return expected


def _normalize_language_filter(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip().lower().replace("-", "_")
    return "n/a" if normalized in {"any", "n/a", "none"} else normalized


def _route_matches_task_alias(route: dict[str, Any], requested_task: str) -> bool:
    route_alias = route.get("task_alias")
    if route_alias is None:
        return True
    return normalize_task(str(route_alias)) == requested_task


def _route_matches_language(route: dict[str, Any], requested_language: str | None) -> bool:
    if requested_language is None:
        return True
    route_language = route.get("language")
    if route_language is None:
        return True
    return _normalize_language_filter(str(route_language)) == requested_language


def _concrete_route_language(
    route: dict[str, Any],
    *,
    requested_language: str | None,
    language_sensitive: bool,
) -> str | None:
    route_language = route.get("language")
    if route_language is not None:
        return str(route_language).lower()
    pipeline_id = str(route.get("pipeline_id") or "")
    if "{language}" not in pipeline_id:
        return "n/a"
    if requested_language and requested_language != "n/a":
        return requested_language
    if language_sensitive:
        return None
    return "n/a"


def _default_pipeline_id(task: str, *, language: str, metric: str) -> str:
    try:
        return str(build_pipeline_spec(task, language=language, metric=metric)["pipeline_id"])
    except (KeyError, TypeError, ValueError):
        return ""


def _route_selectors(route: dict[str, Any]) -> dict[str, Any]:
    structural_fields = {
        "aliases",
        "computation_nodes",
        "executor",
        "executor_metric",
        "family",
        "input_contract",
        "internal_executor_metric",
        "language",
        "metric",
        "nodes",
        "pipeline_id",
        "task_alias",
    }
    return {key: value for key, value in route.items() if key not in structural_fields}


def _declared_route_environments(node_ids: list[str]) -> list[dict[str, Any]]:
    environments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node_id in node_ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        if node_id.startswith("conversion/"):
            runtime_type = "in_process"
            node_env: dict[str, Any] = {}
        else:
            stage, name = node_id.split("/", 1)
            node_env_path = NODES_ROOT / stage / name / "node_env.yaml"
            node_env = load_yaml(node_env_path) if node_env_path.is_file() else {}
            runtime = node_env.get("runtime") if isinstance(node_env.get("runtime"), dict) else {}
            runtime_type = str(runtime.get("type") or "in_process")
        environments.append(
            {
                "node_id": node_id,
                "runtime": runtime_type,
                "setup_required": runtime_type in OPTIONAL_RUNTIME_TYPES,
                "group": str(node_env.get("group") or ""),
            }
        )
    return environments


def run_pipeline_spec(
    pipeline: dict[str, Any],
    *,
    output_dir: str,
    ref_file: str | None = None,
    hyp_file: str | None = None,
    src_file: str | None = None,
    prompt_jsonl: str | None = None,
    label_spec: str | None = None,
    reference_jsonl: str | None = None,
    sample_output: str | None = None,
    trial_manifest: str | None = None,
    wekws_label_file: str | None = None,
    wekws_score_file: str | None = None,
    wekws_frame_score_file: str | None = None,
    keyword: str | None = None,
    macro_recall_false_alarms: int = 0,
    samples_jsonl: str | None = None,
    device: str = "cuda",
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Validate a pipeline spec and execute it through ``scripts.run_task``."""

    if not output_dir:
        raise ValueError("output_dir is required")
    validate_pipeline_selection(pipeline)
    validate_pipeline_identity(pipeline)
    task = normalize_task(str(pipeline["task"]))
    kwargs = _run_kwargs_from_pipeline(pipeline)
    cli_values = {
        "ref_file": ref_file,
        "hyp_file": hyp_file,
        "src_file": src_file,
        "prompt_jsonl": prompt_jsonl,
        "label_spec": label_spec,
        "reference_jsonl": reference_jsonl,
        "sample_output": sample_output,
        "trial_manifest": trial_manifest,
        "wekws_label_file": wekws_label_file,
        "wekws_score_file": wekws_score_file,
        "wekws_frame_score_file": wekws_frame_score_file,
        "keyword": keyword,
        "samples_jsonl": samples_jsonl,
    }
    kwargs.update({key: value for key, value in cli_values.items() if value is not None})
    if task == "kws":
        kwargs["macro_recall_false_alarms"] = macro_recall_false_alarms
    if task in AUDIO_SAMPLE_TASKS:
        kwargs.update(
            _audio_sample_kwargs(
                task, pipeline, samples_jsonl=samples_jsonl, device=device, cache_dir=cache_dir
            )
        )
    kwargs["output_dir"] = output_dir
    _validate_required_args(pipeline, kwargs)
    if task in AUDIO_SAMPLE_TASKS:
        kwargs.pop("samples_jsonl", None)
    report = run_task(task, **kwargs)
    output_path = Path(output_dir)
    return {
        "status": "ok",
        "task": report.task,
        "metric": report.metric,
        "score": report.score,
        "pipeline_id": report.pipeline_id,
        "output_dir": str(output_path),
        "report_path": str(output_path / "report.json"),
        "pipeline_description_path": str(output_path / "pipeline_description.json"),
        "environment_note": ENVIRONMENT_NOTE,
        "node_config_paths": _node_config_paths(pipeline),
    }


def validate_pipeline_selection(pipeline: dict[str, Any]) -> None:
    """Validate node selections against the choices emitted by describe."""

    for slot in pipeline.get("pipeline") or ():
        selected = slot.get("selected")
        choices = slot.get("choices") or []
        if selected is None:
            if not slot.get("nullable", False):
                raise ValueError(f"Pipeline slot {slot.get('slot')!r} is not nullable")
            continue
        if selected == "default":
            if not slot.get("default"):
                raise ValueError(f"Pipeline slot {slot.get('slot')!r} has no default node")
            continue
        if selected not in choices:
            raise ValueError(
                f"Node {selected!r} is not declared in choices for slot {slot.get('slot')!r}"
            )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_task(task: str) -> str:
    return task.strip().lower().replace("-", "_")


def _node_config_paths(pipeline: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for node in pipeline.get("nodes") or ():
        path = node.get("manifest_path")
        if path and path not in paths:
            paths.append(path)
    return paths


def _describe_kwargs(
    task: str,
    *,
    original_task: str,
    language: str | None,
    metric: str | None,
    pipeline_id: str | None,
) -> dict[str, Any]:
    if task == "asr":
        kwargs: dict[str, Any] = {}
        if language:
            kwargs["language"] = language
        if metric:
            kwargs["metric"] = metric
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task == "s2tt":
        kwargs = {"language": language or "zh", "metric": metric or "bleu"}
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task == "kws":
        kwargs = {"metric": metric or "accuracy"}
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task == "vad":
        kwargs = {"metric": metric or "f1"}
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task in {"classification", "ser", "gr"}:
        # scripts/run.py already forwards the correct task alias for SER/GR.
        return {"pipeline_id": pipeline_id} if pipeline_id else {}
    if task == "slu":
        kwargs = {"metric": metric or "accuracy"}
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task == "sd":
        kwargs = {"metric": metric or "der"}
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task == "sv":
        kwargs = {}
        if metric:
            kwargs["metrics"] = split_metric_csv(metric)
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task == "sa_asr":
        kwargs = {"metric": metric or "cpwer", "language": language or "en"}
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task in {"tts", "vc"}:
        kwargs = {}
        if language or not pipeline_id:
            kwargs["language"] = language or "zh"
        if metric:
            kwargs["metrics"] = split_metric_csv(metric)
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task in {"se", "speech_enhancement"}:
        kwargs = {"language": "n/a"}
        if metric:
            kwargs["metrics"] = split_metric_csv(metric)
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task == "tse":
        kwargs = {}
        if language or not pipeline_id:
            kwargs["language"] = language or "zh"
        if metric:
            kwargs["metrics"] = split_metric_csv(metric)
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    return {}


def _route_choices(routes: dict[str, Any], *, language: str | None = None) -> list[dict[str, Any]]:
    choices = []
    for route in routes.get("routes") or ():
        if language and route.get("language") and route["language"] != language:
            continue
        route_language = route.get("language") or language or "n/a"
        pipeline_id = route.get("pipeline_id")
        choices.append(
            {
                "pipeline_id": str(pipeline_id).format(language=route_language) if pipeline_id else None,
                "language": route.get("language"),
                "metric": route.get("metric"),
                "nodes": list(route.get("nodes") or ()),
                "computation_node_ids": list(route.get("computation_nodes") or route.get("nodes") or ()),
                "input_contract": route.get("input_contract"),
                "executor": route.get("executor"),
                "selectors": {
                    key: value
                    for key, value in route.items()
                    if key
                    not in {
                        "pipeline_id",
                        "language",
                        "metric",
                        "nodes",
                        "computation_nodes",
                        "input_contract",
                        "executor",
                        "internal_executor_metric",
                        "executor_metric",
                        "aliases",
                        "family",
                    }
                },
            }
        )
    return choices


def _match_selected_routes(route_choices: list[dict[str, Any]], description) -> list[dict[str, Any]]:
    if description.member_pipeline_ids:
        return [
            _match_route(route_choices, "pipeline_id", pipeline_id)
            for pipeline_id in description.member_pipeline_ids
        ]
    return [_match_route(route_choices, "pipeline_id", description.pipeline_id)]


def _match_route(route_choices: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for route in route_choices:
        if route.get(key) == value:
            return route
    return {key: value, "pipeline_id": value, "nodes": []}


def _node_slots(
    node_ids: tuple[str, ...],
    *,
    selected_route: dict[str, Any],
    route_choices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    route_nodes = selected_route.get("nodes") or list(node_ids)
    slots: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {}
    for index, node_id in enumerate(node_ids):
        stage = node_id.split("/", 1)[0]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        stage_choices = _stage_choices(stage, route_choices)
        slots.append(
            {
                "slot": _slot_name(stage, stage_counts[stage], node_id),
                "stage": stage,
                "selected": "default",
                "default": node_id,
                "nullable": stage != "scoring",
                "metric": selected_route.get("metric") if stage == "scoring" else None,
                "choices": stage_choices or [node_id],
            }
        )
    if not slots and route_nodes:
        stage_counts = {}
        for index, node_id in enumerate(route_nodes):
            stage = node_id.split("/", 1)[0]
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            slots.append(
                {
                    "slot": _slot_name(stage, stage_counts[stage], node_id),
                    "stage": stage,
                    "selected": "default",
                    "default": node_id,
                    "nullable": stage != "scoring",
                    "metric": selected_route.get("metric") if stage == "scoring" else None,
                    "choices": _stage_choices(stage, route_choices) or [node_id],
                }
            )
    return slots


def _stage_choices(stage: str, route_choices: list[dict[str, Any]]) -> list[str]:
    choices: list[str] = []
    for route in route_choices:
        for node_id in route.get("nodes") or ():
            if node_id.startswith(f"{stage}/") and node_id not in choices:
                choices.append(node_id)
    return choices


def _slot_name(stage: str, stage_index: int, node_id: str) -> str:
    if stage in {"normalization", "scoring", "transcription"}:
        return f"{stage}_{stage_index}" if stage_index > 1 else stage
    return node_id.replace("/", "_")


def _run_kwargs_from_pipeline(pipeline: dict[str, Any]) -> dict[str, Any]:
    task = normalize_task(str(pipeline["task"]))
    kwargs: dict[str, Any] = {}
    if (
        task not in AUDIO_SAMPLE_TASKS
        and pipeline.get("language")
        and pipeline["language"] != "n/a"
    ):
        kwargs["language"] = pipeline["language"]
    if task == "classification":
        kwargs["task"] = pipeline.get("task_alias") or "classification"
        if pipeline.get("pipeline_id"):
            kwargs["pipeline_id"] = pipeline["pipeline_id"]
    elif task in {"ser", "gr", "slu"}:
        if pipeline.get("pipeline_id"):
            kwargs["pipeline_id"] = pipeline["pipeline_id"]
    elif task in AUDIO_SAMPLE_TASKS or task == "sv":
        use_atomic_pipeline_id = (
            pipeline.get("pipeline_kind") == "atomic" and pipeline.get("pipeline_id")
        )
        metrics = _metrics_from_pipeline(pipeline, task=task)
        if metrics and not use_atomic_pipeline_id:
            kwargs["metrics"] = metrics
        if use_atomic_pipeline_id:
            kwargs["pipeline_id"] = pipeline["pipeline_id"]
    elif pipeline.get("pipeline_id"):
        kwargs["pipeline_id"] = pipeline["pipeline_id"]
        if pipeline.get("requested_metric"):
            kwargs["metric"] = pipeline["requested_metric"]
        elif pipeline.get("metric"):
            kwargs["metric"] = pipeline["metric"]
    elif pipeline.get("requested_metric"):
        kwargs["metric"] = pipeline["requested_metric"]
    elif pipeline.get("metric"):
        kwargs["metric"] = pipeline["metric"]
    return kwargs


def _validate_required_args(pipeline: dict[str, Any], kwargs: dict[str, Any]) -> None:
    required_args = [
        ROLE_TO_CLI_ARG.get(role, role) for role in pipeline.get("required_roles") or ()
    ]
    required_args.append("output_dir")
    missing = [arg for arg in required_args if not kwargs.get(arg)]
    if missing:
        raise ValueError(f"Missing required CLI argument(s): {', '.join(missing)}")


def split_metric_csv(metric: str | None) -> tuple[str, ...]:
    if metric is None:
        return ()
    return tuple(item.strip().lower() for item in str(metric).split(",") if item.strip())


def _requested_metrics(
    task: str, *, metric: str | None, description_metric: str
) -> tuple[str, ...]:
    if task in AUDIO_SAMPLE_TASKS and metric:
        return split_metric_csv(metric)
    if task in SE_TASKS and description_metric == "multi":
        return SE_DEFAULT_METRICS
    if task == "sv" and description_metric == "multi":
        return SV_DEFAULT_METRICS
    if description_metric and description_metric != "multi":
        return (description_metric,)
    return ()


def _audio_sample_kwargs(
    task: str,
    pipeline: dict[str, Any],
    *,
    samples_jsonl: str | None,
    device: str,
    cache_dir: str | None,
) -> dict[str, Any]:
    if not samples_jsonl:
        return {}
    metrics = tuple(str(metric).lower() for metric in pipeline.get("metrics") or ())
    if not metrics and task in SE_TASKS:
        metrics = SE_DEFAULT_METRICS
    elif not metrics:
        metrics = (str(pipeline["metric"]),)
    if task == "tts":
        from sure_eval.evaluation.audio_runtime import build_tts_runtime
        from sure_eval.evaluation.audio_samples import load_tts_samples_jsonl

        samples = load_tts_samples_jsonl(samples_jsonl, metrics=metrics)
        runtime = build_tts_runtime(
            metrics=metrics,
            language=samples[0].language,
            device=device,
            cache_dir=cache_dir,
            transcription_node_id=_semantic_transcription_node_from_pipeline(pipeline),
        )
    elif task == "vc":
        from sure_eval.evaluation.audio_runtime import build_vc_runtime
        from sure_eval.evaluation.audio_samples import load_vc_samples_jsonl

        samples = load_vc_samples_jsonl(samples_jsonl, metrics=metrics)
        runtime = build_vc_runtime(
            metrics=metrics,
            language=samples[0].language,
            device=device,
            cache_dir=cache_dir,
            transcription_node_id=_semantic_transcription_node_from_pipeline(pipeline),
        )
    elif task in {"se", "speech_enhancement"}:
        from sure_eval.evaluation.audio_runtime import build_se_runtime
        from sure_eval.evaluation.audio_samples import load_se_samples_jsonl

        samples = load_se_samples_jsonl(samples_jsonl, metrics=metrics)
        runtime = build_se_runtime(
            metrics=metrics,
            device=device,
            cache_dir=cache_dir,
        )
    elif task == "tse":
        from sure_eval.evaluation.audio_runtime import build_tse_runtime
        from sure_eval.evaluation.audio_samples import load_tse_samples_jsonl

        samples = load_tse_samples_jsonl(samples_jsonl, metrics=metrics)
        runtime = build_tse_runtime(
            metrics=metrics,
            language=samples[0].language,
            device=device,
            cache_dir=cache_dir,
            transcription_node_id=_semantic_transcription_node_from_pipeline(pipeline),
        )
    else:
        return {}
    payload = {
        "samples": samples,
        "mos_providers": runtime.get("mos_providers", {}),
    }
    if "transcribers" in runtime:
        payload["transcribers"] = runtime["transcribers"]
    if "speaker_providers" in runtime:
        payload["speaker_providers"] = runtime["speaker_providers"]
    if "reference_providers" in runtime:
        payload["reference_providers"] = runtime["reference_providers"]
    return payload


def _metrics_from_pipeline(pipeline: dict[str, Any], *, task: str) -> tuple[str, ...]:
    if pipeline.get("metrics"):
        return tuple(str(metric).lower() for metric in pipeline["metrics"])
    if pipeline.get("metric") and pipeline["metric"] != "multi":
        return (str(pipeline["metric"]).lower(),)
    metrics = [
        slot.get("metric")
        for slot in pipeline.get("pipeline") or ()
        if slot.get("stage") == "scoring"
    ]
    selected = tuple(str(metric).lower() for metric in metrics if metric)
    if selected:
        return selected
    if task in SE_TASKS and pipeline.get("metric") == "multi":
        return SE_DEFAULT_METRICS
    if task == "sv" and pipeline.get("metric") == "multi":
        return SV_DEFAULT_METRICS
    return ()


def _semantic_transcription_node_from_pipeline(pipeline: dict[str, Any]) -> str | None:
    for node in pipeline.get("nodes") or ():
        node_id = str(node.get("node_id") or "")
        if node_id.startswith("transcription/"):
            return node_id
    return None
