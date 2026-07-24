"""Adapters between the metric CLI and configured evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sure_eval.evaluation.scripts import describe_pipeline, run_task
from sure_eval.evaluation.scripts.contracts import (
    load_task_manifest,
    load_task_routes,
)

ROLE_TO_CLI_ARG = {
    "ref": "ref_file",
    "hyp": "hyp_file",
    "src": "src_file",
    "prompt_jsonl": "prompt_jsonl",
    "label_spec": "label_spec",
    "reference_jsonl": "reference_jsonl",
    "sample_output": "sample_output",
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
    if pipeline_id and normalized_task in AUDIO_SAMPLE_TASKS:
        raise ValueError("Use metrics to describe audio bundle tasks; pipeline_id selection is for atomic file routes")
    describe_kwargs = _describe_kwargs(
        normalized_task,
        original_task=task,
        language=language,
        metric=metric,
        pipeline_id=pipeline_id,
    )
    description = describe_pipeline(normalized_task, **describe_kwargs)
    manifest, _ = load_task_manifest(TASK_ALIASES.get(normalized_task, normalized_task))
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
        "schema": "sure.metric.pipeline.v1",
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
    if task == "sa_asr":
        kwargs = {"metric": metric or "cpwer", "language": language or "en"}
        if pipeline_id:
            kwargs["pipeline_id"] = pipeline_id
        return kwargs
    if task in {"tts", "vc"}:
        kwargs = {"language": language or "zh"}
        if metric:
            kwargs["metrics"] = split_metric_csv(metric)
        return kwargs
    if task in {"se", "speech_enhancement"}:
        kwargs = {"language": "n/a"}
        if metric:
            kwargs["metrics"] = split_metric_csv(metric)
        return kwargs
    if task == "tse":
        kwargs = {"language": language or "zh"}
        if metric:
            kwargs["metrics"] = split_metric_csv(metric)
        return kwargs
    return {}


def _route_choices(routes: dict[str, Any], *, language: str | None = None) -> list[dict[str, Any]]:
    choices = []
    for route in routes.get("routes") or ():
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
    elif task in {"ser", "gr", "slu"}:
        pass
    elif task in AUDIO_SAMPLE_TASKS:
        metrics = _metrics_from_pipeline(pipeline, task=task)
        if metrics:
            kwargs["metrics"] = metrics
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
    metrics = tuple(_run_kwargs_from_pipeline(pipeline).get("metrics") or ())
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
    return ()
