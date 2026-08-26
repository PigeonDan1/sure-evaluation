#!/usr/bin/env python3
"""Generate the pipeline catalog from registered atomic routes and bundle presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from sure_eval.evaluation.cli_adapters import TASK_ALIASES, list_metric_routes
from sure_eval.evaluation.scripts import describe_pipeline
from sure_eval.evaluation.scripts.contracts import load_task_manifest, load_task_routes

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "pipeline_catalog.jsonl"

DISCOVERY_TASKS = (
    "asr",
    "s2tt",
    "sd",
    "sa_asr",
    "tts",
    "vc",
    "se",
    "tse",
    "classification",
    "ser",
    "gr",
    "slu",
    "kws",
    "vad",
)

# Multi pipelines are curated presets. Atomic pipelines are always discovered
# from tasks/<task>/routes.yaml and must never be added manually here.
BUNDLE_COMBINATIONS = (
    ("tts", {"language": "zh", "metrics": ("tts_cer", "sim/wavlm-large", "dnsmos")}),
    ("vc", {"language": "zh", "metrics": ("vc_cer", "sim/wavlm-large", "dnsmos")}),
    ("se", {"metrics": ("si-sdr", "stoi", "pesq", "dnsmos", "wv-mos", "utmos")}),
    ("tse", {"language": "zh", "metrics": ("si_sdr", "sim/wavlm-large", "dnsmos")}),
)


def iter_atomic_pipeline_ids() -> Iterator[tuple[str, str]]:
    """Yield every exact atomic pipeline id for documented task/language profiles."""

    seen: set[str] = set()
    for task in DISCOVERY_TASKS:
        for language in _task_language_profiles(task):
            inventory = list_metric_routes(task, language=language)
            for route in inventory["routes"]:
                pipeline_id = str(route["pipeline_id"])
                if pipeline_id in seen:
                    continue
                seen.add(pipeline_id)
                yield task, pipeline_id


def build_catalog_rows() -> list[dict[str, Any]]:
    rows = [
        _description_row(task, describe_pipeline(task, pipeline_id=pipeline_id))
        for task, pipeline_id in iter_atomic_pipeline_ids()
    ]
    rows.extend(
        _description_row(task, describe_pipeline(task, **kwargs))
        for task, kwargs in BUNDLE_COMBINATIONS
    )
    return rows


def _task_language_profiles(task: str) -> tuple[str | None, ...]:
    route_task = TASK_ALIASES.get(task, task)
    routes, _ = load_task_routes(route_task)
    has_language_template = any(
        "{language}" in str(route.get("pipeline_id") or "")
        for route in routes.get("routes") or ()
    )
    if not has_language_template:
        return (None,)

    manifest, _ = load_task_manifest(route_task)
    profiles: list[str] = []
    for field in ("default_metrics", "default_pipelines"):
        values = manifest.get(field)
        if isinstance(values, dict):
            for language in values:
                normalized = str(language).lower()
                if normalized not in profiles:
                    profiles.append(normalized)
    if not profiles:
        raise ValueError(f"Task {task!r} has templated routes but no declared language profiles")
    return tuple(profiles)


def _description_row(task_alias: str, description: Any) -> dict[str, Any]:
    return {
        "task": description.task,
        "task_alias": task_alias,
        "language": description.language,
        "metric": description.metric,
        "pipeline_id": description.pipeline_id,
        "pipeline_kind": description.pipeline_kind,
        "member_pipeline_ids": list(description.member_pipeline_ids),
        "execution_metrics": list(description.execution_metrics),
        "nodes": list(description.node_ids),
        "computation_node_ids": list(description.computation_node_ids),
        "task_config_path": description.task_config_path,
        "route_config_path": description.route_config_path,
        "describe_entrypoint": description.describe_entrypoint,
        "script_entrypoint": description.script_entrypoint,
        "executor": description.executor,
        "required_roles": list(description.required_roles),
        "optional_roles": list(description.optional_roles),
    }


def main() -> None:
    rows = build_catalog_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
