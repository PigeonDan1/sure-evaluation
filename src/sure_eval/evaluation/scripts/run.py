"""Unified dispatch helpers for task-level evaluation scripts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_TASK_MODULES = {
    "asr": "sure_eval.evaluation.scripts.asr",
    "s2tt": "sure_eval.evaluation.scripts.s2tt",
    "kws": "sure_eval.evaluation.scripts.kws",
    "classification": "sure_eval.evaluation.scripts.classification",
    "ser": "sure_eval.evaluation.scripts.classification",
    "gr": "sure_eval.evaluation.scripts.classification",
    "slu": "sure_eval.evaluation.scripts.slu",
    "sd": "sure_eval.evaluation.scripts.sd",
    "sa_asr": "sure_eval.evaluation.scripts.sa_asr",
    "se": "sure_eval.evaluation.scripts.se",
    "speech_enhancement": "sure_eval.evaluation.scripts.se",
    "tts": "sure_eval.evaluation.scripts.tts",
    "vc": "sure_eval.evaluation.scripts.vc",
    "tse": "sure_eval.evaluation.scripts.tse",
}


def describe_pipeline(task: str, **kwargs: Any):
    module = _load_task_module(task)
    if _normalize_task(task) in {"ser", "gr"}:
        kwargs.setdefault("task", task.upper())
    return module.describe_pipeline(**kwargs)


def run_task(task: str, **kwargs: Any):
    module = _load_task_module(task)
    if _normalize_task(task) in {"ser", "gr"}:
        kwargs.setdefault("task", task.upper())
    return module.run(**kwargs)


def _load_task_module(task: str):
    normalized = _normalize_task(task)
    if normalized not in _TASK_MODULES:
        raise ValueError(f"Unsupported evaluation script task: {task}")
    return import_module(_TASK_MODULES[normalized])


def _normalize_task(task: str) -> str:
    return task.strip().lower().replace("-", "_")
