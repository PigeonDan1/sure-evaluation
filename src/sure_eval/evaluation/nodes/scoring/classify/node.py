"""Generic label-classification accuracy scoring."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import punctuation
from typing import Any

import yaml

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult

NODE_ID = "scoring/classify"
NODE_VERSION = "v1"


@dataclass(frozen=True)
class LabelSpec:
    """Dataset or task-specific label normalization rules."""

    id: str
    task: str = "classification"
    labels: tuple[dict[str, Any], ...] = ()
    unknown_policy: str = "invalid"
    case_sensitive: bool = False
    strip_punctuation: bool = True
    alias_to_id: dict[str, str] = field(default_factory=dict)

    def normalize(self, value: str) -> str | None:
        normalized = _normalize_value(
            value,
            case_sensitive=self.case_sensitive,
            strip_punctuation=self.strip_punctuation,
        )
        if normalized in self.alias_to_id:
            return self.alias_to_id[normalized]
        return None if self.unknown_policy == "invalid" else normalized

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "unknown_policy": self.unknown_policy,
            "labels": list(self.labels),
        }


def load_label_spec(path_or_payload: str | Path | dict[str, Any] | LabelSpec | None = None, *, task: str | None = None) -> LabelSpec:
    """Load a label spec from YAML/JSON, a mapping, or a built-in task default."""

    if isinstance(path_or_payload, LabelSpec):
        return path_or_payload
    if path_or_payload is None:
        return default_label_spec(task or "classification")
    if isinstance(path_or_payload, dict):
        return _label_spec_from_payload(path_or_payload)
    path = Path(path_or_payload)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    return _label_spec_from_payload(payload)


def default_label_spec(task: str) -> LabelSpec:
    normalized = task.upper()
    if normalized == "SER":
        return _label_spec_from_payload(
            {
                "id": "ser_default",
                "task": "SER",
                "labels": [
                    {"id": "neu", "aliases": ["neutral"], "numeric_ids": [0]},
                    {"id": "hap", "aliases": ["happy", "happiness"], "numeric_ids": [1]},
                    {"id": "ang", "aliases": ["angry", "anger"], "numeric_ids": [2]},
                    {"id": "sad", "aliases": ["sadness"], "numeric_ids": [3]},
                ],
                "unknown_policy": "invalid",
            }
        )
    if normalized == "GR":
        return _label_spec_from_payload(
            {
                "id": "gr_default",
                "task": "GR",
                "labels": [
                    {"id": "man", "aliases": ["male", "m"], "numeric_ids": [0]},
                    {"id": "woman", "aliases": ["female", "f"], "numeric_ids": [1]},
                ],
                "unknown_policy": "invalid",
            }
        )
    if normalized == "SLU":
        return _label_spec_from_payload(
            {
                "id": "slu_prompt_choice_id",
                "task": "SLU",
                "labels": [],
                "unknown_policy": "keep",
            }
        )
    if normalized in {"CLASSIFICATION", "ACCURACY"}:
        return _label_spec_from_payload(
            {
                "id": "legacy_classification_accuracy",
                "task": "classification",
                "labels": [
                    {"id": "neu", "aliases": ["neutral"], "numeric_ids": [0]},
                    {"id": "hap", "aliases": ["happy", "happiness"], "numeric_ids": [1]},
                    {"id": "ang", "aliases": ["angry", "anger"], "numeric_ids": [2]},
                    {"id": "sad", "aliases": ["sadness"], "numeric_ids": [3]},
                    {"id": "man", "aliases": ["male", "m"]},
                    {"id": "woman", "aliases": ["female", "f"]},
                ],
                "unknown_policy": "keep",
            }
        )
    return _label_spec_from_payload(
        {
            "id": f"{task.lower()}_identity",
            "task": task,
            "labels": [],
            "unknown_policy": "keep",
        }
    )


def score_classification_files(
    *,
    ref_file: str,
    hyp_file: str,
    label_spec: LabelSpec | str | Path | dict[str, Any] | None = None,
    task: str = "classification",
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Score aligned key-label files with a label spec."""

    spec = load_label_spec(label_spec, task=task)
    ref_rows = _read_key_text(ref_file)
    hyp_rows = _read_key_text(hyp_file)
    result = score_classification_rows(ref_rows, hyp_rows, label_spec=spec)
    return (
        KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file),
        PipelineNodeResult(
            stage="scoring",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "backend": "classify",
                "metric": "accuracy",
                "label_spec": spec.as_dict(),
                "result": result,
            },
            internal_stages=("label_normalization", "key_alignment", "accuracy"),
        ),
    )


def score_classification_rows(
    references: list[tuple[str, str]],
    predictions: list[tuple[str, str]],
    *,
    label_spec: LabelSpec,
) -> dict[str, Any]:
    """Score aligned reference/prediction rows."""

    ref_by_key = dict(references)
    hyp_by_key = dict(predictions)
    per_sample: list[dict[str, Any]] = []
    correct = 0
    valid = 0
    for key, raw_ref in references:
        raw_pred = hyp_by_key.get(key, "")
        ref_norm = label_spec.normalize(raw_ref)
        pred_norm = label_spec.normalize(raw_pred)
        ref_valid = ref_norm is not None
        pred_valid = pred_norm is not None
        is_valid = ref_valid and pred_valid and key in hyp_by_key
        is_correct = bool(is_valid and ref_norm == pred_norm)
        valid += 1 if is_valid else 0
        correct += 1 if is_correct else 0
        per_sample.append(
            {
                "key": key,
                "reference_raw": raw_ref,
                "prediction_raw": raw_pred,
                "reference": ref_norm,
                "prediction": pred_norm,
                "reference_valid": ref_valid,
                "prediction_valid": pred_valid,
                "correct": is_correct,
            }
        )

    total = len(references)
    return {
        "metric_name": "accuracy",
        "score": correct / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "missing_predictions": [key for key in ref_by_key if key not in hyp_by_key],
        "extra_predictions": [key for key in hyp_by_key if key not in ref_by_key],
        "label_spec_id": label_spec.id,
        "per_sample": per_sample,
    }


def _label_spec_from_payload(payload: dict[str, Any]) -> LabelSpec:
    labels = tuple(dict(label) for label in (payload.get("labels") or ()))
    case_sensitive = bool(payload.get("normalization", {}).get("case_sensitive", payload.get("case_sensitive", False)))
    strip_punctuation = bool(
        payload.get("normalization", {}).get("strip_punctuation", payload.get("strip_punctuation", True))
    )
    alias_to_id: dict[str, str] = {}
    for label in labels:
        label_id = str(label["id"])
        values = [label_id, label.get("display", "")]
        values.extend(label.get("aliases") or ())
        values.extend(str(item) for item in (label.get("numeric_ids") or ()))
        for value in values:
            if value == "":
                continue
            alias_to_id[
                _normalize_value(
                    str(value),
                    case_sensitive=case_sensitive,
                    strip_punctuation=strip_punctuation,
                )
            ] = label_id
    return LabelSpec(
        id=str(payload.get("id", "classification_labels")),
        task=str(payload.get("task", "classification")),
        labels=labels,
        unknown_policy=str(payload.get("unknown_policy", "invalid")),
        case_sensitive=case_sensitive,
        strip_punctuation=strip_punctuation,
        alias_to_id=alias_to_id,
    )


def _normalize_value(value: str, *, case_sensitive: bool, strip_punctuation: bool) -> str:
    normalized = value.strip()
    if not case_sensitive:
        normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if strip_punctuation:
        normalized = normalized.strip(punctuation + "，。！？；：（）【】《》“”‘’、")
    return normalized


def _read_key_text(path: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            rows.append(tuple(line.split("\t", 1)))
    return rows
