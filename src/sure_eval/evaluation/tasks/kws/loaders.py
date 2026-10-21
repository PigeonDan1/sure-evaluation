"""Strict KWS input alignment and score/timestamp conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import (
    KWSEvent,
    KWSSample,
    KWSTrigger,
    normalize_keyword,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _index(rows: list[dict], source: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("key"), str) or not row["key"]:
            raise ValueError(f"{source} rows require a nonempty string key")
        key = row["key"]
        if key in indexed:
            raise ValueError(f"Duplicate {source} key: {key}")
        indexed[key] = row
    return indexed


def _align(references: dict, outputs: dict) -> None:
    missing, extra = references.keys() - outputs.keys(), outputs.keys() - references.keys()
    if missing or extra:
        raise ValueError(f"KWS key mismatch: missing={sorted(missing)}, extra={sorted(extra)}")


def _expected_from_reference(row: dict) -> tuple[bool, str | None]:
    if "events" in row:
        if not isinstance(row["events"], list):
            raise ValueError("Reference events must be a list")
        if any(key in row for key in ("expected", "label", "expected_detected")):
            explicit, _ = _expected_from_reference(
                {key: value for key, value in row.items() if key != "events"}
            )
            if explicit != bool(row["events"]):
                raise ValueError("Reference label contradicts events")
        return bool(row["events"]), None
    value = row.get("expected", row.get("label", row.get("expected_detected")))
    if isinstance(value, bool):
        expected = value
    elif value is None:
        expected = bool(str(row.get("txt", row.get("text", ""))).strip())
    else:
        normalized = str(value).strip().lower()
        if normalized not in {
            "detect",
            "detected",
            "positive",
            "true",
            "1",
            "yes",
            "reject",
            "rejected",
            "negative",
            "false",
            "0",
            "no",
        }:
            raise ValueError(f"Unknown KWS reference label: {value}")
        expected = normalized in {"detect", "detected", "positive", "true", "1", "yes"}
    keyword = row.get("expected_keyword", row.get("text", row.get("txt"))) if expected else None
    return expected, keyword


def _number(value: Any) -> float | None:
    return float(value) if value is not None else None


def _sample(ref: dict, result: dict) -> KWSSample:
    expected, keyword = _expected_from_reference(ref)
    events = None
    if "events" in ref:
        events = [
            KWSEvent(item["keyword"], _number(item.get("end_sec")), _number(item.get("start_sec")))
            for item in ref["events"]
        ]
    detections = None
    if "detections" in result:
        if not isinstance(result["detections"], list):
            raise ValueError("detections must be a list")
        detections = [
            KWSTrigger(
                item.get("keyword"), float(item["score"]), _number(item.get("trigger_time_sec"))
            )
            for item in result["detections"]
        ]
    return KWSSample(
        key=ref["key"],
        expected_detected=expected,
        expected_keyword=keyword,
        duration=_number(ref.get("duration")),
        detected=bool(result.get("detected", False)),
        predicted_keyword=result.get("keyword"),
        score=_number(result.get("score")),
        scores=[float(score) for score in result.get("scores", [])],
        keyword_end_sec=_number(ref.get("keyword_end_sec")),
        keyword_start_sec=_number(ref.get("keyword_start_sec")),
        trigger_time_sec=_number(result.get("trigger_time_sec")),
        frame_shift_sec=_number(result.get("frame_shift_sec", ref.get("frame_shift_sec"))),
        frame_offset_sec=float(result.get("frame_offset_sec", ref.get("frame_offset_sec", 0.0))),
        events=events,
        detections=detections,
        keyword_scores={
            key: float(value) for key, value in result.get("keyword_scores", {}).items()
        },
        metadata={"audio": ref.get("audio", ref.get("wav")), "raw": result.get("raw", {})},
    )


def load_samples_from_jsonl_and_outputs(
    reference_jsonl: Path | str, output_json: Path | str
) -> list[KWSSample]:
    references = _index(_load_jsonl(Path(reference_jsonl)), "reference")
    text = Path(output_json).read_text(encoding="utf-8")
    try:
        outputs = json.loads(text)
    except json.JSONDecodeError:
        outputs = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(outputs, dict):
        outputs = outputs.get("rows", [outputs] if "key" in outputs else None)
    if not isinstance(outputs, list):
        raise ValueError("KWS output must be a JSON list, an object with rows, or JSONL")
    outputs = _index(outputs, "output")
    _align(references, outputs)
    samples = []
    for key, ref in references.items():
        result = outputs[key].get("result", outputs[key])
        if not isinstance(result, dict):
            raise ValueError(f"KWS result must be an object: {key}")
        samples.append(_sample(ref, result))
    return samples


def _wekws_reference(ref: dict, keyword: str) -> dict:
    text = normalize_keyword(str(ref.get("txt", ref.get("text", "")))) or ""
    expected = normalize_keyword(keyword) in text
    return {**ref, "expected": expected, "expected_keyword": keyword if expected else None}


def load_samples_from_wekws_score_file(
    label_file: Path | str, score_file: Path | str, *, keyword: str
) -> list[KWSSample]:
    if not normalize_keyword(keyword):
        raise ValueError("keyword must not be empty")
    labels = _index(_load_jsonl(Path(label_file)), "reference")
    outputs = []
    for line in Path(score_file).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) not in {4, 5} or parts[1] not in {"detected", "rejected"}:
            raise ValueError(
                "WeKWS FA/hour scoring requires an uncensored score on every row: key detected|rejected keyword score [trigger_time_sec]"
            )
        outputs.append(
            {
                "key": parts[0],
                "keyword": parts[2],
                "score": float(parts[3]),
                "trigger_time_sec": float(parts[4]) if len(parts) == 5 else None,
            }
        )
    indexed = _index(outputs, "score")
    _align(labels, indexed)
    return [_sample(_wekws_reference(ref, keyword), indexed[key]) for key, ref in labels.items()]


def load_samples_from_wekws_frame_score_file(
    label_file: Path | str, score_file: Path | str, *, keyword: str
) -> list[KWSSample]:
    if not normalize_keyword(keyword):
        raise ValueError("keyword must not be empty")
    labels = _index(_load_jsonl(Path(label_file)), "reference")
    outputs = []
    for line in Path(score_file).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 3:
            raise ValueError("WeKWS frame rows require key keyword score...")
        if normalize_keyword(parts[1]) == normalize_keyword(keyword):
            outputs.append(
                {"key": parts[0], "keyword": keyword, "scores": [float(item) for item in parts[2:]]}
            )
    indexed = _index(outputs, "frame score")
    _align(labels, indexed)
    return [_sample(_wekws_reference(ref, keyword), indexed[key]) for key, ref in labels.items()]


__all__ = [
    "load_samples_from_jsonl_and_outputs",
    "load_samples_from_wekws_score_file",
    "load_samples_from_wekws_frame_score_file",
]
