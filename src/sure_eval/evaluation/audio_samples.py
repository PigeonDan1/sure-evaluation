"""Samples JSONL loaders for audio-object evaluation tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sure_eval.evaluation.tasks.tts.types import TTSSample
from sure_eval.evaluation.tasks.vc.types import VCSample


class SampleJsonlError(ValueError):
    """Raised when an audio samples JSONL file violates the input contract."""


def load_tts_samples_jsonl(path: str | Path, *, metrics: tuple[str, ...] | list[str] | None = None) -> list[TTSSample]:
    rows = _read_rows(Path(path))
    _validate_common(rows)
    requested = _normalize_metrics(metrics)
    samples: list[TTSSample] = []
    for row in rows:
        line_no = row["_line_no"]
        _require_fields(row, line_no, ("sample_id", "prediction_audio", "language"))
        if _has_semantic_metric(requested, "tts") and not row.get("reference_text"):
            _fail(line_no, "reference_text is required for semantic metric tts_wer/tts_cer")
        if _has_speaker_metric(requested) and not row.get("reference_audio"):
            _fail(line_no, f"reference_audio is required for speaker metric {_first_speaker_metric(requested)}")
        samples.append(
            TTSSample(
                prediction_audio=_resolve_existing_path(row["prediction_audio"], path, line_no, "prediction_audio"),
                reference_text=str(row.get("reference_text", "")),
                reference_audio=_resolve_optional_path(row.get("reference_audio"), path, line_no, "reference_audio"),
                language=str(row["language"]),
                sample_id=str(row["sample_id"]),
                metadata=_metadata(row, line_no),
            )
        )
    _validate_single_language(samples=[sample.language for sample in samples])
    return samples


def load_vc_samples_jsonl(path: str | Path, *, metrics: tuple[str, ...] | list[str] | None = None) -> list[VCSample]:
    rows = _read_rows(Path(path))
    _validate_common(rows)
    requested = _normalize_metrics(metrics)
    samples: list[VCSample] = []
    for row in rows:
        line_no = row["_line_no"]
        _require_fields(row, line_no, ("sample_id", "converted_audio", "language"))
        if _has_semantic_metric(requested, "vc") and not row.get("reference_text") and not row.get("reference_audio"):
            _fail(line_no, "reference_text or reference_audio is required for semantic metric vc_wer/vc_cer")
        if _has_speaker_metric(requested) and not row.get("reference_audio"):
            _fail(line_no, f"reference_audio is required for speaker metric {_first_speaker_metric(requested)}")
        samples.append(
            VCSample(
                converted_audio=_resolve_existing_path(row["converted_audio"], path, line_no, "converted_audio"),
                reference_audio=_resolve_optional_path(row.get("reference_audio"), path, line_no, "reference_audio"),
                source_audio=_resolve_optional_path(row.get("source_audio"), path, line_no, "source_audio"),
                reference_text=str(row.get("reference_text", "")),
                language=str(row["language"]),
                sample_id=str(row["sample_id"]),
                metadata=_metadata(row, line_no),
            )
        )
    _validate_single_language(samples=[sample.language for sample in samples])
    return samples


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SampleJsonlError(f"samples_jsonl file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                _fail(line_no, f"Invalid JSON: {exc.msg}")
            if not isinstance(row, dict):
                _fail(line_no, "sample row must be a JSON object")
            row["_line_no"] = line_no
            rows.append(row)
    if not rows:
        raise SampleJsonlError(f"samples_jsonl contains no rows: {path}")
    return rows


def _validate_common(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        line_no = row["_line_no"]
        sample_id = row.get("sample_id")
        if sample_id is None or str(sample_id) == "":
            _fail(line_no, "sample_id is required")
        sample_id = str(sample_id)
        if sample_id in seen:
            _fail(line_no, f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)


def _require_fields(row: dict[str, Any], line_no: int, fields: tuple[str, ...]) -> None:
    for field in fields:
        if row.get(field) in (None, ""):
            _fail(line_no, f"{field} is required")


def _resolve_existing_path(value: Any, jsonl_path: str | Path, line_no: int, field: str) -> str:
    if value in (None, ""):
        _fail(line_no, f"{field} is required")
    resolved = _resolve_path(value, jsonl_path)
    if not resolved.exists():
        _fail(line_no, f"{field} file not found: {resolved}")
    return str(resolved)


def _resolve_optional_path(value: Any, jsonl_path: str | Path, line_no: int, field: str) -> str:
    if value in (None, ""):
        return ""
    resolved = _resolve_path(value, jsonl_path)
    if not resolved.exists():
        _fail(line_no, f"{field} file not found: {resolved}")
    return str(resolved)


def _resolve_path(value: Any, jsonl_path: str | Path) -> Path:
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(jsonl_path).parent / candidate
    return candidate.resolve()


def _metadata(row: dict[str, Any], line_no: int) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        _fail(line_no, "metadata must be an object when present")
    return dict(metadata)


def _normalize_metrics(metrics: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    return tuple(str(metric).lower() for metric in (metrics or ()))


def _has_semantic_metric(metrics: tuple[str, ...], prefix: str) -> bool:
    return any(metric in {f"{prefix}_wer", f"{prefix}_cer"} for metric in metrics)


def _has_speaker_metric(metrics: tuple[str, ...]) -> bool:
    return any(metric.startswith("sim/") for metric in metrics)


def _first_speaker_metric(metrics: tuple[str, ...]) -> str:
    return next((metric for metric in metrics if metric.startswith("sim/")), "sim/*")


def _validate_single_language(*, samples: list[str]) -> None:
    languages = {language for language in samples if language}
    if len(languages) != 1:
        raise SampleJsonlError(f"samples_jsonl must contain exactly one language, got: {sorted(languages)}")


def _fail(line_no: int, message: str) -> None:
    raise SampleJsonlError(f"line {line_no}: {message}")
