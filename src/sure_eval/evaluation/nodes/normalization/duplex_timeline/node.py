"""Validate and align two strict full-duplex JSONL timelines without editing events."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult

from .measurement_protocol import validate_measurement_protocol
from .types import ConversationTimeline, DuplexTimelineBundle, SpeechSegment

NODE_ID = "normalization/duplex_timeline"
NODE_VERSION = "v1"
INTERNAL_STAGES = (
    "exact_decimal_jsonl_parse",
    "schema_and_clock_validation",
    "channel_segment_validation",
    "exact_conversation_key_alignment",
)
_COMMON = {"schema_version", "key", "time_unit", "time_origin"}


def normalize_duplex_timeline(
    reference_jsonl: str | Path,
    sample_output: str | Path,
) -> tuple[DuplexTimelineBundle, PipelineNodeResult]:
    """Preserve segment order/overlap across channels; reject malformed timelines."""
    reference = _load_rows(reference_jsonl, role="reference")
    prediction = _load_rows(sample_output, role="prediction")
    if reference.keys() != prediction.keys():
        missing = sorted(reference.keys() - prediction.keys())
        extra = sorted(prediction.keys() - reference.keys())
        raise ValueError(f"conversation key mismatch: missing={missing}, extra={extra}")
    conversations = []
    for key in sorted(reference):
        ref, pred = reference[key], prediction[key]
        duration = _number(ref["duration"], f"reference {key!r}.duration")
        if duration <= 0:
            raise ValueError(f"reference {key!r}.duration must be positive")
        users = _segments(ref["user_segments"], duration, f"reference {key!r}.user_segments")
        if not users:
            raise ValueError(f"reference {key!r}.user_segments must not be empty")
        systems = _segments(
            pred["system_segments"], duration, f"prediction {key!r}.system_segments"
        )
        conversations.append(
            ConversationTimeline(
                key=key,
                duration=duration,
                user_segments=users,
                system_segments=systems,
                metadata={
                    "reference": ref.get("metadata", {}),
                    "prediction": pred.get("metadata", {}),
                },
            )
        )
    bundle = DuplexTimelineBundle(tuple(conversations))
    summary = {key: value for key, value in bundle.as_dict().items() if key != "conversations"}
    return bundle, PipelineNodeResult(
        stage="normalization",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={"backend": "duplex_timeline_jsonl", "input_summary": summary},
        internal_stages=INTERNAL_STAGES,
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _load_rows(path: str | Path, *, role: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    fields = {"duration", "user_segments"} if role == "reference" else {"system_segments"}
    required = _COMMON | fields
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{role} {path}:{line_number}"
            try:
                try:
                    payload = json.loads(
                        line,
                        parse_float=Decimal,
                        parse_constant=_reject_constant,
                        object_pairs_hook=_unique_object,
                    )
                except ValueError as exc:
                    if str(exc).startswith("non-finite JSON number"):
                        # Locate provenance constants without changing the legacy
                        # rejection of non-finite numbers elsewhere in a row.
                        parsed = json.loads(
                            line,
                            parse_float=Decimal,
                            parse_constant=Decimal,
                            object_pairs_hook=_unique_object,
                        )
                        _measurement_protocol(parsed)
                    raise
                if not isinstance(payload, dict):
                    raise ValueError("row must be a JSON object")
                _fields(payload, required, required | {"metadata"}, location)
                version = payload["schema_version"]
                if type(version) is not int or version != 1:
                    raise ValueError("schema_version must be integer 1")
                key = _identifier(payload["key"], "key")
                if payload["time_unit"] != "seconds":
                    raise ValueError("time_unit must be 'seconds'")
                if payload["time_origin"] != "conversation_start":
                    raise ValueError("time_origin must be 'conversation_start'")
                if "metadata" in payload and not isinstance(payload["metadata"], dict):
                    raise ValueError("metadata must be an object")
                _measurement_protocol(payload)
                if key in rows:
                    raise ValueError(f"duplicate conversation key {key!r}")
                rows[key] = payload
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{location}: {exc}") from exc
    if not rows:
        raise ValueError(f"{role} file must contain at least one conversation")
    return rows


def _measurement_protocol(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and "measurement_protocol" in metadata:
        validate_measurement_protocol(
            metadata["measurement_protocol"],
            location=f"conversation {payload.get('key')!r}.metadata.measurement_protocol",
        )


def _fields(payload: dict[str, Any], required: set[str], allowed: set[str], location: str) -> None:
    missing, unknown = required - payload.keys(), payload.keys() - allowed
    if missing:
        raise ValueError(f"{location}: missing required fields {sorted(missing)}")
    if unknown:
        raise ValueError(f"{location}: unknown fields {sorted(unknown)}")


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _number(value: Any, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError(f"{field} must be a finite JSON number")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError(f"{field} must be a finite JSON number")
    return result


def _segments(payload: Any, duration: Decimal, location: str) -> tuple[SpeechSegment, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{location} must be an array")
    segments: list[SpeechSegment] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        field = f"{location}[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{field} must be an object")
        _fields(item, {"id", "start", "end"}, {"id", "start", "end"}, field)
        segment_id = _identifier(item["id"], f"{field}.id")
        field = f"{field} (segment_id={segment_id!r})"
        if segment_id in seen:
            raise ValueError(f"{field}: duplicate segment id {segment_id!r}")
        seen.add(segment_id)
        start = _number(item["start"], f"{field}.start")
        end = _number(item["end"], f"{field}.end")
        if not 0 <= start < end <= duration:
            raise ValueError(f"{field} must satisfy 0 <= start < end <= duration")
        if segments and start <= segments[-1].start:
            raise ValueError(
                f"{field}: segment starts must be strictly increasing; "
                f"previous_segment_id={segments[-1].id!r}"
            )
        if segments and start < segments[-1].end:
            raise ValueError(
                f"{field}: segments on the same channel must not overlap; "
                f"previous_segment_id={segments[-1].id!r}"
            )
        segments.append(SpeechSegment(segment_id, start, end))
    return tuple(segments)
