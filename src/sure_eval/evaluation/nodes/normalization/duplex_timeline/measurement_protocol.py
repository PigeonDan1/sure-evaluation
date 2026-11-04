"""Optional, score-independent provenance for measured full-duplex timelines.

Unknown fields are rejected inside this object and its named sections. All
sections and fields are optional for input validation. A complete recording
contains all five VAD fields, all three playback fields, reference.source, and
both preamble fields; capture and notes are optional even for completeness.
VAD numbers must have a finite binary64 representation and nonzero values must
remain nonzero, so reporting never silently overflows or underflows them.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import DuplexTimelineBundle


_SECTIONS = {
    "vad": {"name", "version", "threshold", "min_silence_ms", "padding_ms"},
    "playback": {"queue_policy", "clock", "speaker_output_observed"},
    "reference": {"source"},
    "preamble": {"removed", "reason"},
    "capture": {"model_id", "endpoint", "sample_rates"},
}
_COMPLETE_SECTIONS = ("vad", "playback", "reference", "preamble")
_QUEUE_POLICIES = {"retain_all_received_audio", "flush_on_server_speech_started", "other"}
_REFERENCE_SOURCES = {"human_annotation", "upstream_machine_schedule", "other"}


def validate_measurement_protocol(value: Any, *, location: str) -> None:
    """Validate a supplied object without filling fields or changing its values."""
    _object(value, set(_SECTIONS) | {"notes"}, location)
    for name, allowed in _SECTIONS.items():
        if name not in value:
            continue
        section = value[name]
        field = f"{location}.{name}"
        _object(section, allowed, field)
        for key, item in section.items():
            item_field = f"{field}.{key}"
            if (name, key) in {
                ("vad", "threshold"),
                ("vad", "min_silence_ms"),
                ("vad", "padding_ms"),
            }:
                _number(item, item_field)
                if key == "threshold" and not 0 <= item <= 1:
                    raise ValueError(f"{item_field} must be between 0 and 1 inclusive")
                if key != "threshold" and item < 0:
                    raise ValueError(f"{item_field} must be nonnegative")
            elif (name, key) in {
                ("playback", "speaker_output_observed"),
                ("preamble", "removed"),
            }:
                if type(item) is not bool:
                    raise ValueError(f"{item_field} must be a boolean")
            elif (name, key) == ("capture", "sample_rates"):
                if not isinstance(item, dict):
                    raise ValueError(f"{item_field} must be an object")
                for channel, rate in item.items():
                    _string(channel, f"{item_field} channel")
                    if type(rate) is not int or rate <= 0:
                        raise ValueError(f"{item_field}.{channel} must be a positive integer")
            else:
                _string(item, item_field)
                if (name, key) == ("playback", "queue_policy") and item not in _QUEUE_POLICIES:
                    raise ValueError(f"{item_field} must be one of {sorted(_QUEUE_POLICIES)}")
                if (name, key) == ("reference", "source") and item not in _REFERENCE_SOURCES:
                    raise ValueError(f"{item_field} must be one of {sorted(_REFERENCE_SOURCES)}")
    if "notes" in value and not isinstance(value["notes"], str):
        raise ValueError(f"{location}.notes must be a string")


def summarize_measurement_protocol(bundle: DuplexTimelineBundle) -> dict[str, Any]:
    """Preserve each role separately and report recording completeness.

    One complete role object records a conversation. An absent pair is missing;
    any other pair is partial. A bundle is recorded or missing only when every
    conversation has that status; mixed statuses and incomplete objects are
    partial. An empty bundle is missing. Numeric provenance values are exported
    as JSON numbers; exact timeline timestamps and other metadata are untouched.
    """
    conversations: dict[str, Any] = {}
    for conversation in bundle.conversations:
        objects = {
            role: conversation.metadata.get(role, {}).get("measurement_protocol")
            for role in ("reference", "prediction")
        }
        if any(_complete(value) for value in objects.values()):
            status = "recorded"
        elif all(value is None for value in objects.values()):
            status = "missing"
        else:
            status = "partial"
        conversations[conversation.key] = {
            "status": status,
            **{role: _report_value(value) for role, value in objects.items()},
        }
    statuses = {row["status"] for row in conversations.values()}
    status = next(iter(statuses)) if len(statuses) == 1 else "partial"
    return {"status": status if statuses else "missing", "conversations": conversations}


def _object(value: Any, allowed: set[str], field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"{field}: unknown fields {sorted(unknown)}")


def _string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _number(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{field} must be a finite JSON number")
    if isinstance(value, Decimal):
        finite = value.is_finite()
    else:
        finite = type(value) is int or math.isfinite(value)
    if not finite:
        raise ValueError(f"{field} must be a finite JSON number")
    try:
        report_number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be representable as a finite JSON number") from exc
    if not math.isfinite(report_number):
        raise ValueError(f"{field} must be representable as a finite JSON number")
    if value != 0 and report_number == 0:
        raise ValueError(f"{field} must be representable as a JSON number without underflow")


def _complete(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(value.get(name), dict) and _SECTIONS[name] <= value[name].keys()
        for name in _COMPLETE_SECTIONS
    )


def _report_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        numerator, denominator = value.as_integer_ratio()
        return numerator if denominator == 1 else float(value)
    if isinstance(value, dict):
        return {key: _report_value(item) for key, item in value.items()}
    return value
