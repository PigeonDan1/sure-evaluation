"""Shared, model-independent conversation timelines, measured in exact seconds."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


def json_safe(value: Any) -> Any:
    """Keep decimal literals exact when exporting a trace to standard JSON."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class SpeechSegment:
    id: str
    start: Decimal
    end: Decimal

    def as_dict(self) -> dict[str, str]:
        return {"id": self.id, "start": str(self.start), "end": str(self.end)}


@dataclass(frozen=True)
class ConversationTimeline:
    key: str
    duration: Decimal
    user_segments: tuple[SpeechSegment, ...]
    system_segments: tuple[SpeechSegment, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "duration": str(self.duration),
            "user_segments": [segment.as_dict() for segment in self.user_segments],
            "system_segments": [segment.as_dict() for segment in self.system_segments],
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class DuplexTimelineBundle:
    conversations: tuple[ConversationTimeline, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "time_unit": "seconds",
            "time_origin": "conversation_start",
            "num_conversations": len(self.conversations),
            "num_user_segments": sum(len(row.user_segments) for row in self.conversations),
            "num_system_segments": sum(len(row.system_segments) for row in self.conversations),
            "input_order": "conversations_sorted_by_key_segments_preserved",
            "conversations": [row.as_dict() for row in self.conversations],
        }
