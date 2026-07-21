"""TSE task sample types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TSESample:
    """One target-speaker-extraction sample with references needed by TSE metrics."""

    prediction_audio: str
    reference_audio: str
    mixed_audio: str = ""
    enrollment_audio: str = ""
    reference_text: str = ""
    language: str = "en"
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["TSESample"]