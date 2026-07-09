"""VC task sample types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VCSample:
    """One converted-audio sample with references needed by VC metrics."""

    converted_audio: str
    reference_audio: str
    source_audio: str = ""
    reference_text: str = ""
    language: str = "en"
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["VCSample"]
