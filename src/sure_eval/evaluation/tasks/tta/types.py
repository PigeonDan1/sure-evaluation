"""Text-to-audio task sample types."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TTASample:
    """One generated audio clip and its optional references."""

    prediction_audio: str
    reference_audio: str = ""
    caption: str = ""
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["TTASample"]
