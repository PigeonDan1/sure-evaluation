"""RIR task sample types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RIRSample:
    """One predicted/reference RIR pair."""

    prediction_rir: str
    reference_rir: str
    observed_audio: str = ""
    reference_audio: str = ""
    sample_rate: int | None = None
    channel_layout: str = "mono"
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["RIRSample"]
