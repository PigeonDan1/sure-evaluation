"""Speech enhancement task sample and report types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sure_eval.evaluation.base import MetricResult


@dataclass(frozen=True)
class SESample:
    """One speech enhancement sample with optional noisy and clean references."""

    enhanced_audio: str
    noisy_audio: str = ""
    reference_audio: str = ""
    language: str = "n/a"
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SEMetricReport:
    """Structured output from the connected SE metric pipeline."""

    results: dict[str, MetricResult]
    rows: list[dict[str, Any]]


__all__ = ["SEMetricReport", "SESample"]
