"""TTS task sample and report types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sure_eval.evaluation.base import MetricResult


@dataclass(frozen=True)
class TTSSample:
    """One synthesized-audio sample with the references needed by TTS metrics."""

    prediction_audio: str
    reference_text: str
    reference_audio: str = ""
    language: str = "en"
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSMetricReport:
    """Structured output from the connected TTS metric pipeline."""

    results: dict[str, MetricResult]
    rows: list[dict[str, Any]]


__all__ = ["TTSMetricReport", "TTSSample"]
