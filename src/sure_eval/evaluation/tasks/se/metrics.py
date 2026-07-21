"""Speech enhancement metric definitions."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Dict, Union

from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.nodes.scoring._full_reference_audio import PESQProvider, SISDRProvider, STOIProvider

ScoreProvider = Callable[..., Union[float, Dict[str, Any]]]


@dataclass(frozen=True)
class MetricSource:
    """Reference implementation metadata for a metric definition."""

    primary_reference: str
    method: str
    score_key: str
    higher_is_better: bool
    dependencies: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_reference": self.primary_reference,
            "method": self.method,
            "score_key": self.score_key,
            "higher_is_better": self.higher_is_better,
            "dependencies": list(self.dependencies),
        }


class _FullReferenceMetric:
    metric_name = "full_reference_audio"
    source = MetricSource(
        primary_reference="",
        method="",
        score_key="score",
        higher_is_better=True,
    )

    def __init__(self, score_provider: ScoreProvider | None = None) -> None:
        self.score_provider = score_provider

    def calculate(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        return self.calculate_batch([prediction], [reference], **kwargs)

    def calculate_batch(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> MetricResult:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")
        if not predictions:
            raise ValueError("at least one prediction is required")
        provider = self._provider()
        rows = [
            self._normalize_provider_result(provider(prediction, reference, **kwargs))
            for prediction, reference in zip(predictions, references, strict=True)
        ]
        scores = [float(row[self.source.score_key]) for row in rows]
        return MetricResult(
            metric_name=self.metric_name,
            score=fmean(scores),
            details={
                "num_samples": len(rows),
                "score_key": self.source.score_key,
                "source": self.source.as_dict(),
                "per_sample": rows,
            },
        )

    def _provider(self) -> ScoreProvider:
        if self.score_provider is None:
            return self._default_provider()
        return self.score_provider

    def _default_provider(self) -> ScoreProvider:
        raise RuntimeError(f"{self.__class__.__name__} requires a score_provider")

    def _normalize_provider_result(self, raw_result: float | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw_result, (float, int)):
            return {self.source.score_key: float(raw_result)}
        if not isinstance(raw_result, dict):
            raise TypeError("score_provider must return a float or a dict")
        row = dict(raw_result)
        if self.source.score_key not in row:
            fallback = self._find_score_key(row)
            if fallback is None:
                raise KeyError(f"score_provider result must contain '{self.source.score_key}'")
            row[self.source.score_key] = row[fallback]
        row[self.source.score_key] = float(row[self.source.score_key])
        return row

    def _find_score_key(self, row: dict[str, Any]) -> str | None:
        for key in ("score",):
            if key in row:
                return key
        return None


class SISDRMetric(_FullReferenceMetric):
    """Scale-invariant SDR in dB for enhanced speech against clean speech."""

    metric_name = "si-sdr"
    source = MetricSource(
        primary_reference="Le Roux et al., SDR half-baked or well done?",
        method="Scale-invariant projection of enhanced speech onto clean reference speech, averaged in dB.",
        score_key="si_sdr",
        higher_is_better=True,
        dependencies=("numpy",),
    )

    def _default_provider(self) -> ScoreProvider:
        return SISDRProvider()

    def _find_score_key(self, row: dict[str, Any]) -> str | None:
        for key in ("sisdr", "si-sdr", "sdr", "score"):
            if key in row:
                return key
        return None


class STOIMetric(_FullReferenceMetric):
    """Short-time objective intelligibility for speech enhancement."""

    metric_name = "stoi"
    source = MetricSource(
        primary_reference="pystoi",
        method="Short-time objective intelligibility against clean reference speech.",
        score_key="stoi",
        higher_is_better=True,
        dependencies=("pystoi",),
    )

    def _default_provider(self) -> ScoreProvider:
        return STOIProvider()


class PESQMetric(_FullReferenceMetric):
    """Wide-band PESQ for speech enhancement."""

    metric_name = "pesq"
    source = MetricSource(
        primary_reference="pesq",
        method="Wide-band PESQ against clean reference speech at 16 kHz.",
        score_key="pesq",
        higher_is_better=True,
        dependencies=("pesq",),
    )

    def _default_provider(self) -> ScoreProvider:
        return PESQProvider()


__all__ = ["PESQMetric", "SISDRMetric", "STOIMetric"]
