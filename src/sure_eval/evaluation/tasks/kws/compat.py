"""Python convenience API for the current KWS evaluation contract."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import (
    DEFAULT_TARGET_FA_PER_HOUR,
    KWSSample,
)


@dataclass
class KWSMetricReport:
    results: dict[str, MetricResult]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


class KWSMetricPipeline:
    def __init__(
        self,
        *,
        target_fa_per_hour: float = DEFAULT_TARGET_FA_PER_HOUR,
        thresholds: list[float] | None = None,
        include_roc: bool = False,
        match_tolerance_sec: float = 1.0,
    ) -> None:
        self.options = dict(
            target_fa_per_hour=target_fa_per_hour,
            thresholds=thresholds,
            include_roc=include_roc,
            match_tolerance_sec=match_tolerance_sec,
        )

    def evaluate(self, samples: list[KWSSample]) -> KWSMetricReport:
        from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_samples

        report = evaluate_kws_samples(samples, **self.options)
        return KWSMetricReport(
            results={
                name: MetricResult(**payload) for name, payload in report.details["results"].items()
            },
            rows=report.details["rows"],
            summary=report.details["summary"],
        )


def report_to_dict(report: KWSMetricReport) -> dict[str, Any]:
    return {
        "metrics": {
            name: {
                "metric_name": result.metric_name,
                "score": result.score,
                "details": result.details,
            }
            for name, result in report.results.items()
        },
        "rows": report.rows,
        "summary": report.summary,
    }


__all__ = ["KWSMetricPipeline", "KWSMetricReport", "report_to_dict"]
