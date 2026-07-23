"""Compatibility wrappers for the task-level KWS route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import KWSSample


@dataclass
class KWSMetricReport:
    """Structured KWS metric report."""

    results: dict[str, MetricResult]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _metric_from_dict(payload: dict[str, Any]) -> MetricResult:
    return MetricResult(
        metric_name=str(payload["metric_name"]),
        score=float(payload["score"]),
        details=dict(payload.get("details", {})),
    )


class KWSMetricPipeline:
    """Evaluate keyword spotting outputs against reference labels."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        thresholds: list[float] | None = None,
        threshold_step: float = 0.01,
        macro_recall_false_alarms: int = 0,
    ) -> None:
        self.threshold = threshold
        self.thresholds = thresholds
        self.threshold_step = threshold_step
        self.macro_recall_false_alarms = macro_recall_false_alarms

    def evaluate(self, samples: list[KWSSample]) -> KWSMetricReport:
        from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_samples

        report = evaluate_kws_samples(
            samples,
            threshold=self.threshold,
            thresholds=self.thresholds,
            threshold_step=self.threshold_step,
            macro_recall_false_alarms=self.macro_recall_false_alarms,
        )
        results = report.details["results"]
        return KWSMetricReport(
            results={
                "accuracy": _metric_from_dict(results["accuracy"]),
                "precision": _metric_from_dict(results["precision"]),
                "recall": _metric_from_dict(results["recall"]),
                "f1": _metric_from_dict(results["f1"]),
                "false_reject_rate": _metric_from_dict(results["false_reject_rate"]),
                "false_alarm_rate": _metric_from_dict(results["false_alarm_rate"]),
                "false_alarm_per_hour": _metric_from_dict(results["false_alarm_per_hour"]),
                "macro-recall": _metric_from_dict(results["macro-recall"]),
                "det_curve": MetricResult(
                    metric_name="det_curve",
                    score=float(results["det_curve"]["score"]),
                    details=results["det_curve"]["details"],
                ),
            },
            rows=report.details["rows"],
            summary=report.details["summary"],
        )


def report_to_dict(report: KWSMetricReport) -> dict[str, Any]:
    """Serialize a KWSMetricReport into JSON-compatible dictionaries."""
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


__all__ = [
    "KWSMetricPipeline",
    "KWSMetricReport",
    "report_to_dict",
]
