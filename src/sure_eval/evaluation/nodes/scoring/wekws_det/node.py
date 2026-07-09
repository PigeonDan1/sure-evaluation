"""WekWS-style DET scoring wrapper for KWS."""

from __future__ import annotations

from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import (
    KWSMetric,
    KWSSample,
    build_rows,
    mean_score,
    summarize_det_curve,
)

NODE_ID = "scoring/wekws_det"
NODE_VERSION = "v1"
INTERNAL_STAGES = (
    "keyword_normalization",
    "threshold_decision",
    "det_curve",
    "operating_point_summary",
)


def score_wekws_det(
    samples: list[KWSSample],
    *,
    threshold: float = 0.5,
    thresholds: list[float] | None = None,
    threshold_step: float = 0.01,
) -> PipelineNodeResult:
    """Score aligned KWS samples with the existing WekWS-style DET semantics."""

    metric = KWSMetric(
        threshold=threshold,
        thresholds=thresholds,
        threshold_step=threshold_step,
    ).calculate_samples(samples)
    details = metric.details
    det_points = details["det_curve"]
    rows = build_rows(samples, threshold=threshold)
    summary = {
        "threshold": threshold,
        "num_samples": len(samples),
        "num_positive": details["positive_samples"],
        "num_negative": details["negative_samples"],
        "mean_score": mean_score(samples),
        **summarize_det_curve(det_points),
    }
    results = {
        "accuracy": _metric_dict(
            "accuracy",
            metric.score,
            correct=details["correct"],
            total=details["total"],
            threshold=threshold,
        ),
        "precision": _metric_dict("precision", float(details["precision"]), threshold=threshold),
        "recall": _metric_dict("recall", float(details["recall"]), threshold=threshold),
        "f1": _metric_dict("f1", float(details["f1"]), threshold=threshold),
        "false_reject_rate": _metric_dict(
            "false_reject_rate",
            float(details["false_reject_rate"]),
            false_rejects=details["false_rejects"],
            positives=details["positive_samples"],
            threshold=threshold,
        ),
        "false_alarm_rate": _metric_dict(
            "false_alarm_rate",
            float(details["false_alarm_rate"]),
            false_alarms=details["false_alarms"],
            negatives=details["negative_samples"],
            threshold=threshold,
        ),
        "false_alarm_per_hour": _metric_dict(
            "false_alarm_per_hour",
            float(details["false_alarm_per_hour"]),
            false_alarms=details["false_alarms"],
            threshold=threshold,
        ),
        "det_curve": {
            "metric_name": "det_curve",
            "score": float(summary.get("best_false_reject_rate", 0.0)),
            "details": {
                "points": det_points,
                "best": summarize_det_curve(det_points),
                "source": {
                    "primary_reference": "wenet-e2e/wekws",
                    "method": "WekWS compute_det / compute_det_ctc threshold sweep semantics",
                },
            },
        },
    }
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "backend": "wekws_det",
            "metric": "accuracy",
            "threshold": threshold,
            "threshold_step": threshold_step,
            "num_samples": len(samples),
            "results": results,
            "rows": rows,
            "summary": summary,
        },
        internal_stages=INTERNAL_STAGES,
    )


def _metric_dict(metric_name: str, score: float, **details: Any) -> dict[str, Any]:
    return {
        "metric_name": metric_name,
        "score": score,
        "details": details,
    }
