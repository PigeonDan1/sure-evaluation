"""KWS scoring at a shared FA/hour operating point."""

from __future__ import annotations

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import (
    DEFAULT_TARGET_FA_PER_HOUR,
    KWSMetric,
    KWSSample,
)

NODE_ID = "scoring/wekws_det"
NODE_VERSION = "v2"
INTERNAL_STAGES = (
    "keyword_normalization",
    "threshold_sweep",
    "fa_per_hour_operating_point",
    "per_keyword_recall",
    "trigger_latency",
)


def score_wekws_det(
    samples: list[KWSSample],
    *,
    target_fa_per_hour: float = DEFAULT_TARGET_FA_PER_HOUR,
    thresholds: list[float] | None = None,
    include_roc: bool = False,
    match_tolerance_sec: float = 1.0,
) -> PipelineNodeResult:
    metric = KWSMetric(
        target_fa_per_hour=target_fa_per_hour,
        thresholds=thresholds,
        include_roc=include_roc,
        match_tolerance_sec=match_tolerance_sec,
    ).calculate_samples(samples)
    details = metric.details
    point = details["operating_point"]
    latency = details["trigger_latency"]
    results = {
        "macro_recall": {
            "metric_name": "macro_recall",
            "score": metric.score,
            "details": {
                "target_fa_per_hour": target_fa_per_hour,
                "negative_hours": details["negative_hours"],
                **point,
            },
        },
        "per_keyword_recall": {
            "metric_name": "per_keyword_recall",
            "score": None,
            "details": point["per_keyword"],
        },
        "trigger_latency": {
            "metric_name": "trigger_latency",
            "score": latency["overall"]["mean_ms"],
            "details": latency,
        },
    }
    if include_roc:
        roc = details["roc"]
        results["roc_curve"] = {"metric_name": "roc_curve", "score": None, "details": roc}
        results["auc"] = {"metric_name": "auc", "score": roc.get("macro_auc"), "details": roc}
    summary = {
        "target_fa_per_hour": target_fa_per_hour,
        "negative_hours": details["negative_hours"],
        "threshold_scope": "global",
        "match_tolerance_sec": match_tolerance_sec,
        "num_samples": len(samples),
        **point,
    }
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "backend": "wekws_det",
            "metric": "macro_recall",
            "results": results,
            "rows": details["rows"],
            "summary": summary,
            "operating_curve": details["operating_curve"],
        },
        internal_stages=INTERNAL_STAGES + (("roc_auc",) if include_roc else ()),
    )
