"""Duration-overlap scoring for VAD speech segment predictions."""

from __future__ import annotations

from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.vad_timebase import VADNormalizedBundle
from sure_eval.evaluation.nodes.validation.vad_contract import DETECTION_METRICS, Segment

NODE_ID = "scoring/vad_detection_duration"
NODE_VERSION = "v1"
INTERNAL_STAGES = (
    "interval_intersection",
    "duration_confusion",
    "micro_average",
    "cost_summary",
)


def score_vad_detection_duration(bundle: VADNormalizedBundle) -> PipelineNodeResult:
    """Compute duration-based VAD F1, false alarm, miss, and NIST-style DCF."""

    scored_rows = [row for row in bundle.rows if row.prediction_segments is not None]
    skipped = [
        {
            "key": row.key,
            "metrics": list(DETECTION_METRICS),
            "reason": row.skipped_metrics.get("f1", "missing prediction field: speech_segments"),
        }
        for row in bundle.rows
        if row.prediction_segments is None
    ]

    per_sample: list[dict[str, Any]] = []
    total_duration_sec = 0.0
    tp_sec = 0.0
    fp_sec = 0.0
    fn_sec = 0.0
    speech_scored_sec = 0.0
    nonspeech_scored_sec = 0.0

    for row in scored_rows:
        assert row.prediction_segments is not None
        row_ref_sec = _duration(row.reference_segments)
        row_hyp_sec = _duration(row.prediction_segments)
        row_tp_sec = _intersection_duration(row.reference_segments, row.prediction_segments)
        row_fp_sec = max(0.0, row_hyp_sec - row_tp_sec)
        row_fn_sec = max(0.0, row_ref_sec - row_tp_sec)
        row_nonspeech_sec = max(0.0, row.duration - row_ref_sec)

        total_duration_sec += row.duration
        tp_sec += row_tp_sec
        fp_sec += row_fp_sec
        fn_sec += row_fn_sec
        speech_scored_sec += row_ref_sec
        nonspeech_scored_sec += row_nonspeech_sec
        per_sample.append(
            {
                "key": row.key,
                "duration": row.duration,
                "tp_sec": row_tp_sec,
                "fp_sec": row_fp_sec,
                "fn_sec": row_fn_sec,
                "reference_speech_sec": row_ref_sec,
                "prediction_speech_sec": row_hyp_sec,
                "nonspeech_scored_sec": row_nonspeech_sec,
            }
        )

    precision = _safe_div(tp_sec, tp_sec + fp_sec)
    recall = _safe_div(tp_sec, tp_sec + fn_sec)
    f1 = _f1(precision, recall)
    p_fa = _safe_div(fp_sec, nonspeech_scored_sec)
    p_miss = _safe_div(fn_sec, speech_scored_sec)
    dcf_nist = None if p_fa is None or p_miss is None else 0.25 * p_fa + 0.75 * p_miss
    primary_scores = {
        "f1": f1,
        "p_fa": p_fa,
        "p_miss": p_miss,
        "dcf_nist": dcf_nist,
    }
    auxiliary = {
        "precision": precision,
        "recall": recall,
        "false_alarm_sec": fp_sec,
        "miss_sec": fn_sec,
        "true_positive_sec": tp_sec,
        "speech_scored_sec": speech_scored_sec,
        "nonspeech_scored_sec": nonspeech_scored_sec,
        "total_scored_sec": total_duration_sec,
        "num_detection_samples": len(scored_rows),
    }
    results = {
        metric: _metric_result(metric, primary_scores[metric], auxiliary)
        for metric in DETECTION_METRICS
    }
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "backend": "vad_duration_overlap",
            "primary_scores": primary_scores,
            "auxiliary": auxiliary,
            "results": results,
            "per_sample": per_sample,
            "skipped": skipped,
            "num_detection_samples": len(scored_rows),
        },
        internal_stages=INTERNAL_STAGES,
    )


def _metric_result(metric_name: str, score: float | None, auxiliary: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric_name": metric_name,
        "score": score,
        "details": {
            "higher_is_better": metric_name == "f1",
            "aggregation": "micro_average_duration",
            "precision": auxiliary["precision"],
            "recall": auxiliary["recall"],
            "false_alarm_sec": auxiliary["false_alarm_sec"],
            "miss_sec": auxiliary["miss_sec"],
            "speech_scored_sec": auxiliary["speech_scored_sec"],
            "nonspeech_scored_sec": auxiliary["nonspeech_scored_sec"],
            "dcf_weights": {"p_fa": 0.25, "p_miss": 0.75}
            if metric_name == "dcf_nist"
            else None,
        },
    }


def _duration(segments: list[Segment]) -> float:
    return sum(max(0.0, segment.end - segment.start) for segment in segments)


def _intersection_duration(left: list[Segment], right: list[Segment]) -> float:
    total = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_segment = left[left_index]
        right_segment = right[right_index]
        overlap_start = max(left_segment.start, right_segment.start)
        overlap_end = min(left_segment.end, right_segment.end)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
        if left_segment.end <= right_segment.end:
            left_index += 1
        else:
            right_index += 1
    return total


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None and recall is None:
        return None
    precision_value = precision if precision is not None else 0.0
    recall_value = recall if recall is not None else 0.0
    if precision_value + recall_value <= 0.0:
        return 0.0
    return 2.0 * precision_value * recall_value / (precision_value + recall_value)
