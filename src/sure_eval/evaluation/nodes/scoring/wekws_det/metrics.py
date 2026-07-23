"""Keyword spotting metrics with WekWS-style DET scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Iterable

from sure_eval.evaluation.base import MetricResult


@dataclass(frozen=True)
class KWSSample:
    """One keyword spotting prediction aligned with its reference label."""

    key: str
    expected_detected: bool
    expected_keyword: str | None = None
    duration: float | None = None
    detected: bool = False
    predicted_keyword: str | None = None
    score: float | None = None
    scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def detection_score(self) -> float:
        """Return the scalar confidence used by utterance-level metrics."""
        if self.score is not None:
            return float(self.score)
        if self.scores:
            return max(float(score) for score in self.scores)
        return 0.0


def normalize_keyword(value: str | None) -> str | None:
    """Normalize keywords for strict but whitespace-tolerant matching."""
    if value is None:
        return None
    normalized = "".join(str(value).upper().split())
    return normalized or None


def _default_thresholds(step: float = 0.01) -> list[float]:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    thresholds: list[float] = []
    value = 0.0
    while value <= 1.0 + 1e-9:
        thresholds.append(round(value, 6))
        value += step
    return thresholds


def _is_expected_keyword(sample: KWSSample) -> bool:
    if not sample.expected_detected:
        return False
    if sample.expected_keyword is None:
        return sample.detected
    return normalize_keyword(sample.predicted_keyword) == normalize_keyword(sample.expected_keyword)


def _predicts_positive_at_threshold(sample: KWSSample, threshold: float) -> bool:
    return sample.detected and sample.detection_score >= threshold


def _is_correct_at_threshold(sample: KWSSample, threshold: float) -> bool:
    predicts_positive = _predicts_positive_at_threshold(sample, threshold)
    if sample.expected_detected:
        if not predicts_positive:
            return False
        if sample.expected_keyword is None:
            return True
        return normalize_keyword(sample.predicted_keyword) == normalize_keyword(sample.expected_keyword)
    return not predicts_positive


def _error_type(sample: KWSSample, threshold: float) -> str | None:
    predicts_positive = _predicts_positive_at_threshold(sample, threshold)
    if sample.expected_detected:
        if not predicts_positive:
            return "false_reject"
        if sample.expected_keyword is not None and not _is_expected_keyword(sample):
            return "wrong_keyword"
        return None
    if predicts_positive:
        return "false_alarm"
    return None


def _duration_hours(samples: Iterable[KWSSample]) -> float:
    seconds = sum(float(sample.duration or 0.0) for sample in samples)
    return seconds / 3600.0


def compute_det_curve(
    samples: list[KWSSample],
    *,
    thresholds: list[float] | None = None,
    step: float = 0.01,
) -> list[dict[str, Any]]:
    """Compute a WekWS-style DET curve from aligned KWS samples."""
    thresholds = thresholds if thresholds is not None else _default_thresholds(step)
    positives = [sample for sample in samples if sample.expected_detected]
    negatives = [sample for sample in samples if not sample.expected_detected]
    filler_hours = _duration_hours(negatives)

    points: list[dict[str, Any]] = []
    for threshold in thresholds:
        false_rejects = sum(
            1
            for sample in positives
            if not _predicts_positive_at_threshold(sample, threshold)
            or (
                sample.expected_keyword is not None
                and normalize_keyword(sample.predicted_keyword) != normalize_keyword(sample.expected_keyword)
            )
        )
        true_detects = len(positives) - false_rejects
        false_alarms = sum(1 for sample in negatives if _predicts_positive_at_threshold(sample, threshold))
        false_reject_rate = false_rejects / len(positives) if positives else 0.0
        true_detect_rate = true_detects / len(positives) if positives else 0.0
        false_alarm_rate = false_alarms / len(negatives) if negatives else 0.0
        false_alarm_per_hour = false_alarms / filler_hours if filler_hours else 0.0
        points.append(
            {
                "threshold": float(threshold),
                "false_alarm_per_hour": false_alarm_per_hour,
                "false_reject_rate": false_reject_rate,
                "false_alarm_rate": false_alarm_rate,
                "true_detect_rate": true_detect_rate,
                "false_alarms": false_alarms,
                "false_rejects": false_rejects,
            }
        )
    return points


def compute_macro_recall_at_false_alarms(
    points: list[dict[str, Any]],
    *,
    false_alarm_budget: int = 0,
) -> dict[str, Any]:
    """Return max recall from DET points within a false-alarm count budget."""
    if false_alarm_budget < 0:
        raise ValueError("macro-recall false alarm budget must be non-negative")

    candidates = [point for point in points if int(point["false_alarms"]) <= false_alarm_budget]
    if not candidates:
        return {
            "score": 0.0,
            "false_alarm_budget": false_alarm_budget,
            "threshold": None,
            "achieved_false_alarms": None,
            "achieved_false_alarm_per_hour": None,
            "true_detect_rate": 0.0,
            "false_reject_rate": None,
            "feasible": False,
        }

    best = max(
        candidates,
        key=lambda point: (
            float(point["true_detect_rate"]),
            -int(point["false_alarms"]),
            -float(point["false_alarm_per_hour"]),
            -abs(float(point["threshold"]) - 0.5),
        ),
    )
    return {
        "score": float(best["true_detect_rate"]),
        "false_alarm_budget": false_alarm_budget,
        "threshold": best["threshold"],
        "achieved_false_alarms": best["false_alarms"],
        "achieved_false_alarm_per_hour": best["false_alarm_per_hour"],
        "true_detect_rate": best["true_detect_rate"],
        "false_reject_rate": best["false_reject_rate"],
        "feasible": True,
    }


class KWSMetric:
    """Aggregate keyword spotting metrics at a selected threshold."""

    def __init__(
        self,
        threshold: float = 0.5,
        thresholds: list[float] | None = None,
        threshold_step: float = 0.01,
    ) -> None:
        self.threshold = threshold
        self.thresholds = thresholds
        self.threshold_step = threshold_step

    def calculate(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        """Calculate KWS accuracy for one string prediction/reference pair."""
        pred_detected = str(prediction).strip().lower() in {"detect", "detected", "true", "1", "yes"}
        ref_detected = str(reference).strip().lower() in {"detect", "detected", "true", "1", "yes"}
        sample = KWSSample(
            key="sample1",
            expected_detected=ref_detected,
            detected=pred_detected,
            score=1.0 if pred_detected else 0.0,
        )
        return self.calculate_samples([sample], **kwargs)

    def calculate_batch(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs,
    ) -> MetricResult:
        samples = []
        for index, (prediction, reference) in enumerate(zip(predictions, references), start=1):
            pred_detected = str(prediction).strip().lower() in {"detect", "detected", "true", "1", "yes"}
            ref_detected = str(reference).strip().lower() in {"detect", "detected", "true", "1", "yes"}
            samples.append(
                KWSSample(
                    key=f"sample{index}",
                    expected_detected=ref_detected,
                    detected=pred_detected,
                    score=1.0 if pred_detected else 0.0,
                )
            )
        return self.calculate_samples(samples, **kwargs)

    def calculate_samples(self, samples: list[KWSSample], **kwargs) -> MetricResult:
        threshold = float(kwargs.get("threshold", self.threshold))
        thresholds = kwargs.get("thresholds", self.thresholds)
        threshold_step = float(kwargs.get("threshold_step", self.threshold_step))
        macro_recall_false_alarms = int(kwargs.get("macro_recall_false_alarms", 0))
        rows = build_rows(samples, threshold=threshold)
        total = len(rows)
        correct = sum(1 for row in rows if row["correct"])
        positives = [row for row in rows if row["expected_detected"]]
        negatives = [row for row in rows if not row["expected_detected"]]
        true_positives = sum(1 for row in positives if row["correct"])
        false_rejects = sum(1 for row in positives if not row["correct"])
        false_alarms = sum(1 for row in negatives if not row["correct"])
        precision_denominator = true_positives + false_alarms
        recall_denominator = len(positives)
        precision = true_positives / precision_denominator if precision_denominator else 0.0
        recall = true_positives / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        negative_hours = _duration_hours(sample for sample in samples if not sample.expected_detected)
        det_points = compute_det_curve(samples, thresholds=thresholds, step=threshold_step)
        macro_recall = compute_macro_recall_at_false_alarms(
            det_points,
            false_alarm_budget=macro_recall_false_alarms,
        )

        return MetricResult(
            metric_name="kws_accuracy",
            score=correct / total if total else 0.0,
            details={
                "threshold": threshold,
                "total": total,
                "correct": correct,
                "positive_samples": len(positives),
                "negative_samples": len(negatives),
                "true_positives": true_positives,
                "false_rejects": false_rejects,
                "false_alarms": false_alarms,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_reject_rate": false_rejects / len(positives) if positives else 0.0,
                "false_alarm_rate": false_alarms / len(negatives) if negatives else 0.0,
                "false_alarm_per_hour": false_alarms / negative_hours if negative_hours else 0.0,
                "det_curve": det_points,
                "macro_recall": macro_recall["score"],
                "macro_recall_operating_point": macro_recall,
            },
        )


def build_rows(samples: list[KWSSample], *, threshold: float) -> list[dict[str, Any]]:
    """Build per-sample rows at the selected threshold."""
    rows: list[dict[str, Any]] = []
    for sample in samples:
        correct = _is_correct_at_threshold(sample, threshold)
        rows.append(
            {
                "key": sample.key,
                "expected_detected": sample.expected_detected,
                "expected_keyword": sample.expected_keyword,
                "detected": sample.detected,
                "predicted_keyword": sample.predicted_keyword,
                "score": sample.score,
                "max_score": sample.detection_score,
                "duration": sample.duration,
                "correct": correct,
                "error_type": None if correct else _error_type(sample, threshold),
                "metadata": sample.metadata,
            }
        )
    return rows


def summarize_det_curve(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the operating point with the smallest FRR+FAR-rate."""
    if not points:
        return {}
    best = min(
        points,
        key=lambda point: (
            float(point["false_reject_rate"]) + float(point["false_alarm_rate"]),
            float(point["false_alarm_per_hour"]),
            abs(float(point["threshold"]) - 0.5),
        ),
    )
    return {
        "best_threshold": best["threshold"],
        "best_false_alarm_per_hour": best["false_alarm_per_hour"],
        "best_false_reject_rate": best["false_reject_rate"],
        "best_false_alarm_rate": best["false_alarm_rate"],
    }


def mean_score(samples: list[KWSSample]) -> float | None:
    scores = [sample.detection_score for sample in samples if sample.score is not None or sample.scores]
    return fmean(scores) if scores else None


__all__ = [
    "KWSMetric",
    "KWSSample",
    "build_rows",
    "compute_det_curve",
    "compute_macro_recall_at_false_alarms",
    "mean_score",
    "normalize_keyword",
    "summarize_det_curve",
]
