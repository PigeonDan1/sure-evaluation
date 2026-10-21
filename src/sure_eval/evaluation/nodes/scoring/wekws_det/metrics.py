"""KWS recall and latency at a shared false-alarm-per-hour operating point."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from sure_eval.evaluation.base import MetricResult

DEFAULT_TARGET_FA_PER_HOUR = 0.1


@dataclass(frozen=True)
class KWSEvent:
    keyword: str
    end_sec: float | None = None
    start_sec: float | None = None


@dataclass(frozen=True)
class KWSTrigger:
    keyword: str | None
    score: float
    trigger_time_sec: float | None = None


@dataclass(frozen=True)
class KWSSample:
    """A clip or recording with uncensored scores or pre-threshold trigger candidates.

    Frame-score input represents a clip with at most one trigger. Event lists
    support multiple references and triggers; candidates must already represent
    distinct detector events, with the system's own debouncing applied.
    """

    key: str
    expected_detected: bool
    expected_keyword: str | None = None
    duration: float | None = None
    detected: bool = False
    predicted_keyword: str | None = None
    score: float | None = None
    scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    keyword_end_sec: float | None = None
    keyword_start_sec: float | None = None
    trigger_time_sec: float | None = None
    frame_shift_sec: float | None = None
    frame_offset_sec: float = 0.0
    events: list[KWSEvent] | None = None
    detections: list[KWSTrigger] | None = None
    keyword_scores: dict[str, float] = field(default_factory=dict)

    @property
    def detection_score(self) -> float:
        return max(self.scores) if self.scores else float(self.score or 0.0)


def normalize_keyword(value: str | None) -> str | None:
    if value is None:
        return None
    return "".join(str(value).upper().split()) or None


def reference_events(sample: KWSSample) -> list[KWSEvent]:
    if sample.events is not None:
        return sample.events
    if sample.expected_detected:
        return [
            KWSEvent(
                sample.expected_keyword or "", sample.keyword_end_sec, sample.keyword_start_sec
            )
        ]
    return []


def _finite(value: float, name: str, *, nonnegative: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"{name} must be finite" + (" and non-negative" if nonnegative else ""))
    return number


def _raw_scores(sample: KWSSample) -> list[float]:
    if sample.detections is not None:
        return [trigger.score for trigger in sample.detections]
    return sample.scores or ([sample.score] if sample.score is not None else [])


def _validate(samples: list[KWSSample]) -> tuple[list[str], float]:
    if not samples:
        raise ValueError("KWS evaluation requires samples")
    keys: set[str] = set()
    keywords: set[str] = set()
    negative_seconds = 0.0
    for sample in samples:
        if sample.key in keys:
            raise ValueError(f"Duplicate KWS key: {sample.key}")
        keys.add(sample.key)
        events = reference_events(sample)
        if sample.expected_detected != bool(events):
            raise ValueError(f"Inconsistent expected_detected/events for {sample.key}")
        if (
            sample.duration is not None
            and _finite(sample.duration, "duration", nonnegative=True) <= 0
        ):
            raise ValueError("duration must be positive")
        if not events:
            if sample.duration is None:
                raise ValueError(f"Negative sample {sample.key} requires duration in seconds")
            negative_seconds += float(sample.duration)
        for event in events:
            keyword = normalize_keyword(event.keyword)
            if not keyword:
                raise ValueError(f"Positive sample {sample.key} requires a keyword")
            keywords.add(keyword)
            for value in (event.start_sec, event.end_sec):
                if value is not None:
                    _finite(value, "reference timestamp", nonnegative=True)
                    if sample.duration is not None and value > sample.duration:
                        raise ValueError("Reference timestamp exceeds recording duration")
            if event.start_sec is not None and event.end_sec is None:
                raise ValueError("start_sec requires end_sec")
            if event.start_sec is not None and event.start_sec > event.end_sec:
                raise ValueError("start_sec must not exceed end_sec")
        if len(events) > 1 and (
            sample.detections is None or any(event.end_sec is None for event in events)
        ):
            raise ValueError(
                "Multiple reference events require end_sec and an explicit detections list"
            )
        if sample.detections is None and not _raw_scores(sample):
            raise ValueError(
                f"Raw score required for {sample.key}; use detections=[] for no candidates"
            )
        if sample.detections is not None and (sample.score is not None or sample.scores):
            raise ValueError("Provide detections or clip/frame scores, not both")
        for value in [*_raw_scores(sample), *sample.keyword_scores.values()]:
            _finite(value, "score")
        score_keywords = [normalize_keyword(key) for key in sample.keyword_scores]
        if None in score_keywords or len(score_keywords) != len(set(score_keywords)):
            raise ValueError("keyword_scores requires distinct nonempty normalized keywords")
        times = (
            [trigger.trigger_time_sec for trigger in sample.detections]
            if sample.detections is not None
            else [sample.trigger_time_sec]
        )
        for value in times:
            if value is not None:
                _finite(value, "trigger_time_sec", nonnegative=True)
                if sample.duration is not None and value > sample.duration:
                    raise ValueError("Trigger timestamp exceeds recording duration")
            elif len(events) > 1:
                raise ValueError("Multiple reference events require trigger timestamps")
        if (
            sample.frame_shift_sec is not None
            and _finite(sample.frame_shift_sec, "frame_shift_sec", nonnegative=True) <= 0
        ):
            raise ValueError("frame_shift_sec must be positive")
        _finite(sample.frame_offset_sec, "frame_offset_sec", nonnegative=True)
        if sample.scores and sample.frame_shift_sec is not None and sample.duration is not None:
            if (
                sample.frame_offset_sec + len(sample.scores) * sample.frame_shift_sec
                > sample.duration + 1e-9
            ):
                raise ValueError("Frame timestamps exceed recording duration")
    if not math.isfinite(negative_seconds) or negative_seconds <= 0:
        raise ValueError("FA/hour requires positive total negative audio duration")
    if not keywords:
        raise ValueError("Macro recall requires at least one positive keyword")
    return sorted(keywords), negative_seconds / 3600.0


def _thresholds(values: list[float], requested: list[float] | None = None) -> list[float]:
    upper = math.nextafter(max(values, default=0.0), math.inf)
    _finite(upper, "all-rejected threshold")
    thresholds = {
        _finite(value, "threshold") for value in (requested if requested is not None else values)
    }
    return sorted(thresholds | {upper})


def _triggers_at(sample: KWSSample, threshold: float) -> list[KWSTrigger]:
    if sample.detections is not None:
        triggers = [trigger for trigger in sample.detections if trigger.score >= threshold]
    elif sample.scores:
        # First qualifying frame, never the peak frame or a cached detected flag.
        index = next((i for i, score in enumerate(sample.scores) if score >= threshold), None)
        if index is None:
            return []
        time_sec = None
        if sample.frame_shift_sec is not None:
            time_sec = sample.frame_offset_sec + (index + 1) * sample.frame_shift_sec
        triggers = [KWSTrigger(sample.predicted_keyword, sample.scores[index], time_sec)]
    elif sample.score is not None and sample.score >= threshold:
        triggers = [KWSTrigger(sample.predicted_keyword, sample.score, sample.trigger_time_sec)]
    else:
        return []
    return sorted(
        triggers,
        key=lambda trigger: (
            trigger.trigger_time_sec if trigger.trigger_time_sec is not None else math.inf
        ),
    )


def _match(sample: KWSSample, threshold: float, tolerance: float) -> tuple[list[dict], int, int]:
    events = reference_events(sample)
    triggers = _triggers_at(sample, threshold)
    unmatched = set(range(len(events)))
    matches: dict[int, KWSTrigger] = {}
    for trigger in triggers:
        eligible = []
        for index in unmatched:
            event = events[index]
            if normalize_keyword(trigger.keyword) != normalize_keyword(event.keyword):
                continue
            if event.end_sec is not None and trigger.trigger_time_sec is not None:
                start = (
                    event.start_sec
                    if event.start_sec is not None
                    else max(0.0, event.end_sec - tolerance)
                )
                if not start <= trigger.trigger_time_sec <= event.end_sec + tolerance:
                    continue
            eligible.append(index)
        if eligible:
            # Earliest closing interval preserves one-to-one matching for repeated words.
            index = min(
                eligible,
                key=lambda i: (events[i].end_sec if events[i].end_sec is not None else math.inf, i),
            )
            unmatched.remove(index)
            matches[index] = trigger
    rows = []
    for index, event in enumerate(events):
        trigger = matches.get(index)
        latency = None
        if (
            trigger is not None
            and trigger.trigger_time_sec is not None
            and event.end_sec is not None
        ):
            latency = (trigger.trigger_time_sec - event.end_sec) * 1000.0
        rows.append(
            {
                "key": sample.key,
                "event_index": index,
                "keyword": normalize_keyword(event.keyword),
                "matched": trigger is not None,
                "keyword_end_sec": event.end_sec,
                "trigger_time_sec": trigger.trigger_time_sec if trigger else None,
                "latency_ms": latency,
            }
        )
    return rows, len(triggers) if not events else 0, len(triggers) - len(matches)


def _curve(
    samples: list[KWSSample],
    keywords: list[str],
    hours: float,
    tolerance: float,
    requested: list[float] | None,
) -> list[dict]:
    """Sweep tied scores together, updating only recordings that can change."""
    changes: dict[float, set[int]] = {}
    totals = Counter(
        normalize_keyword(event.keyword) for sample in samples for event in reference_events(sample)
    )
    for index, sample in enumerate(samples):
        values = _raw_scores(sample)
        if sample.detections is None and sample.scores:
            # A first-crossing decision changes only at a new prefix maximum.
            peak = -math.inf
            values = []
            for score in sample.scores:
                if score > peak:
                    values.append(score)
                    peak = score
        for score in values:
            changes.setdefault(score, set()).add(index)
    thresholds = _thresholds(list(changes), requested)
    pending = sorted(changes, reverse=True)
    cursor = 0
    hits: Counter = Counter()
    previous: dict[int, tuple[Counter, int, int]] = {}
    false_alarms = unmatched = 0
    points = []
    for threshold in reversed(thresholds):
        affected: set[int] = set()
        while cursor < len(pending) and pending[cursor] >= threshold:
            affected.update(changes[pending[cursor]])
            cursor += 1
        for index in affected:
            old_hits, old_fa, old_unmatched = previous.get(index, (Counter(), 0, 0))
            rows, fa, unpaired = _match(samples[index], threshold, tolerance)
            new_hits = Counter(row["keyword"] for row in rows if row["matched"])
            hits.subtract(old_hits)
            hits.update(new_hits)
            false_alarms += fa - old_fa
            unmatched += unpaired - old_unmatched
            previous[index] = new_hits, fa, unpaired
        per_keyword = {
            keyword: {
                "recall": hits[keyword] / totals[keyword],
                "true_positives": hits[keyword],
                "positive_events": totals[keyword],
            }
            for keyword in keywords
        }
        points.append(
            {
                "threshold": threshold,
                "false_alarms": false_alarms,
                "false_alarm_per_hour": false_alarms / hours,
                "macro_recall": fmean(item["recall"] for item in per_keyword.values()),
                "per_keyword": per_keyword,
                "unmatched_triggers": unmatched,
            }
        )
    return list(reversed(points))


def _latency(rows: list[dict]) -> dict:
    values = sorted(row["latency_ms"] for row in rows if row["latency_ms"] is not None)
    matched = sum(row["matched"] for row in rows)

    def percentile(fraction: float) -> float | None:
        if not values:
            return None
        position = (len(values) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "count": len(values),
        "matched_events": matched,
        "coverage": len(values) / matched if matched else None,
        "mean_ms": fmean(values) if values else None,
        "p50_ms": percentile(0.5),
        "p95_ms": percentile(0.95),
    }


def compute_roc(samples: list[KWSSample], keywords: list[str]) -> dict:
    """One-vs-rest clip ROC; event streams have no negative-trial definition."""
    if any(sample.events is not None or sample.detections is not None for sample in samples):
        return {
            "available": False,
            "reason": "ROC/AUC requires scored single-event clips, not event lists",
        }
    scores = []
    for sample in samples:
        table = {
            normalize_keyword(key): float(value) for key, value in sample.keyword_scores.items()
        }
        if not table and len(keywords) == 1:
            if normalize_keyword(sample.predicted_keyword) not in {None, keywords[0]}:
                return {
                    "available": False,
                    "reason": "ROC requires a score for the target keyword, not another predicted keyword",
                }
            table = {keywords[0]: sample.detection_score}
        if not all(keyword in table for keyword in keywords):
            return {
                "available": False,
                "reason": "Multiclass ROC requires keyword_scores for every keyword on every clip",
            }
        scores.append(table)
    curves = {}
    for keyword in keywords:
        labels = [
            normalize_keyword(sample.expected_keyword) == keyword
            if sample.expected_detected
            else False
            for sample in samples
        ]
        positives, negatives = sum(labels), len(labels) - sum(labels)
        groups: dict[float, list[int]] = {}
        for label, score in zip(labels, scores):
            counts = groups.setdefault(score[keyword], [0, 0])
            counts[0 if label else 1] += 1
        points = [{"threshold": math.nextafter(max(groups), math.inf), "fpr": 0.0, "tpr": 0.0}]
        tp = fp = 0
        for threshold in sorted(groups, reverse=True):
            positive_count, negative_count = groups[threshold]
            tp += positive_count
            fp += negative_count
            points.append({"threshold": threshold, "fpr": fp / negatives, "tpr": tp / positives})
        auc = sum(
            (right["fpr"] - left["fpr"]) * (left["tpr"] + right["tpr"]) / 2
            for left, right in zip(points, points[1:])
        )
        curves[keyword] = {"points": points, "auc": auc}
    return {
        "available": True,
        "per_keyword": curves,
        "macro_auc": fmean(curve["auc"] for curve in curves.values()),
        "x_axis": "false_positive_rate",
    }


class KWSMetric:
    def __init__(
        self,
        *,
        target_fa_per_hour: float = DEFAULT_TARGET_FA_PER_HOUR,
        thresholds: list[float] | None = None,
        include_roc: bool = False,
        match_tolerance_sec: float = 1.0,
    ) -> None:
        self.target_fa_per_hour = target_fa_per_hour
        self.thresholds = thresholds
        self.include_roc = include_roc
        self.match_tolerance_sec = match_tolerance_sec

    def calculate(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        raise ValueError(
            "KWS FA/hour evaluation requires scored samples and negative audio durations; use calculate_samples"
        )

    def calculate_batch(
        self, predictions: list[str], references: list[str], **kwargs
    ) -> MetricResult:
        raise ValueError(
            "KWS FA/hour evaluation requires scored samples and negative audio durations; use calculate_samples"
        )

    def calculate_samples(self, samples: list[KWSSample], **kwargs) -> MetricResult:
        target = _finite(
            kwargs.get("target_fa_per_hour", self.target_fa_per_hour),
            "target_fa_per_hour",
            nonnegative=True,
        )
        tolerance = _finite(
            kwargs.get("match_tolerance_sec", self.match_tolerance_sec),
            "match_tolerance_sec",
            nonnegative=True,
        )
        keywords, hours = _validate(samples)
        points = _curve(
            samples, keywords, hours, tolerance, kwargs.get("thresholds", self.thresholds)
        )
        feasible = [point for point in points if point["false_alarm_per_hour"] <= target]
        best = max(
            feasible,
            key=lambda point: (point["macro_recall"], -point["false_alarms"], point["threshold"]),
        )
        rows = [
            row for sample in samples for row in _match(sample, best["threshold"], tolerance)[0]
        ]
        latency = {
            "overall": _latency(rows),
            "per_keyword": {
                keyword: _latency([row for row in rows if row["keyword"] == keyword])
                for keyword in keywords
            },
            "reference": "keyword_end",
            "unit": "ms",
        }
        details = {
            "target_fa_per_hour": target,
            "negative_hours": hours,
            "threshold_scope": "global",
            "match_tolerance_sec": tolerance,
            "operating_point": best,
            "operating_curve": points,
            "rows": rows,
            "trigger_latency": latency,
        }
        if kwargs.get("include_roc", self.include_roc):
            details["roc"] = compute_roc(samples, keywords)
        return MetricResult(metric_name="macro_recall", score=best["macro_recall"], details=details)


KWSMacroRecallMetric = KWSMetric
