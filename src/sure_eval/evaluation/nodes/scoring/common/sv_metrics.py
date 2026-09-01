"""Shared speaker-verification DET metrics."""

from __future__ import annotations

import numpy as np


def sorted_det_points(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.uint8)
    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ValueError("scores and labels must be one-dimensional")
    if score_array.size != label_array.size:
        raise ValueError("scores and labels must have the same length")
    if score_array.size == 0:
        raise ValueError("at least one speaker-verification trial is required")
    if not np.isfinite(score_array).all():
        raise ValueError("speaker-verification scores contain NaN or infinity")
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("speaker-verification labels must contain only 0 and 1")

    target_total = int(np.count_nonzero(label_array == 1))
    nontarget_total = int(np.count_nonzero(label_array == 0))
    if target_total == 0 or nontarget_total == 0:
        raise ValueError("both target and nontarget trials are required")

    order = np.argsort(-score_array, kind="mergesort")
    sorted_scores = score_array[order]
    sorted_labels = label_array[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    distinct = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    true_positives = true_positives[distinct]
    false_positives = false_positives[distinct]
    thresholds = sorted_scores[distinct]

    false_alarm_rate = false_positives.astype(np.float64) / float(nontarget_total)
    miss_rate = (target_total - true_positives).astype(np.float64) / float(target_total)
    false_alarm_rate = np.concatenate(([0.0], false_alarm_rate))
    miss_rate = np.concatenate(([1.0], miss_rate))
    thresholds = np.concatenate(([np.inf], thresholds))
    return false_alarm_rate, miss_rate, thresholds


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    false_alarm_rate, miss_rate, thresholds = sorted_det_points(scores, labels)
    difference = false_alarm_rate - miss_rate
    exact = np.flatnonzero(np.isclose(difference, 0.0, atol=1e-12))
    if exact.size:
        index = int(exact[0])
        return float(false_alarm_rate[index] * 100.0), float(thresholds[index])

    crossings = np.flatnonzero(difference[:-1] * difference[1:] < 0.0)
    if crossings.size == 0:
        index = int(np.argmin(np.abs(difference)))
        eer = (false_alarm_rate[index] + miss_rate[index]) * 0.5
        return float(eer * 100.0), float(thresholds[index])

    index = int(crossings[0])
    left = float(difference[index])
    right = float(difference[index + 1])
    weight = left / (left - right)
    eer = false_alarm_rate[index] + weight * (false_alarm_rate[index + 1] - false_alarm_rate[index])
    threshold = _interpolate_threshold(thresholds[index], thresholds[index + 1], weight)
    return float(eer * 100.0), float(threshold)


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> tuple[float, float]:
    if not 0.0 < p_target < 1.0:
        raise ValueError("p_target must be between 0 and 1")
    false_alarm_rate, miss_rate, thresholds = sorted_det_points(scores, labels)
    dcf = c_miss * p_target * miss_rate + c_fa * (1.0 - p_target) * false_alarm_rate
    normalization = min(c_miss * p_target, c_fa * (1.0 - p_target))
    if normalization <= 0.0:
        raise ValueError("DCF normalization must be positive")
    normalized = dcf / normalization
    index = int(np.argmin(normalized))
    return float(normalized[index]), float(thresholds[index])


def _interpolate_threshold(left: float, right: float, weight: float) -> float:
    if np.isfinite(left) and np.isfinite(right):
        return float(left + weight * (right - left))
    if np.isfinite(right):
        return float(right)
    return float(left)
