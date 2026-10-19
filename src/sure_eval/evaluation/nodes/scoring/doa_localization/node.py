"""Protocol-explicit 2D/3D DOA scoring with pure-Python assignment."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.doa_jsonl import DOABundle, DOAFrame, DOASource

NODE_ID = "scoring/doa_localization"
NODE_VERSION = "v1"
SUPPORTED_METRICS = (
    "mae",
    "rmse",
    "acc_10",
    "acc_15",
    "acc_30",
    "localization_recall",
    "localization_precision",
    "localization_f1",
)
FIXED_METRIC_THRESHOLDS = {"acc_10": 10.0, "acc_15": 15.0, "acc_30": 30.0}
REPORT_THRESHOLDS = (10.0, 15.0, 30.0)
INTERNAL_STAGES = (
    "active_source_selection",
    "cost_matrix_angular_distance",
    "one_to_one_assignment",
    "raw_error_accumulation",
    "angular_gating",
    "miss_false_accounting",
    "recording_aggregation",
    "dataset_aggregation",
)


def score_doa_localization(
    bundle: DOABundle,
    *,
    metric: str = "mae",
    threshold_deg: float = 10.0,
    aggregation: str = "micro",
) -> PipelineNodeResult:
    """Score raw localization and thresholded detection/localization metrics."""

    metric = metric.lower().replace("-", "_")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported DOA metric: {metric}")
    if not math.isfinite(threshold_deg) or threshold_deg <= 0:
        raise ValueError("threshold_deg must be finite and positive")
    if aggregation not in {"macro_recording", "micro"}:
        raise ValueError("aggregation must be 'macro_recording' or 'micro'")
    effective_threshold = _metric_threshold(metric, threshold_deg)

    ref_by_recording: dict[str, list[DOAFrame]] = defaultdict(list)
    for frame in bundle.reference:
        ref_by_recording[frame.recording_id].append(frame)
    pred_by_key = {(frame.recording_id, frame.frame_index): frame for frame in bundle.prediction}
    per_recording: dict[str, dict[str, Any]] = {}
    all_errors: list[float] = []
    total_gt = total_pred = 0

    for recording_id, frames in ref_by_recording.items():
        raw_errors: list[float] = []
        gt_count = pred_count = tp_count = 0
        frame_rows: list[dict[str, Any]] = []
        for reference_frame in frames:
            prediction_frame = pred_by_key[(recording_id, reference_frame.frame_index)]
            active_refs = [source for source in reference_frame.sources if source.activity]
            active_preds = _active_predictions(
                prediction_frame,
                policy=bundle.prediction_activity_policy,
            )
            pairs = _minimum_cost_assignment(active_refs, active_preds, dimension=bundle.dimension)
            errors = [
                _angular_distance(
                    active_refs[ref_index], active_preds[pred_index], bundle.dimension
                )
                for ref_index, pred_index in pairs
            ]
            raw_errors.extend(errors)
            gated = [error < effective_threshold for error in errors]
            tp = sum(gated)
            gt_count += len(active_refs)
            pred_count += len(active_preds)
            tp_count += tp
            frame_rows.append(
                {
                    "frame_index": reference_frame.frame_index,
                    "timestamp": reference_frame.timestamp,
                    "num_reference_active": len(active_refs),
                    "num_prediction_active": len(active_preds),
                    "num_assigned": len(pairs),
                    "num_true_positive": tp,
                    "num_missed_reference": len(active_refs) - tp,
                    "num_false_prediction": len(active_preds) - tp,
                    "true_positive_10": sum(error < 10.0 for error in errors),
                    "true_positive_15": sum(error < 15.0 for error in errors),
                    "true_positive_30": sum(error < 30.0 for error in errors),
                    "raw_errors_deg": errors,
                    "gated_threshold_deg": effective_threshold,
                }
            )
        all_errors.extend(raw_errors)
        per_recording[recording_id] = _summary(
            recording_id,
            raw_errors,
            gt_count,
            pred_count,
            tp_count,
            frame_rows,
            selected_threshold=effective_threshold,
        )
        total_gt += gt_count
        total_pred += pred_count

    total_tp = sum(error < effective_threshold for error in all_errors)
    micro = _summary(
        "__dataset__",
        all_errors,
        total_gt,
        total_pred,
        total_tp,
        [],
        selected_threshold=effective_threshold,
    )
    micro_results = {
        name: _score_for_metric(micro, name, threshold_deg) for name in SUPPORTED_METRICS
    }
    results = {
        name: _aggregate_metric(
            per_recording,
            micro,
            metric=name,
            configured_threshold=threshold_deg,
            aggregation=aggregation,
        )
        for name in SUPPORTED_METRICS
    }
    dataset_score = results[metric]
    aggregation_summary = {
        name: _aggregation_counts(per_recording, name, threshold_deg) for name in SUPPORTED_METRICS
    }
    details = {
        "backend": "doa_angular_localization",
        "dimension": bundle.dimension,
        "angle_unit": "degree",
        "coordinate_system": bundle.input_summary.get("coordinate_system", []),
        "alignment": bundle.alignment,
        "threshold_deg": effective_threshold,
        "configured_threshold_deg": threshold_deg,
        "fixed_metric_thresholds_deg": dict(FIXED_METRIC_THRESHOLDS),
        "gating_rule": "assigned pair is TP iff angular_error_deg < threshold_deg",
        "raw_error_policy": "all assigned pairs, including pairs that fail gating",
        "assignment_policy": "minimum total angular cost over one-to-one rectangular assignment before gating",
        "assignment_tp_policy": "assignment minimizes raw angular cost and does not maximize the number of gated true positives",
        "assignment_complexity": "O(n^3) time and O(n^2) memory in max active source count",
        "prediction_activity_policy": bundle.prediction_activity_policy,
        "activity_rule": "reference activity uses source.activity; prediction uses presence or explicit activity policy",
        "aggregation": aggregation,
        "macro_recording_policy": "arithmetic mean over recordings where the requested metric is defined",
        "undefined_score_policy": "None when a metric denominator is zero or MAE/RMSE has no assigned pair",
        "metric_denominators": {
            "mae_rmse": "all assigned pairs",
            "acc_and_recall": "active reference sources",
            "precision": "active prediction sources",
            "f1": "harmonic mean of precision and recall",
        },
        "score": dataset_score,
        "results": results,
        "micro_results": micro_results,
        "aggregation_summary": aggregation_summary,
        "micro": micro,
        "per_recording": per_recording,
        "input_summary": bundle.input_summary,
    }
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details=details,
        internal_stages=INTERNAL_STAGES,
    )


def _active_predictions(prediction: DOAFrame, *, policy: str) -> list[DOASource]:
    if policy == "presence":
        return list(prediction.sources)
    if policy == "explicit":
        return [source for source in prediction.sources if source.activity]
    raise ValueError(f"Unsupported prediction activity policy: {policy}")


def _angular_distance(reference: DOASource, prediction: DOASource, dimension: str) -> float:
    azimuth_delta = math.radians(
        abs(((prediction.azimuth - reference.azimuth + 180.0) % 360.0) - 180.0)
    )
    if dimension == "2d":
        return math.degrees(azimuth_delta)
    if reference.elevation is None or prediction.elevation is None:
        raise ValueError("3d scoring requires elevation on both reference and prediction sources")
    elevation_ref = math.radians(reference.elevation)
    elevation_pred = math.radians(prediction.elevation)
    cosine = math.cos(elevation_ref) * math.cos(elevation_pred) * math.cos(
        azimuth_delta
    ) + math.sin(elevation_ref) * math.sin(elevation_pred)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _minimum_cost_assignment(
    references: list[DOASource], predictions: list[DOASource], *, dimension: str
) -> list[tuple[int, int]]:
    if not references or not predictions:
        return []
    if len(references) <= len(predictions):
        costs = [
            [_angular_distance(reference, prediction, dimension) for prediction in predictions]
            for reference in references
        ]
        assignment = _hungarian_rectangular(costs)
        return [
            (reference_index, prediction_index)
            for reference_index, prediction_index in enumerate(assignment)
        ]

    reverse_costs = [
        [_angular_distance(prediction, reference, dimension) for reference in references]
        for prediction in predictions
    ]
    reverse = _hungarian_rectangular(reverse_costs)
    return [
        (reference_index, prediction_index)
        for prediction_index, reference_index in enumerate(reverse)
    ]


def _hungarian_rectangular(costs: list[list[float]]) -> list[int]:
    """Return the minimum-cost column for every row of an n-by-m matrix.

    The caller guarantees ``n <= m``. Row and column iteration is ascending,
    which makes equal-cost tie breaking deterministic.
    """

    rows = len(costs)
    columns = len(costs[0]) if rows else 0
    if not rows:
        return []
    if columns < rows:
        raise ValueError("Hungarian rectangular solver requires rows <= columns")
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    matching = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        matching[0] = row
        column0 = 0
        minimum = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = matching[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                reduced = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[matching[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matching[column0] == 0:
                break
        while True:
            previous = way[column0]
            matching[column0] = matching[previous]
            column0 = previous
            if column0 == 0:
                break
    assignment = [0] * rows
    for column in range(1, columns + 1):
        if matching[column]:
            assignment[matching[column] - 1] = column - 1
    return assignment


def _summary(
    recording_id: str,
    errors: list[float],
    gt_count: int,
    pred_count: int,
    tp_count: int,
    frame_rows: list[dict[str, Any]],
    *,
    selected_threshold: float,
) -> dict[str, Any]:
    mae = sum(errors) / len(errors) if errors else None
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None
    threshold_counts = {
        _threshold_key(threshold): _counts_at_threshold(errors, gt_count, pred_count, threshold)
        for threshold in REPORT_THRESHOLDS
    }
    selected_counts = _counts_at_threshold(errors, gt_count, pred_count, selected_threshold)
    return {
        "recording_id": recording_id,
        "mae": mae,
        "rmse": rmse,
        "assigned_pairs": len(errors),
        "active_reference_sources": gt_count,
        "active_prediction_sources": pred_count,
        "true_positive_10": threshold_counts["10"]["true_positive"],
        "true_positive_15": threshold_counts["15"]["true_positive"],
        "true_positive_30": threshold_counts["30"]["true_positive"],
        "true_positive": tp_count,
        "missed_reference_sources": selected_counts["missed_reference_sources"],
        "false_prediction_sources": selected_counts["false_prediction_sources"],
        "selected_threshold_deg": selected_threshold,
        "counts_by_threshold_deg": threshold_counts,
        "raw_errors_deg": errors,
        "frames": frame_rows,
    }


def _metric_threshold(metric: str, configured: float) -> float:
    return FIXED_METRIC_THRESHOLDS.get(metric, configured)


def _score_for_metric(
    summary: dict[str, Any], metric: str, configured_threshold: float
) -> float | None:
    if metric == "mae":
        return summary["mae"]
    if metric == "rmse":
        return summary["rmse"]
    threshold = _metric_threshold(metric, configured_threshold)
    errors = summary["raw_errors_deg"]
    tp = sum(error < threshold for error in errors)
    gt = summary["active_reference_sources"]
    pred = summary["active_prediction_sources"]
    if metric.startswith("acc_") or metric == "localization_recall":
        return _safe_div(tp, gt)
    if metric == "localization_precision":
        return _safe_div(tp, pred)
    if metric == "localization_f1":
        precision = _safe_div(tp, pred)
        recall = _safe_div(tp, gt)
        if precision is None or recall is None:
            return None
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    return None


def _safe_div(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _aggregate_metric(
    per_recording: dict[str, dict[str, Any]],
    micro: dict[str, Any],
    *,
    metric: str,
    configured_threshold: float,
    aggregation: str,
) -> float | None:
    if aggregation == "micro":
        return _score_for_metric(micro, metric, configured_threshold)
    scores = [
        _score_for_metric(summary, metric, configured_threshold)
        for summary in per_recording.values()
    ]
    defined_scores = [score for score in scores if score is not None]
    return sum(defined_scores) / len(defined_scores) if defined_scores else None


def _aggregation_counts(
    per_recording: dict[str, dict[str, Any]], metric: str, configured_threshold: float
) -> dict[str, int]:
    scores = [
        _score_for_metric(summary, metric, configured_threshold)
        for summary in per_recording.values()
    ]
    defined = sum(score is not None for score in scores)
    return {
        "defined_recordings": defined,
        "excluded_undefined_recordings": len(scores) - defined,
    }


def _counts_at_threshold(
    errors: list[float], gt_count: int, pred_count: int, threshold: float
) -> dict[str, int]:
    true_positive = sum(error < threshold for error in errors)
    return {
        "true_positive": true_positive,
        "missed_reference_sources": gt_count - true_positive,
        "false_prediction_sources": pred_count - true_positive,
    }


def _threshold_key(threshold: float) -> str:
    return str(int(threshold)) if threshold.is_integer() else str(threshold)
