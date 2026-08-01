"""ROC AUC scoring for VAD frame_scores."""

from __future__ import annotations

from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.vad_timebase import VADNormalizedBundle
from sure_eval.evaluation.nodes.validation.vad_contract import FrameScore, Segment

NODE_ID = "scoring/vad_auc_roc"
NODE_VERSION = "v1"
INTERNAL_STAGES = (
    "frame_center_sampling",
    "reference_label_lookup",
    "score_coverage_filter",
    "rank_auc",
)


def score_vad_auc_roc(bundle: VADNormalizedBundle) -> PipelineNodeResult:
    """Compute ROC AUC from explicit frame_scores without hard-label fallback."""

    labels: list[int] = []
    scores: list[float] = []
    per_sample: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for row in bundle.rows:
        if row.frame_scores is None:
            skipped.append(
                {
                    "key": row.key,
                    "metrics": ["auc_roc"],
                    "reason": row.skipped_metrics.get(
                        "auc_roc",
                        "missing prediction field: frame_scores",
                    ),
                }
            )
            continue
        row_labels, row_scores = _covered_frame_labels_and_scores(
            duration=row.duration,
            reference_segments=row.reference_segments,
            frame_scores=row.frame_scores,
            frame_shift_sec=bundle.frame_shift_sec,
        )
        labels.extend(row_labels)
        scores.extend(row_scores)
        positive_frames = sum(row_labels)
        per_sample.append(
            {
                "key": row.key,
                "duration": row.duration,
                "num_scored_frames": len(row_labels),
                "positive_frames": positive_frames,
                "negative_frames": len(row_labels) - positive_frames,
            }
        )
        if not row_labels:
            skipped.append(
                {
                    "key": row.key,
                    "metrics": ["auc_roc"],
                    "reason": "no frame center covered by frame_scores",
                }
            )

    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    auc_roc: float | None
    corpus_skip_reason = ""
    if not labels:
        auc_roc = None
        corpus_skip_reason = "no covered frame_scores"
    elif positive_count == 0 or negative_count == 0:
        auc_roc = None
        corpus_skip_reason = "single_class_labels"
    else:
        auc_roc = _rank_auc(labels, scores)

    if corpus_skip_reason:
        skipped.append(
            {
                "key": None,
                "metrics": ["auc_roc"],
                "reason": corpus_skip_reason,
            }
        )

    details = {
        "backend": "vad_frame_score_rank_auc",
        "auc_roc": auc_roc,
        "num_auc_samples": len(labels),
        "positive_frames": positive_count,
        "negative_frames": negative_count,
        "frame_shift_sec": bundle.frame_shift_sec,
        "per_sample": per_sample,
        "skipped": skipped,
        "results": {
            "auc_roc": {
                "metric_name": "auc_roc",
                "score": auc_roc,
                "details": {
                    "higher_is_better": True,
                    "aggregation": "pooled_frame_rank_auc",
                    "num_auc_samples": len(labels),
                    "positive_frames": positive_count,
                    "negative_frames": negative_count,
                    "frame_shift_sec": bundle.frame_shift_sec,
                    "hard_label_fallback": False,
                },
            }
        },
    }
    return PipelineNodeResult(
        stage="scoring",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details=details,
        internal_stages=INTERNAL_STAGES,
    )


def _covered_frame_labels_and_scores(
    *,
    duration: float,
    reference_segments: list[Segment],
    frame_scores: list[FrameScore],
    frame_shift_sec: float,
) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    frame_score_index = 0
    frame_index = 0
    while True:
        center = frame_index * frame_shift_sec + frame_shift_sec / 2.0
        if center >= duration:
            break
        while (
            frame_score_index < len(frame_scores)
            and frame_scores[frame_score_index].end <= center
        ):
            frame_score_index += 1
        if frame_score_index < len(frame_scores):
            frame_score = frame_scores[frame_score_index]
            if frame_score.start <= center < frame_score.end:
                labels.append(_label_for_center(center, reference_segments))
                scores.append(frame_score.score)
        frame_index += 1
    return labels, scores


def _label_for_center(center: float, reference_segments: list[Segment]) -> int:
    for segment in reference_segments:
        if segment.start <= center < segment.end:
            return 1
        if segment.start > center:
            return 0
    return 0


def _rank_auc(labels: list[int], scores: list[float]) -> float:
    indexed_scores = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(indexed_scores):
        end = index + 1
        while end < len(indexed_scores) and indexed_scores[end][1] == indexed_scores[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for rank_index in range(index, end):
            original_index = indexed_scores[rank_index][0]
            ranks[original_index] = average_rank
        index = end

    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
