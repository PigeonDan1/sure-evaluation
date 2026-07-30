"""Normalize VAD intervals onto a strict reference seconds timebase."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult
from sure_eval.evaluation.nodes.validation.vad_contract import (
    FrameScore,
    Segment,
    VADValidatedBundle,
)

NODE_ID = "normalization/vad_timebase"
NODE_VERSION = "v1"
INTERNAL_STAGES = (
    "strict_profile_selection",
    "duration_clip",
    "invalid_interval_drop",
    "overlap_merge",
    "scored_region_summary",
)


@dataclass(frozen=True)
class VADNormalizedRow:
    """One VAD row after seconds-timebase normalization."""

    key: str
    duration: float
    reference_segments: list[Segment]
    prediction_segments: list[Segment] | None
    frame_scores: list[FrameScore] | None
    prediction_audio_duration: float | None
    available_metrics: set[str]
    skipped_metrics: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "duration": self.duration,
            "reference_segments": [segment.as_dict() for segment in self.reference_segments],
            "prediction_segments": (
                [segment.as_dict() for segment in self.prediction_segments]
                if self.prediction_segments is not None
                else None
            ),
            "frame_scores": (
                [frame_score.as_dict() for frame_score in self.frame_scores]
                if self.frame_scores is not None
                else None
            ),
            "prediction_audio_duration": self.prediction_audio_duration,
            "available_metrics": sorted(self.available_metrics),
            "skipped_metrics": dict(self.skipped_metrics),
        }


@dataclass(frozen=True)
class VADNormalizedBundle:
    """Normalized VAD rows and timebase settings."""

    rows: list[VADNormalizedRow]
    input_summary: dict[str, Any]
    frame_shift_sec: float = 0.01
    profile: str = "strict"
    collar_sec: float = 0.0
    boundary_exclusion_sec: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.as_dict() for row in self.rows],
            "input_summary": dict(self.input_summary),
            "frame_shift_sec": self.frame_shift_sec,
            "profile": self.profile,
            "collar_sec": self.collar_sec,
            "boundary_exclusion_sec": self.boundary_exclusion_sec,
        }


def normalize_vad_timebase(
    bundle: VADValidatedBundle,
    *,
    frame_shift_sec: float = 0.01,
    profile: str = "strict",
    collar_sec: float = 0.0,
    boundary_exclusion_sec: float = 0.0,
) -> tuple[VADNormalizedBundle, PipelineNodeResult]:
    """Clip, sort, and merge VAD intervals on the reference duration timebase."""

    _validate_config(
        frame_shift_sec=frame_shift_sec,
        profile=profile,
        collar_sec=collar_sec,
        boundary_exclusion_sec=boundary_exclusion_sec,
    )
    rows: list[VADNormalizedRow] = []
    dropped_reference_segments = 0
    dropped_prediction_segments = 0
    dropped_frame_scores = 0

    for row in bundle.rows:
        reference_segments, dropped_ref = _normalize_segments(
            row.reference_segments,
            duration=row.duration,
        )
        prediction_segments = None
        dropped_pred = 0
        if row.prediction_segments is not None:
            prediction_segments, dropped_pred = _normalize_segments(
                row.prediction_segments,
                duration=row.duration,
            )
        frame_scores = None
        dropped_scores = 0
        if row.frame_scores is not None:
            frame_scores, dropped_scores = _normalize_frame_scores(
                row.frame_scores,
                duration=row.duration,
            )
        dropped_reference_segments += dropped_ref
        dropped_prediction_segments += dropped_pred
        dropped_frame_scores += dropped_scores
        rows.append(
            VADNormalizedRow(
                key=row.key,
                duration=row.duration,
                reference_segments=reference_segments,
                prediction_segments=prediction_segments,
                frame_scores=frame_scores,
                prediction_audio_duration=row.prediction_audio_duration,
                available_metrics=set(row.available_metrics),
                skipped_metrics=dict(row.skipped_metrics),
            )
        )

    speech_scored_sec = sum(_duration(row.reference_segments) for row in rows)
    total_scored_sec = sum(row.duration for row in rows)
    input_summary = {
        **bundle.input_summary,
        "profile": profile,
        "frame_shift_sec": frame_shift_sec,
        "collar_sec": collar_sec,
        "boundary_exclusion_sec": boundary_exclusion_sec,
        "total_scored_sec": total_scored_sec,
        "speech_scored_sec": speech_scored_sec,
        "nonspeech_scored_sec": max(0.0, total_scored_sec - speech_scored_sec),
        "dropped_reference_segments": dropped_reference_segments,
        "dropped_prediction_segments": dropped_prediction_segments,
        "dropped_frame_scores": dropped_frame_scores,
    }
    normalized = VADNormalizedBundle(
        rows=rows,
        input_summary=input_summary,
        frame_shift_sec=frame_shift_sec,
        profile=profile,
        collar_sec=collar_sec,
        boundary_exclusion_sec=boundary_exclusion_sec,
    )
    result = PipelineNodeResult(
        stage="normalization",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "profile": profile,
            "frame_shift_sec": frame_shift_sec,
            "collar_sec": collar_sec,
            "boundary_exclusion_sec": boundary_exclusion_sec,
            "input_summary": input_summary,
            "rows": [
                {
                    "key": row.key,
                    "duration": row.duration,
                    "reference_segments": [segment.as_dict() for segment in row.reference_segments],
                    "prediction_segments": (
                        [segment.as_dict() for segment in row.prediction_segments]
                        if row.prediction_segments is not None
                        else None
                    ),
                    "num_frame_scores": len(row.frame_scores or ()),
                }
                for row in rows
            ],
        },
        internal_stages=INTERNAL_STAGES,
    )
    return normalized, result


def _validate_config(
    *,
    frame_shift_sec: float,
    profile: str,
    collar_sec: float,
    boundary_exclusion_sec: float,
) -> None:
    if not isinstance(frame_shift_sec, (int, float)) or isinstance(frame_shift_sec, bool):
        raise TypeError("frame_shift_sec must be a finite positive number")
    if not math.isfinite(float(frame_shift_sec)) or float(frame_shift_sec) <= 0.0:
        raise ValueError("frame_shift_sec must be a finite positive number")
    if profile != "strict":
        raise ValueError("VAD only supports profile='strict' in this task version")
    if float(collar_sec) != 0.0:
        raise ValueError("VAD strict profile does not support non-zero collar_sec")
    if float(boundary_exclusion_sec) != 0.0:
        raise ValueError(
            "VAD strict profile does not support non-zero boundary_exclusion_sec"
        )


def _normalize_segments(
    segments: list[Segment],
    *,
    duration: float,
) -> tuple[list[Segment], int]:
    clipped: list[Segment] = []
    dropped = 0
    for segment in segments:
        start = _clip(segment.start, duration=duration)
        end = _clip(segment.end, duration=duration)
        if end <= start:
            dropped += 1
            continue
        clipped.append(Segment(start=start, end=end))
    clipped.sort(key=lambda item: (item.start, item.end))
    return _merge_segments(clipped), dropped


def _normalize_frame_scores(
    frame_scores: list[FrameScore],
    *,
    duration: float,
) -> tuple[list[FrameScore], int]:
    clipped: list[FrameScore] = []
    dropped = 0
    for frame_score in frame_scores:
        start = _clip(frame_score.start, duration=duration)
        end = _clip(frame_score.end, duration=duration)
        if end <= start:
            dropped += 1
            continue
        clipped.append(FrameScore(start=start, end=end, score=frame_score.score))
    clipped.sort(key=lambda item: (item.start, item.end, item.score))
    return clipped, dropped


def _merge_segments(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    merged = [segments[0]]
    for segment in segments[1:]:
        current = merged[-1]
        if segment.start <= current.end:
            merged[-1] = Segment(start=current.start, end=max(current.end, segment.end))
        else:
            merged.append(segment)
    return merged


def _clip(value: float, *, duration: float) -> float:
    return min(max(float(value), 0.0), float(duration))


def _duration(segments: list[Segment]) -> float:
    return sum(max(0.0, segment.end - segment.start) for segment in segments)
