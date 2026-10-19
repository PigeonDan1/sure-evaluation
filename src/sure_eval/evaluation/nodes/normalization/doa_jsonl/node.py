"""Normalize strict, frame-aligned canonical DOA JSONL inputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sure_eval.evaluation.core.types import PipelineNodeResult

NODE_ID = "normalization/doa_jsonl"
NODE_VERSION = "v1"
INTERNAL_STAGES = (
    "jsonl_parse",
    "row_contract",
    "source_contract",
    "coordinate_and_angle_unit_validation",
    "azimuth_wrap",
    "exact_frame_alignment",
    "activity_policy_validation",
)


@dataclass(frozen=True)
class DOASource:
    source_id: str
    azimuth: float
    elevation: float | None
    activity: bool
    confidence: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_id": self.source_id,
            "azimuth": self.azimuth,
            "activity": self.activity,
        }
        if self.elevation is not None:
            payload["elevation"] = self.elevation
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


@dataclass(frozen=True)
class DOAFrame:
    recording_id: str
    frame_index: int
    timestamp: float
    num_channels: int
    coordinate_system: str
    angle_unit: str
    sources: tuple[DOASource, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "num_channels": self.num_channels,
            "coordinate_system": self.coordinate_system,
            "angle_unit": self.angle_unit,
            "sources": [source.as_dict() for source in self.sources],
        }


@dataclass(frozen=True)
class DOABundle:
    reference: tuple[DOAFrame, ...]
    prediction: tuple[DOAFrame, ...]
    dimension: str
    alignment: str
    prediction_activity_policy: str
    input_summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "alignment": self.alignment,
            "prediction_activity_policy": self.prediction_activity_policy,
            "input_summary": dict(self.input_summary),
            "num_reference_frames": len(self.reference),
            "num_prediction_frames": len(self.prediction),
        }


def normalize_doa_jsonl(
    reference_jsonl: str | Path,
    sample_output: str | Path,
    *,
    dimension: str = "2d",
    timestamp_tolerance_sec: float = 1e-6,
    prediction_activity_policy: str = "presence",
) -> tuple[DOABundle, PipelineNodeResult]:
    """Parse, validate, normalize, and exactly align canonical DOA rows."""

    if dimension not in {"2d", "3d"}:
        raise ValueError("dimension must be '2d' or '3d'")
    if not math.isfinite(timestamp_tolerance_sec) or timestamp_tolerance_sec < 0:
        raise ValueError("timestamp_tolerance_sec must be finite and non-negative")
    if prediction_activity_policy not in {"presence", "explicit"}:
        raise ValueError("prediction_activity_policy must be 'presence' or 'explicit'")

    reference = tuple(_load_frames(reference_jsonl, role="reference", dimension=dimension))
    prediction = tuple(_load_frames(sample_output, role="prediction", dimension=dimension))
    if not reference:
        raise ValueError("reference_jsonl must contain at least one frame")
    if not prediction:
        raise ValueError("sample_output must contain at least one frame")

    ref_keys = [(row.recording_id, row.frame_index) for row in reference]
    pred_keys = [(row.recording_id, row.frame_index) for row in prediction]
    if ref_keys != pred_keys:
        raise ValueError(
            "reference and prediction rows must have identical recording_id/frame_index keys"
        )
    for ref, pred in zip(reference, prediction):
        if abs(ref.timestamp - pred.timestamp) > timestamp_tolerance_sec:
            raise ValueError(
                f"timestamp mismatch for {ref.recording_id}/{ref.frame_index}: "
                f"{ref.timestamp} vs {pred.timestamp}"
            )
        if ref.num_channels != pred.num_channels:
            raise ValueError(f"num_channels mismatch for {ref.recording_id}/{ref.frame_index}")
        if ref.coordinate_system != pred.coordinate_system:
            raise ValueError(f"coordinate_system mismatch for {ref.recording_id}/{ref.frame_index}")

    recordings = sorted({row.recording_id for row in reference})
    summary = {
        "num_recordings": len(recordings),
        "recording_ids": recordings,
        "num_frames": len(reference),
        "num_channels": sorted({row.num_channels for row in reference}),
        "coordinate_system": sorted({row.coordinate_system for row in reference}),
        "angle_unit": "degree",
        "dimension": dimension,
        "timestamp_tolerance_sec": timestamp_tolerance_sec,
        "input_order": "normalized_by_recording_id_then_frame_index",
    }
    bundle = DOABundle(
        reference=reference,
        prediction=prediction,
        dimension=dimension,
        alignment="exact_frame_index_with_timestamp_tolerance",
        prediction_activity_policy=prediction_activity_policy,
        input_summary=summary,
    )
    result = PipelineNodeResult(
        stage="normalization",
        node_id=NODE_ID,
        version=NODE_VERSION,
        details={
            "backend": "canonical_doa_jsonl",
            "dimension": dimension,
            "alignment": bundle.alignment,
            "prediction_activity_policy": prediction_activity_policy,
            "input_summary": summary,
        },
        internal_stages=INTERNAL_STAGES,
    )
    return bundle, result


def frame_from_payload(payload: dict[str, Any], *, role: str, dimension: str) -> DOAFrame:
    """Build one normalized standard DOA frame."""

    if not isinstance(payload, dict):
        raise ValueError(f"{role} row must be a JSON object")
    required = {
        "recording_id",
        "frame_index",
        "timestamp",
        "num_channels",
        "coordinate_system",
        "angle_unit",
        "sources",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{role} row missing required field(s): {', '.join(missing)}")
    recording_id = payload["recording_id"]
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError(f"{role}.recording_id must be a non-empty string")
    frame_index = payload["frame_index"]
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError(f"{role}.frame_index must be a non-negative integer")
    timestamp = _finite_number(payload["timestamp"], f"{role}.timestamp")
    if timestamp < 0:
        raise ValueError(f"{role}.timestamp must be non-negative")
    num_channels = payload["num_channels"]
    if isinstance(num_channels, bool) or not isinstance(num_channels, int) or num_channels < 2:
        raise ValueError(f"{role}.num_channels must be an integer >= 2 for multi-channel DOA")
    coordinate_system = payload["coordinate_system"]
    if coordinate_system != "array_local_spherical":
        raise ValueError("only coordinate_system='array_local_spherical' is supported")
    angle_unit = payload["angle_unit"]
    if angle_unit != "degree":
        raise ValueError("only angle_unit='degree' is supported")
    sources_payload = payload["sources"]
    if not isinstance(sources_payload, list):
        raise ValueError(f"{role}.sources must be a list")
    sources: list[DOASource] = []
    seen_ids: set[str] = set()
    for index, source_payload in enumerate(sources_payload):
        if not isinstance(source_payload, dict):
            raise ValueError(f"{role}.sources[{index}] must be an object")
        source_id = source_payload.get("source_id", f"source_{index}")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"{role}.sources[{index}].source_id must be a non-empty string")
        if source_id in seen_ids:
            raise ValueError(
                f"duplicate source_id {source_id!r} in {role}/{recording_id}/{frame_index}"
            )
        seen_ids.add(source_id)
        azimuth = _normalize_azimuth(source_payload.get("azimuth"), f"{role}.azimuth")
        elevation_value = source_payload.get("elevation")
        if dimension == "3d" and elevation_value is None:
            raise ValueError(f"{role}.sources[{index}] requires elevation for 3d")
        if dimension == "2d" and elevation_value is not None:
            raise ValueError(f"{role}.sources[{index}] must not include elevation in 2d mode")
        elevation = (
            None
            if elevation_value is None
            else _finite_number(elevation_value, f"{role}.elevation")
        )
        if elevation is not None and not -90.0 <= elevation <= 90.0:
            raise ValueError(f"{role}.elevation must be within [-90, 90] degrees")
        activity = source_payload.get("activity", True)
        if not isinstance(activity, bool):
            raise ValueError(f"{role}.activity must be boolean")
        confidence_value = source_payload.get("confidence")
        confidence = (
            None
            if confidence_value is None
            else _finite_number(confidence_value, f"{role}.confidence")
        )
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{role}.confidence must be within [0, 1]")
        sources.append(DOASource(source_id, azimuth, elevation, activity, confidence))
    return DOAFrame(
        recording_id=recording_id.strip(),
        frame_index=frame_index,
        timestamp=timestamp,
        num_channels=num_channels,
        coordinate_system=coordinate_system,
        angle_unit=angle_unit,
        sources=tuple(sources),
    )


def _load_frames(path: str | Path, *, role: str, dimension: str) -> list[DOAFrame]:
    frames: list[DOAFrame] = []
    seen: set[tuple[str, int]] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            frame = frame_from_payload(payload, role=role, dimension=dimension)
            key = (frame.recording_id, frame.frame_index)
            if key in seen:
                raise ValueError(f"duplicate recording_id/frame_index {key!r} in {role}")
            seen.add(key)
            frames.append(frame)
    frames.sort(key=lambda row: (row.recording_id, row.frame_index))
    return frames


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _normalize_azimuth(value: Any, field: str) -> float:
    angle = _finite_number(value, field)
    return ((angle + 180.0) % 360.0) - 180.0
