"""Shared helpers for provider-backed audio scoring nodes."""

from __future__ import annotations

from statistics import fmean
from typing import Any, Callable, Dict, List, Tuple, Union

from sure_eval.evaluation.core.types import PipelineNodeResult

SpeakerProvider = Callable[..., Union[float, Dict[str, Any]]]
SpeakerRow = Tuple[str, str, str]
MOSProvider = Callable[..., Union[float, Dict[str, Any]]]
MOSRow = Tuple[str, str]

SPEAKER_INTERNAL_STAGES = ("embedding_or_score_provider", "score_normalization", "mean_aggregation")
MOS_INTERNAL_STAGES = ("audio_score_provider", "score_normalization", "mean_aggregation")
_RUNTIME_DETAIL_KEYS = {
    "audio_path",
    "decode_time",
    "decode_time_ms",
    "duration",
    "duration_ms",
    "duration_sec",
    "duration_seconds",
    "elapsed",
    "elapsed_ms",
    "elapsed_sec",
    "elapsed_seconds",
    "filename",
    "latency",
    "latency_ms",
    "len_in_sec",
    "num_hops",
    "prediction_audio",
    "reference_audio",
    "reference_text",
    "realtime_factor",
    "real_time_factor",
    "rtf",
    "speed",
    "sr",
    "throughput",
}


def _score_key(metric_name: str) -> str:
    if metric_name.startswith("sim/") or metric_name == "sim":
        return "ASV"
    if metric_name == "dnsmos":
        return "OVRL"
    if metric_name == "wv-mos":
        return "mos"
    if metric_name == "utmos":
        return "utmos"
    return "score"


def _fallback_score_key(row: dict[str, Any], metric_name: str) -> str | None:
    candidates = {
        "dnsmos": ("ovrl", "overall", "P808_MOS", "p808_mos", "mos", "score"),
        "wv-mos": ("mos", "wv_mos", "score"),
        "utmos": ("utmos", "mos", "score"),
    }.get(metric_name, ("score", "avg_score", "avg score", "hyp_score", "similarity", "sim", "cosine_similarity"))
    for key in candidates:
        if key in row:
            return key
    return None


def _normalize_provider_row(raw_result: float | dict[str, Any], *, metric_name: str) -> dict[str, Any]:
    score_key = _score_key(metric_name)
    if isinstance(raw_result, (float, int)):
        return {score_key: float(raw_result)}
    if not isinstance(raw_result, dict):
        raise TypeError("score_provider must return a float or a dict")

    row = {
        key: value
        for key, value in raw_result.items()
        if str(key).lower() not in _RUNTIME_DETAIL_KEYS
    }
    if score_key not in row:
        fallback = _fallback_score_key(row, metric_name)
        if fallback is None:
            raise KeyError(f"score_provider result must contain '{score_key}' or a recognized metric score key")
        row[score_key] = row[fallback]
    row[score_key] = float(row[score_key])
    return row


def _aggregate_rows(metric_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_key = _score_key(metric_name)
    result: dict[str, Any] = {
        "metric_name": metric_name,
        "score": fmean(float(row[score_key]) for row in rows),
        "num_samples": len(rows),
        "score_key": score_key,
        "per_sample": rows,
    }
    if metric_name == "dnsmos":
        for output_key, aliases in {
            "SIG": ("SIG", "sig"),
            "BAK": ("BAK", "bak"),
            "P808_MOS": ("P808_MOS", "p808_mos", "P808MOS"),
            "OVRL_raw": ("OVRL_raw", "ovrl_raw"),
            "SIG_raw": ("SIG_raw", "sig_raw"),
            "BAK_raw": ("BAK_raw", "bak_raw"),
        }.items():
            values = [float(row[key]) for row in rows for key in aliases if key in row]
            if values:
                result[f"mean_{output_key}"] = fmean(values)
    if metric_name.startswith("sim/") or metric_name == "sim":
        if all("ref_score" in row for row in rows):
            result["mean_ref_similarity"] = fmean(float(row["ref_score"]) for row in rows)
        variance_values = [
            float(row[key])
            for row in rows
            for key in ("ASV-var", "ASV_var", "score_var", "var")
            if key in row
        ]
        if variance_values:
            result["mean_ASV_var"] = fmean(variance_values)
    return result


def score_speaker_backend(
    rows: List[SpeakerRow],
    *,
    backend_name: str,
    metric_name: str,
    node_id: str,
    provider: SpeakerProvider,
    version: str = "v1",
) -> PipelineNodeResult:
    if not rows:
        raise ValueError("speaker similarity scoring requires at least one row")

    batch_provider = getattr(provider, "score_batch", None)
    if callable(batch_provider):
        raw_rows = batch_provider(rows, metric_name=metric_name)
        if len(raw_rows) != len(rows):
            raise RuntimeError(
                f"{node_id} returned {len(raw_rows)} speaker score row(s) for {len(rows)} input row(s)"
            )
        per_sample = [_normalize_provider_row(raw_row, metric_name=metric_name) for raw_row in raw_rows]
    else:
        per_sample = [
            _normalize_provider_row(provider(prediction, reference), metric_name=metric_name)
            for _key, prediction, reference in rows
        ]
    result = _aggregate_rows(metric_name, per_sample)
    return PipelineNodeResult(
        stage="scoring",
        node_id=node_id,
        version=version,
        details={
            "metric": metric_name,
            "backend": backend_name,
            "keys": [key for key, _prediction, _reference in rows],
            "result": result,
        },
        internal_stages=SPEAKER_INTERNAL_STAGES,
    )


def score_mos_backend(
    rows: List[MOSRow],
    *,
    metric_name: str,
    node_id: str,
    provider: MOSProvider,
    version: str = "v1",
) -> PipelineNodeResult:
    if not rows:
        raise ValueError("MOS scoring requires at least one row")

    batch_provider = getattr(provider, "score_batch", None)
    if callable(batch_provider):
        raw_rows = batch_provider(rows, metric_name=metric_name)
        if len(raw_rows) != len(rows):
            raise RuntimeError(f"{node_id} returned {len(raw_rows)} MOS score row(s) for {len(rows)} input row(s)")
        per_sample = [_normalize_provider_row(raw_row, metric_name=metric_name) for raw_row in raw_rows]
    else:
        per_sample = [
            _normalize_provider_row(provider(prediction, ""), metric_name=metric_name)
            for _key, prediction in rows
        ]
    result = _aggregate_rows(metric_name, per_sample)
    return PipelineNodeResult(
        stage="scoring",
        node_id=node_id,
        version=version,
        details={
            "metric": metric_name,
            "keys": [key for key, _prediction in rows],
            "result": result,
        },
        internal_stages=MOS_INTERNAL_STAGES,
    )
