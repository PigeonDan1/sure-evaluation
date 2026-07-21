"""Shared helpers for full-reference speech enhancement scoring nodes."""

from __future__ import annotations

import math
import wave
from statistics import fmean
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

from sure_eval.evaluation.core.types import PipelineNodeResult

FullReferenceAudioProvider = Callable[..., Union[float, Dict[str, Any]]]
FullReferenceAudioRow = Tuple[str, str, str]

FULL_REFERENCE_INTERNAL_STAGES = (
    "audio_pair_decode",
    "metric_provider",
    "score_normalization",
    "mean_aggregation",
)
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
    "enhanced_audio",
    "filename",
    "latency",
    "latency_ms",
    "len_in_sec",
    "noisy_audio",
    "prediction_audio",
    "reference_audio",
    "realtime_factor",
    "real_time_factor",
    "rtf",
    "sample_rate",
    "speed",
    "sr",
    "throughput",
}


def score_full_reference_audio_backend(
    rows: List[FullReferenceAudioRow],
    *,
    metric_name: str,
    node_id: str,
    provider: FullReferenceAudioProvider,
    version: str = "v1",
) -> PipelineNodeResult:
    if not rows:
        raise ValueError("full-reference audio scoring requires at least one row")

    batch_provider = getattr(provider, "score_batch", None)
    if callable(batch_provider):
        raw_rows = batch_provider(rows, metric_name=metric_name)
        if len(raw_rows) != len(rows):
            raise RuntimeError(
                f"{node_id} returned {len(raw_rows)} score row(s) for {len(rows)} input row(s)"
            )
        per_sample = [_normalize_provider_row(raw_row, metric_name=metric_name) for raw_row in raw_rows]
    else:
        per_sample = [
            _normalize_provider_row(provider(enhanced, reference), metric_name=metric_name)
            for _key, enhanced, reference in rows
        ]
    result = _aggregate_rows(metric_name, per_sample)
    return PipelineNodeResult(
        stage="scoring",
        node_id=node_id,
        version=version,
        details={
            "metric": metric_name,
            "keys": [key for key, _enhanced, _reference in rows],
            "result": result,
        },
        internal_stages=FULL_REFERENCE_INTERNAL_STAGES,
    )


def read_audio_mono(path: str, *, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Read audio as float32 mono, using stdlib wave for WAV and soundfile as fallback."""

    try:
        samples, sample_rate = _read_wav_mono(path)
    except (wave.Error, EOFError, OSError, ValueError):
        try:
            import soundfile as sf
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"cannot read non-WAV or unsupported audio file without soundfile: {path}"
            ) from exc
        data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        samples = np.asarray(data, dtype=np.float32).mean(axis=1)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        samples = resample_linear(samples, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return samples.astype(np.float32, copy=False), sample_rate


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if len(samples) == 0 or source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    target_length = max(1, int(round(len(samples) * target_rate / source_rate)))
    old_positions = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


def calculate_si_sdr(reference: np.ndarray, estimation: np.ndarray, *, eps: float = 1e-8) -> float:
    reference, estimation = align_pair(reference, estimation)
    reference = reference - float(np.mean(reference))
    estimation = estimation - float(np.mean(estimation))
    reference_energy = float(np.sum(reference * reference))
    if reference_energy <= eps:
        raise ValueError("reference audio is silent; SI-SDR is undefined")
    projection = float(np.sum(estimation * reference)) * reference / (reference_energy + eps)
    noise = estimation - projection
    projection_energy = float(np.sum(projection * projection))
    noise_energy = float(np.sum(noise * noise))
    if noise_energy <= eps:
        return 100.0
    return float(10.0 * math.log10((projection_energy + eps) / (noise_energy + eps)))


def align_pair(reference: np.ndarray, estimation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(reference), len(estimation))
    if length <= 0:
        raise ValueError("audio arrays must be non-empty")
    return reference[:length], estimation[:length]


class SISDRProvider:
    """In-process SI-SDR provider backed by numpy and stdlib WAV decoding."""

    def __call__(self, enhanced_audio: str, reference_audio: str, **kwargs: Any) -> dict[str, Any]:
        reference, sample_rate = read_audio_mono(reference_audio)
        enhanced, enhanced_sample_rate = read_audio_mono(enhanced_audio, target_sample_rate=sample_rate)
        score = calculate_si_sdr(reference, enhanced)
        return {
            "si_sdr": score,
            "score": score,
            "backend": "numpy-si-sdr",
            "sample_rate": sample_rate,
            "enhanced_sample_rate": enhanced_sample_rate,
        }


class STOIProvider:
    """STOI provider using the optional pystoi package."""

    def __init__(self, *, sample_rate: int = 16000, extended: bool = False) -> None:
        self.sample_rate = sample_rate
        self.extended = extended

    def __call__(self, enhanced_audio: str, reference_audio: str, **kwargs: Any) -> dict[str, Any]:
        try:
            from pystoi import stoi
        except ModuleNotFoundError as exc:
            raise RuntimeError("STOI scoring requires `python -m pip install pystoi`") from exc
        reference, _ = read_audio_mono(reference_audio, target_sample_rate=self.sample_rate)
        enhanced, _ = read_audio_mono(enhanced_audio, target_sample_rate=self.sample_rate)
        reference, enhanced = align_pair(reference, enhanced)
        score = float(stoi(reference, enhanced, self.sample_rate, extended=self.extended))
        return {
            "stoi": score,
            "score": score,
            "backend": "pystoi",
            "sample_rate": self.sample_rate,
            "extended": self.extended,
        }


class PESQProvider:
    """PESQ provider using the optional pesq package."""

    def __init__(self, *, sample_rate: int = 16000, mode: str = "wb") -> None:
        if mode not in {"wb", "nb"}:
            raise ValueError("PESQ mode must be 'wb' or 'nb'")
        if sample_rate not in {8000, 16000}:
            raise ValueError("PESQ sample_rate must be 8000 or 16000")
        if mode == "nb" and sample_rate != 8000:
            raise ValueError("PESQ narrow-band mode requires 8000 Hz")
        if mode == "wb" and sample_rate != 16000:
            raise ValueError("PESQ wide-band mode requires 16000 Hz")
        self.sample_rate = sample_rate
        self.mode = mode

    def __call__(self, enhanced_audio: str, reference_audio: str, **kwargs: Any) -> dict[str, Any]:
        try:
            from pesq import pesq
        except ModuleNotFoundError as exc:
            raise RuntimeError("PESQ scoring requires `python -m pip install pesq`") from exc
        reference, _ = read_audio_mono(reference_audio, target_sample_rate=self.sample_rate)
        enhanced, _ = read_audio_mono(enhanced_audio, target_sample_rate=self.sample_rate)
        reference, enhanced = align_pair(reference, enhanced)
        score = float(pesq(self.sample_rate, reference, enhanced, self.mode))
        return {
            "pesq": score,
            "score": score,
            "backend": "pesq",
            "sample_rate": self.sample_rate,
            "mode": self.mode,
        }


def _score_key(metric_name: str) -> str:
    return {
        "si-sdr": "si_sdr",
        "stoi": "stoi",
        "pesq": "pesq",
    }.get(metric_name, "score")


def _fallback_score_key(row: dict[str, Any], metric_name: str) -> str | None:
    candidates = {
        "si-sdr": ("si_sdr", "sisdr", "si-sdr", "sdr", "score"),
        "stoi": ("stoi", "estoi", "score"),
        "pesq": ("pesq", "mos_lqo", "score"),
    }.get(metric_name, ("score",))
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
    return {
        "metric_name": metric_name,
        "score": fmean(float(row[score_key]) for row in rows),
        "num_samples": len(rows),
        "score_key": score_key,
        "per_sample": rows,
    }


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
    samples = _pcm_bytes_to_float32(raw, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32, copy=False), sample_rate


def _pcm_bytes_to_float32(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        sign = (data[:, 2] & 0x80) != 0
        padded = np.zeros((len(data), 4), dtype=np.uint8)
        padded[:, :3] = data
        padded[sign, 3] = 0xFF
        return padded.view("<i4").reshape(-1).astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported WAV sample width: {sample_width}")
