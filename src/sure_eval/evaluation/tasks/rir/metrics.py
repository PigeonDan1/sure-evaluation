"""Deterministic RIR and spatial-audio metrics."""

from __future__ import annotations

import numpy as np
from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.tasks.rir.types import RIRSample


def _load(path: str) -> tuple[np.ndarray, int]:
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    if suffix in {"npy", "npz"}:
        loaded = np.load(path)
        if suffix == "npz":
            if "rir" in loaded:
                data = loaded["rir"]
            elif loaded.files:
                data = loaded[loaded.files[0]]
            else:
                raise ValueError(f"RIR archive is empty: {path}")
        else:
            data = loaded
        data = np.asarray(data, dtype="float64")
        if data.ndim == 1:
            data = data[:, None]
        elif data.ndim == 2 and data.shape[0] <= data.shape[1]:
            # Dataset RIR arrays are commonly stored as (channels, samples),
            # while soundfile returns (samples, channels).
            data = data.T
        if data.ndim != 2:
            raise ValueError(f"RIR array must be 1-D or 2-D: {path}")
        if data.size == 0 or not np.isfinite(data).all():
            raise ValueError(f"RIR audio is empty or non-finite: {path}")
        return data, 0
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("RIR scoring requires the audio extra: pip install 'sure-evaluation[audio]'") from exc
    data, rate = sf.read(path, always_2d=True, dtype="float64")
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError(f"RIR audio is empty or non-finite: {path}")
    return np.asarray(data), int(rate)


def _pair(sample: RIRSample) -> tuple[np.ndarray, np.ndarray, int]:
    prediction, prediction_rate = _load(sample.prediction_rir)
    reference, reference_rate = _load(sample.reference_rir)
    rate = int(sample.sample_rate or reference_rate)
    if rate <= 0:
        raise ValueError(f"sample_rate is required for RIR arrays without embedded rate: {sample.sample_id}")
    if prediction_rate and prediction_rate != rate:
        raise ValueError(f"sample rates differ for {sample.sample_id}: {prediction_rate}, {rate}")
    if reference_rate and reference_rate != rate:
        raise ValueError(f"sample rates differ for {sample.sample_id}: {reference_rate}, {rate}")
        raise ValueError(f"sample rates differ for {sample.sample_id}: {prediction_rate}, {reference_rate}, {rate}")
    if prediction.shape[1] != reference.shape[1]:
        raise ValueError(f"channel counts differ for {sample.sample_id}")
    length = min(prediction.shape[0], reference.shape[0])
    if length == 0:
        raise ValueError(f"RIR pair is empty for {sample.sample_id}")
    return prediction[:length], reference[:length], rate


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("no RIR metric values were produced")
    return float(np.mean(values))


def rir_nmse(samples: list[RIRSample]) -> MetricResult:
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, _ = _pair(sample)
        denominator = float(np.mean(reference**2))
        score = float(np.mean((prediction - reference) ** 2) / max(denominator, 1e-12))
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score})
    return MetricResult("rir_nmse", _mean(values), {"per_sample": per_sample, "higher_is_better": False})


def rir_corr(samples: list[RIRSample]) -> MetricResult:
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, _ = _pair(sample)
        channel_scores = []
        for channel in range(reference.shape[1]):
            left = prediction[:, channel] - np.mean(prediction[:, channel])
            right = reference[:, channel] - np.mean(reference[:, channel])
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            channel_scores.append(1.0 if denom == 0 and np.allclose(left, right) else float(np.dot(left, right) / denom) if denom else 0.0)
        score = _mean(channel_scores)
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score, "channels": channel_scores})
    return MetricResult("rir_corr", _mean(values), {"per_sample": per_sample, "higher_is_better": True})


def _rt60(signal: np.ndarray, sample_rate: int) -> float:
    energy = np.maximum(np.cumsum(signal[::-1] ** 2)[::-1], 1e-12)
    decay = 10.0 * np.log10(energy / energy[0])
    mask = (decay <= -5.0) & (decay >= -35.0)
    if int(mask.sum()) < 2:
        return float("nan")
    times = np.flatnonzero(mask) / sample_rate
    slope, _ = np.polyfit(times, decay[mask], 1)
    return float(-60.0 / slope) if slope < 0 else float("nan")


def rt60_mae(samples: list[RIRSample]) -> MetricResult:
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, rate = _pair(sample)
        pred = np.nanmean([_rt60(prediction[:, c], rate) for c in range(prediction.shape[1])])
        ref = np.nanmean([_rt60(reference[:, c], rate) for c in range(reference.shape[1])])
        if not np.isfinite(pred) or not np.isfinite(ref):
            raise ValueError(f"RT60 could not be estimated for {sample.sample_id}")
        score = float(abs(pred - ref))
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score, "prediction_rt60": float(pred), "reference_rt60": float(ref)})
    return MetricResult("rt60_mae", _mean(values), {"per_sample": per_sample, "higher_is_better": False, "unit": "seconds"})


def _early_late(signal: np.ndarray, rate: int, window_ms: float) -> tuple[float, float]:
    peak = int(np.argmax(np.max(np.abs(signal), axis=1)))
    window = max(1, int(rate * window_ms / 1000.0))
    end = min(signal.shape[0], peak + window)
    early = float(np.sum(signal[max(0, peak - window // 4):end] ** 2))
    late = float(np.sum(signal[end:] ** 2))
    return early, max(late, 1e-12)


def _clarity_mae(samples: list[RIRSample], window_ms: float) -> MetricResult:
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, rate = _pair(sample)
        pred_early, pred_late = _early_late(prediction, rate, window_ms)
        ref_early, ref_late = _early_late(reference, rate, window_ms)
        score = float(abs(10.0 * np.log10(pred_early / pred_late) - 10.0 * np.log10(ref_early / ref_late)))
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score, "unit": "dB"})
    name = f"rir_c{int(window_ms)}_mae"
    return MetricResult(name, _mean(values), {"per_sample": per_sample, "higher_is_better": False, "unit": "dB"})


def rir_drr_mae(samples: list[RIRSample]) -> MetricResult:
    values = []
    rows = []
    for sample in samples:
        prediction, reference, rate = _pair(sample)
        pred_early, pred_late = _early_late(prediction, rate, 2.5)
        ref_early, ref_late = _early_late(reference, rate, 2.5)
        score = abs(float(10 * np.log10(pred_early / pred_late)) - float(10 * np.log10(ref_early / ref_late)))
        values.append(score)
        rows.append({"sample_id": sample.sample_id, "score": float(score), "unit": "dB"})
    return MetricResult("rir_drr_mae", _mean(values), {"per_sample": rows, "higher_is_better": False, "unit": "dB"})


def rir_c50_mae(samples: list[RIRSample]) -> MetricResult:
    return _clarity_mae(samples, 50.0)


def rir_c80_mae(samples: list[RIRSample]) -> MetricResult:
    return _clarity_mae(samples, 80.0)


def _spectral_pair_error(samples: list[RIRSample], phase: bool) -> MetricResult:
    values = []
    rows = []
    for sample in samples:
        prediction, reference, _ = _pair(sample)
        if prediction.shape[1] < 2:
            raise ValueError(f"{sample.sample_id} requires at least two channels")
        pred_fft = np.fft.rfft(prediction[:, :2], axis=0)
        ref_fft = np.fft.rfft(reference[:, :2], axis=0)
        if phase:
            pred_denominator = np.where(np.abs(pred_fft[:, 1]) > 1e-12, pred_fft[:, 1], 1.0)
            ref_denominator = np.where(np.abs(ref_fft[:, 1]) > 1e-12, ref_fft[:, 1], 1.0)
            pred_value = np.angle(pred_fft[:, 0] / pred_denominator)
            ref_value = np.angle(ref_fft[:, 0] / ref_denominator)
            delta = np.angle(np.exp(1j * (pred_value - ref_value)))
            score = float(np.mean(np.abs(delta)))
            unit = "radians"
            name = "rir_ipd_mae"
        else:
            pred_value = 20 * np.log10(np.maximum(np.abs(pred_fft[:, 0]), 1e-12) / np.maximum(np.abs(pred_fft[:, 1]), 1e-12))
            ref_value = 20 * np.log10(np.maximum(np.abs(ref_fft[:, 0]), 1e-12) / np.maximum(np.abs(ref_fft[:, 1]), 1e-12))
            score = float(np.mean(np.abs(pred_value - ref_value)))
            unit = "dB"
            name = "rir_ild_mae"
        values.append(score)
        rows.append({"sample_id": sample.sample_id, "score": score, "unit": unit})
    return MetricResult(name, _mean(values), {"per_sample": rows, "higher_is_better": False, "unit": unit})


def rir_ild_mae(samples: list[RIRSample]) -> MetricResult:
    return _spectral_pair_error(samples, phase=False)


def rir_ipd_mae(samples: list[RIRSample]) -> MetricResult:
    return _spectral_pair_error(samples, phase=True)


def rir_doa_mae(samples: list[RIRSample]) -> MetricResult:
    values = []
    rows = []
    for sample in samples:
        spacing = float(sample.metadata.get("microphone_spacing_m", 0.0))
        if spacing <= 0:
            raise ValueError(f"microphone_spacing_m is required for {sample.sample_id}")
        sound_speed = float(sample.metadata.get("sound_speed_mps", 343.0))
        prediction, reference, rate = _pair(sample)
        if prediction.shape[1] < 2:
            raise ValueError(f"{sample.sample_id} requires at least two channels")
        pred_lag = int(np.argmax(np.correlate(prediction[:, 0], prediction[:, 1], mode="full")) - len(prediction) + 1)
        ref_lag = int(np.argmax(np.correlate(reference[:, 0], reference[:, 1], mode="full")) - len(reference) + 1)
        pred_angle = np.degrees(np.arcsin(np.clip(pred_lag / rate * sound_speed / spacing, -1.0, 1.0)))
        ref_angle = np.degrees(np.arcsin(np.clip(ref_lag / rate * sound_speed / spacing, -1.0, 1.0)))
        score = float(abs(pred_angle - ref_angle))
        values.append(score)
        rows.append({"sample_id": sample.sample_id, "score": score, "unit": "degrees"})
    return MetricResult("rir_doa_mae", _mean(values), {"per_sample": rows, "higher_is_better": False, "unit": "degrees"})
