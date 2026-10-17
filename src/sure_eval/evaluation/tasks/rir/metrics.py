"""Deterministic RIR and spatial-audio metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sure_eval.evaluation.base import MetricResult
from sure_eval.evaluation.tasks.rir.types import RIRSample

_EPSILON = 1e-12


def _load(path: str) -> tuple[np.ndarray, int]:
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    if suffix in {"npy", "npz"}:
        loaded = np.load(path)
        try:
            if suffix == "npz":
                if "rir" in loaded:
                    data = loaded["rir"]
                elif loaded.files:
                    data = loaded[loaded.files[0]]
                else:
                    raise ValueError(f"RIR archive is empty: {path}")
            else:
                data = loaded
        finally:
            if suffix == "npz":
                loaded.close()
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
        raise RuntimeError(
            "RIR scoring requires the audio extra: "
            "pip install 'sure-evaluation[audio]'"
        ) from exc
    data, rate = sf.read(path, always_2d=True, dtype="float64")
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError(f"RIR audio is empty or non-finite: {path}")
    return np.asarray(data), int(rate)


def _layout_kind(layout: str | None) -> str:
    normalized = str(layout or "auto").strip().lower().replace("-", "_")
    if normalized in {"", "auto", "any", "unknown"}:
        return "auto"
    if normalized in {"mono", "single", "single_channel", "1ch"}:
        return "single"
    if normalized in {"binaural", "binaural_lr", "stereo", "2ch"}:
        return "binaural"
    if normalized in {"multi", "multichannel", "multi_channel"}:
        return "multi"
    raise ValueError(f"unsupported channel_layout for {layout!r}")


def _validate_layout(sample: RIRSample, channel_count: int) -> None:
    layout = _layout_kind(sample.channel_layout)
    if layout == "single" and channel_count != 1:
        raise ValueError(f"{sample.sample_id} declares mono but has {channel_count} channels")
    if layout == "binaural" and channel_count != 2:
        raise ValueError(
            f"{sample.sample_id} declares binaural audio but has {channel_count} channels"
        )
    if layout == "multi" and channel_count < 2:
        raise ValueError(f"{sample.sample_id} declares multi-channel audio but has one channel")


def validate_rir_samples(
    samples: Iterable[RIRSample], *, channel_mode: str = "auto"
) -> None:
    """Validate the channel branch selected by an RIR evaluation chain."""

    normalized = str(channel_mode).strip().lower().replace("-", "_")
    aliases = {
        "mono": "single",
        "single_channel": "single",
        "multichannel": "multi",
        "multi_channel": "multi",
        "binaural_lr": "binaural",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "single", "multi", "binaural"}:
        raise ValueError(f"unsupported RIR channel mode: {channel_mode!r}")
    for sample in samples:
        prediction, reference, _ = _pair(sample)
        channel_count = prediction.shape[1]
        if normalized == "single" and channel_count != 1:
            raise ValueError(f"{sample.sample_id} requires a single-channel RIR")
        if normalized == "multi" and channel_count < 2:
            raise ValueError(f"{sample.sample_id} requires a multi-channel RIR")
        if normalized == "binaural" and channel_count != 2:
            raise ValueError(f"{sample.sample_id} requires a binaural two-channel RIR")
        if normalized == "auto" and reference.shape[1] != channel_count:
            raise ValueError(f"channel counts differ for {sample.sample_id}")


def _pair(sample: RIRSample) -> tuple[np.ndarray, np.ndarray, int]:
    prediction, prediction_rate = _load(sample.prediction_rir)
    reference, reference_rate = _load(sample.reference_rir)
    rate = int(sample.sample_rate or reference_rate or prediction_rate)
    if rate <= 0:
        raise ValueError(
            f"sample_rate is required for RIR arrays without embedded rate: {sample.sample_id}"
        )
    if prediction_rate and prediction_rate != rate:
        raise ValueError(f"sample rates differ for {sample.sample_id}: {prediction_rate}, {rate}")
    if reference_rate and reference_rate != rate:
        raise ValueError(f"sample rates differ for {sample.sample_id}: {reference_rate}, {rate}")
    if prediction.shape[1] != reference.shape[1]:
        raise ValueError(f"channel counts differ for {sample.sample_id}")
    _validate_layout(sample, prediction.shape[1])
    length = min(prediction.shape[0], reference.shape[0])
    if length == 0:
        raise ValueError(f"RIR pair is empty for {sample.sample_id}")
    return prediction[:length], reference[:length], rate


def _binaural_pair(sample: RIRSample) -> tuple[np.ndarray, np.ndarray, int]:
    prediction, reference, rate = _pair(sample)
    if prediction.shape[1] != 2:
        raise ValueError(f"{sample.sample_id} requires exactly two binaural channels")
    return prediction, reference, rate


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
        score = float(np.mean((prediction - reference) ** 2) / max(denominator, _EPSILON))
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score})
    return MetricResult(
        "rir_nmse", _mean(values), {"per_sample": per_sample, "higher_is_better": False}
    )


def rir_corr(samples: list[RIRSample]) -> MetricResult:
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, _ = _pair(sample)
        channel_scores = []
        for channel in range(reference.shape[1]):
            left = prediction[:, channel] - np.mean(prediction[:, channel])
            right = reference[:, channel] - np.mean(reference[:, channel])
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator == 0:
                channel_scores.append(1.0 if np.allclose(left, right) else 0.0)
            else:
                channel_scores.append(float(np.dot(left, right) / denominator))
        score = _mean(channel_scores)
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score, "channels": channel_scores})
    return MetricResult(
        "rir_corr", _mean(values), {"per_sample": per_sample, "higher_is_better": True}
    )


def _rt60(signal: np.ndarray, sample_rate: int) -> float:
    energy = np.maximum(np.cumsum(signal[::-1] ** 2)[::-1], _EPSILON)
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
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "score": score,
                "prediction_rt60": float(pred),
                "reference_rt60": float(ref),
            }
        )
    return MetricResult(
        "rt60_mae",
        _mean(values),
        {"per_sample": per_sample, "higher_is_better": False, "unit": "seconds"},
    )


def _early_late(signal: np.ndarray, rate: int, window_ms: float) -> tuple[float, float]:
    peak = int(np.argmax(np.max(np.abs(signal), axis=1)))
    window = max(1, int(rate * window_ms / 1000.0))
    end = min(signal.shape[0], peak + window)
    early = float(np.sum(signal[max(0, peak - window // 4) : end] ** 2))
    late = float(np.sum(signal[end:] ** 2))
    return early, max(late, _EPSILON)


def _clarity_mae(samples: list[RIRSample], window_ms: float) -> MetricResult:
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, rate = _pair(sample)
        pred_early, pred_late = _early_late(prediction, rate, window_ms)
        ref_early, ref_late = _early_late(reference, rate, window_ms)
        prediction_value = 10.0 * np.log10(pred_early / pred_late)
        reference_value = 10.0 * np.log10(ref_early / ref_late)
        score = float(abs(prediction_value - reference_value))
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score, "unit": "dB"})
    name = f"rir_c{int(window_ms)}_mae"
    return MetricResult(
        name, _mean(values), {"per_sample": per_sample, "higher_is_better": False, "unit": "dB"}
    )


def rir_drr_mae(samples: list[RIRSample]) -> MetricResult:
    values = []
    rows = []
    for sample in samples:
        prediction, reference, rate = _pair(sample)
        pred_early, pred_late = _early_late(prediction, rate, 2.5)
        ref_early, ref_late = _early_late(reference, rate, 2.5)
        pred_drr = 10.0 * np.log10(pred_early / pred_late)
        ref_drr = 10.0 * np.log10(ref_early / ref_late)
        score = float(abs(pred_drr - ref_drr))
        values.append(score)
        rows.append({"sample_id": sample.sample_id, "score": score, "unit": "dB"})
    return MetricResult(
        "rir_drr_mae", _mean(values), {"per_sample": rows, "higher_is_better": False, "unit": "dB"}
    )


def rir_c50_mae(samples: list[RIRSample]) -> MetricResult:
    return _clarity_mae(samples, 50.0)


def rir_c80_mae(samples: list[RIRSample]) -> MetricResult:
    return _clarity_mae(samples, 80.0)


def _cross_correlation(
    left: np.ndarray, right: np.ndarray, sample_rate: int, max_lag_ms: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    correlation = np.correlate(left, right, mode="full")
    lags = np.arange(-len(left) + 1, len(left))
    max_lag = max(0, int(round(sample_rate * max_lag_ms / 1000.0)))
    mask = np.abs(lags) <= max_lag
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPSILON:
        return lags[mask], np.zeros(int(mask.sum()), dtype="float64")
    return lags[mask], correlation[mask] / denominator


def _iacc(signal: np.ndarray, sample_rate: int) -> float:
    left, right = signal[:, 0], signal[:, 1]
    _, correlation = _cross_correlation(left, right, sample_rate)
    return float(np.max(np.abs(correlation))) if correlation.size else 0.0


def _itd(signal: np.ndarray, sample_rate: int) -> float:
    left, right = signal[:, 0], signal[:, 1]
    lags, correlation = _cross_correlation(left, right, sample_rate)
    if not correlation.size or not np.any(np.abs(correlation) > _EPSILON):
        return 0.0
    return float(lags[int(np.argmax(np.abs(correlation)))]) / sample_rate


def rir_iacc_mae(samples: list[RIRSample]) -> MetricResult:
    values = []
    rows = []
    for sample in samples:
        prediction, reference, rate = _binaural_pair(sample)
        prediction_iacc = _iacc(prediction, rate)
        reference_iacc = _iacc(reference, rate)
        score = float(abs(prediction_iacc - reference_iacc))
        values.append(score)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "score": score,
                "prediction_iacc": prediction_iacc,
                "reference_iacc": reference_iacc,
                "unit": "coefficient",
            }
        )
    return MetricResult(
        "rir_iacc_mae", _mean(values), {"per_sample": rows, "higher_is_better": False}
    )


def rir_itd_mae(samples: list[RIRSample]) -> MetricResult:
    values = []
    rows = []
    for sample in samples:
        prediction, reference, rate = _binaural_pair(sample)
        prediction_itd = _itd(prediction, rate)
        reference_itd = _itd(reference, rate)
        score = float(abs(prediction_itd - reference_itd))
        values.append(score)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "score": score,
                "prediction_itd": prediction_itd,
                "reference_itd": reference_itd,
                "unit": "seconds",
            }
        )
    return MetricResult(
        "rir_itd_mae",
        _mean(values),
        {"per_sample": rows, "higher_is_better": False, "unit": "seconds"},
    )


def _spectral_pair_error(samples: list[RIRSample], phase: bool) -> MetricResult:
    values = []
    rows = []
    for sample in samples:
        prediction, reference, _ = _binaural_pair(sample)
        pred_fft = np.fft.rfft(prediction[:, :2], axis=0)
        ref_fft = np.fft.rfft(reference[:, :2], axis=0)
        pred_magnitude = np.abs(pred_fft)
        ref_magnitude = np.abs(ref_fft)
        mask = (
            (pred_magnitude[:, 0] > _EPSILON)
            & (pred_magnitude[:, 1] > _EPSILON)
            & (ref_magnitude[:, 0] > _EPSILON)
            & (ref_magnitude[:, 1] > _EPSILON)
        )
        if not np.any(mask):
            score = 0.0
        elif phase:
            pred_value = np.angle(pred_fft[mask, 0] / pred_fft[mask, 1])
            ref_value = np.angle(ref_fft[mask, 0] / ref_fft[mask, 1])
            delta = np.angle(np.exp(1j * (pred_value - ref_value)))
            score = float(np.mean(np.abs(delta)))
        else:
            pred_value = 20.0 * np.log10(pred_magnitude[mask, 0] / pred_magnitude[mask, 1])
            ref_value = 20.0 * np.log10(ref_magnitude[mask, 0] / ref_magnitude[mask, 1])
            score = float(np.mean(np.abs(pred_value - ref_value)))
        unit = "radians" if phase else "dB"
        name = "rir_ipd_mae" if phase else "rir_ild_mae"
        values.append(score)
        rows.append({"sample_id": sample.sample_id, "score": score, "unit": unit})
    return MetricResult(
        name, _mean(values), {"per_sample": rows, "higher_is_better": False, "unit": unit}
    )


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
        prediction, reference, rate = _binaural_pair(sample)
        pred_lag = _itd(prediction, rate)
        ref_lag = _itd(reference, rate)
        pred_angle = np.degrees(np.arcsin(np.clip(pred_lag * sound_speed / spacing, -1.0, 1.0)))
        ref_angle = np.degrees(np.arcsin(np.clip(ref_lag * sound_speed / spacing, -1.0, 1.0)))
        score = float(abs(pred_angle - ref_angle))
        values.append(score)
        rows.append({"sample_id": sample.sample_id, "score": score, "unit": "degrees"})
    return MetricResult(
        "rir_doa_mae", _mean(values), {"per_sample": rows, "higher_is_better": False, "unit": "degrees"}
    )


# Short aliases keep direct task-level callers aligned with the public metric
# selectors while reports retain the explicit ``*_mae`` names.
iacc = rir_iacc_mae
itd = rir_itd_mae


__all__ = [
    "rir_c50_mae",
    "rir_c80_mae",
    "rir_corr",
    "rir_doa_mae",
    "rir_drr_mae",
    "rir_iacc_mae",
    "rir_ild_mae",
    "rir_ipd_mae",
    "rir_itd_mae",
    "rir_nmse",
    "rt60_mae",
    "iacc",
    "itd",
    "validate_rir_samples",
]
