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


def rir_npm(samples: list[RIRSample]) -> MetricResult:
    """Compute Rec-BIR's scale/lag-invariant normalized projection misalignment."""
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, _ = _pair(sample)
        reference_energy = float(np.sum(reference * reference))
        prediction_energy = float(np.sum(prediction * prediction))
        if reference_energy <= _EPSILON or prediction_energy <= _EPSILON:
            score = float("nan")
        else:
            cross = sum(
                np.correlate(reference[:, channel], prediction[:, channel], mode="full")
                for channel in range(reference.shape[1])
            )
            lags = np.arange(-prediction.shape[0] + 1, reference.shape[0])
            valid = np.abs(lags) <= 256
            correlation_max = float(np.max(np.abs(cross[valid])))
            cosine_sq = min(1.0, correlation_max * correlation_max / (reference_energy * prediction_energy))
            score = float(10.0 * np.log10(max(1.0 - cosine_sq, 1e-20)))
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score, "unit": "dB"})
    return MetricResult(
        "rir_npm", _mean(values),
        {"per_sample": per_sample, "higher_is_better": False, "unit": "dB"},
    )


def _preprocess_acoustic_signal(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(signal, dtype="float64").reshape(-1))
    if x.size == 0:
        return x
    peak = int(np.argmax(np.abs(x)))
    start = max(0, peak - int(round(sample_rate * 2.5 / 1000.0)))
    x = x[start:]
    peak_value = float(np.max(np.abs(x)))
    if peak_value > 1.0:
        x = x / peak_value
    return x


def _acoustic_pair(prediction: np.ndarray, reference: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    predicted = _preprocess_acoustic_signal(prediction, sample_rate)
    target = _preprocess_acoustic_signal(reference, sample_rate)
    length = min(predicted.size, target.size)
    predicted = predicted[:length]
    target = target[:length]
    if length and float(np.dot(predicted, target)) < 0.0:
        predicted = -predicted
    return predicted, target


def _fit_rt60(edc_db: np.ndarray, sample_rate: int, decay_db: float) -> float:
    upper_db = -5.0
    lower_db = -(5.0 + decay_db)
    below_upper = np.flatnonzero(edc_db <= upper_db)
    below_lower = np.flatnonzero(edc_db <= lower_db)
    if below_upper.size == 0 or below_lower.size == 0:
        return float("nan")
    start = int(below_upper[0])
    end = int(below_lower[0])
    if end - start < 4:
        return float("nan")
    times = np.arange(start, end + 1, dtype="float64") / sample_rate
    slope, _ = np.polyfit(times, edc_db[start : end + 1], 1)
    return float(-60.0 / slope) if slope < 0.0 else float("nan")


def _rt60(signal: np.ndarray, sample_rate: int) -> float:
    x = np.ascontiguousarray(np.asarray(signal, dtype="float64").reshape(-1))
    if x.size < 4:
        return float("nan")
    energy = np.cumsum((x**2)[::-1])[::-1]
    edc_db = 10.0 * np.log10(energy / (energy[0] + _EPSILON) + _EPSILON)
    return next(
        (value for value in (_fit_rt60(edc_db, sample_rate, 30.0), _fit_rt60(edc_db, sample_rate, 20.0)) if np.isfinite(value)),
        float("nan"),
    )


def rt60_mae(samples: list[RIRSample]) -> MetricResult:
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, rate = _pair(sample)
        errors = []
        predictions = []
        references = []
        for channel in range(prediction.shape[1]):
            predicted, target = _acoustic_pair(prediction[:, channel], reference[:, channel], rate)
            pred = _rt60(predicted, rate)
            ref = _rt60(target, rate)
            if np.isfinite(pred) and np.isfinite(ref):
                errors.append(abs(pred - ref))
                predictions.append(pred)
                references.append(ref)
        if not errors:
            raise ValueError(f"RT60 could not be estimated for {sample.sample_id}")
        score = float(np.mean(errors))
        values.append(score)
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "score": score,
                "prediction_rt60": float(np.mean(predictions)),
                "reference_rt60": float(np.mean(references)),
                "unit": "seconds",
            }
        )
    return MetricResult(
        "rt60_mae",
        _mean(values),
        {"per_sample": per_sample, "higher_is_better": False, "unit": "seconds"},
    )


def _early_late(signal: np.ndarray, rate: int, window_ms: float) -> tuple[float, float]:
    x = np.ascontiguousarray(np.asarray(signal, dtype="float64").reshape(-1))
    peak = int(np.argmax(np.abs(x)))
    window = max(1, int(round(rate * window_ms / 1000.0)))
    end = min(x.size, peak + window + 1)
    if end >= x.size:
        return float("nan"), float("nan")
    early = float(np.sum(x[peak:end] ** 2))
    late = float(np.sum(x[end:] ** 2))
    if early <= _EPSILON or late <= _EPSILON:
        return float("nan"), float("nan")
    return early, late


def _clarity_mae(samples: list[RIRSample], window_ms: float) -> MetricResult:
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, rate = _pair(sample)
        channel_errors = []
        for channel in range(prediction.shape[1]):
            predicted, target = _acoustic_pair(prediction[:, channel], reference[:, channel], rate)
            pred_early, pred_late = _early_late(predicted, rate, window_ms)
            ref_early, ref_late = _early_late(target, rate, window_ms)
            if np.isfinite(pred_early + pred_late + ref_early + ref_late):
                prediction_value = 10.0 * np.log10(pred_early / pred_late)
                reference_value = 10.0 * np.log10(ref_early / ref_late)
                channel_errors.append(abs(prediction_value - reference_value))
        if not channel_errors:
            raise ValueError(f"C{int(window_ms)} could not be estimated for {sample.sample_id}")
        score = float(np.mean(channel_errors))
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
        channel_errors = []
        for channel in range(prediction.shape[1]):
            predicted, target = _acoustic_pair(prediction[:, channel], reference[:, channel], rate)
            pred_early, pred_late = _early_late(predicted, rate, 2.5)
            ref_early, ref_late = _early_late(target, rate, 2.5)
            if np.isfinite(pred_early + pred_late + ref_early + ref_late):
                pred_drr = 10.0 * np.log10(pred_early / pred_late)
                ref_drr = 10.0 * np.log10(ref_early / ref_late)
                channel_errors.append(abs(pred_drr - ref_drr))
        if not channel_errors:
            raise ValueError(f"DRR could not be estimated for {sample.sample_id}")
        score = float(np.mean(channel_errors))
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
    "rir_doa_mae",
    "rir_drr_mae",
    "rir_iacc_mae",
    "rir_ild_mae",
    "rir_ipd_mae",
    "rir_itd_mae",
    "rir_npm",
    "rt60_mae",
    "iacc",
    "itd",
    "validate_rir_samples",
]
