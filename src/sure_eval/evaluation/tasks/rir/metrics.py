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
) -> list[dict]:
    """Validate the channel branch and describe the inputs actually scored."""

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
    rows = []
    for sample in samples:
        prediction, reference, rate = _load_pair(sample)
        channel_count = prediction.shape[1]
        if normalized == "single" and channel_count != 1:
            raise ValueError(f"{sample.sample_id} requires a single-channel RIR")
        if normalized == "multi" and channel_count < 2:
            raise ValueError(f"{sample.sample_id} requires a multi-channel RIR")
        if normalized == "binaural" and channel_count != 2:
            raise ValueError(f"{sample.sample_id} requires a binaural two-channel RIR")
        if normalized == "auto" and reference.shape[1] != channel_count:
            raise ValueError(f"channel counts differ for {sample.sample_id}")
        rows.append({
            "sample_id": sample.sample_id,
            "prediction_rir": sample.prediction_rir,
            "reference_rir": sample.reference_rir,
            "sample_rate": rate,
            "channel_layout": sample.channel_layout,
            "channel_count": channel_count,
            "metadata": dict(sample.metadata),
            "prediction_samples": prediction.shape[0],
            "reference_samples": reference.shape[0],
            "paired_samples": min(prediction.shape[0], reference.shape[0]),
            "length_policy": "truncate_to_shorter",
        })
    return rows


def _load_pair(sample: RIRSample) -> tuple[np.ndarray, np.ndarray, int]:
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
    return prediction, reference, rate


def _pair(sample: RIRSample) -> tuple[np.ndarray, np.ndarray, int]:
    prediction, reference, rate = _load_pair(sample)
    length = min(prediction.shape[0], reference.shape[0])
    return prediction[:length], reference[:length], rate


def _binaural_pair(sample: RIRSample) -> tuple[np.ndarray, np.ndarray, int]:
    prediction, reference, rate = _pair(sample)
    if prediction.shape[1] != 2:
        raise ValueError(f"{sample.sample_id} requires exactly two binaural channels")
    return prediction, reference, rate


def _spatial_pair(sample: RIRSample) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a multi-channel RIR for ch0-relative spatial scoring."""
    prediction, reference, rate = _pair(sample)
    if prediction.shape[1] < 2:
        raise ValueError(f"{sample.sample_id} requires at least two spatial channels")
    return prediction, reference, rate


def _channel_pairs(signal: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    """Yield each channel paired with channel 0 as the spatial reference."""
    for channel in range(1, signal.shape[1]):
        yield channel, signal[:, [0, channel]]


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
            raise ValueError(f"NPM requires nonzero RIR energy for {sample.sample_id}")
        max_lag = min(256, prediction.shape[0] - 1, reference.shape[0] - 1)
        fft_length = reference.shape[0] + prediction.shape[0] - 1
        correlation = np.zeros(fft_length, dtype="float64")
        for channel in range(reference.shape[1]):
            reference_fft = np.fft.rfft(reference[:, channel], fft_length)
            prediction_fft = np.fft.rfft(prediction[:, channel], fft_length)
            full = np.fft.irfft(reference_fft * np.conj(prediction_fft), fft_length)
            correlation += np.concatenate((full[-(prediction.shape[0] - 1) :], full[: reference.shape[0]]))
        center = reference.shape[0] - 1
        # The legacy loop indexes negative lags first; reverse the standard
        # correlation slice to preserve both scores and reported lag signs.
        lag_values = correlation[center - max_lag : center + max_lag + 1][::-1]
        best_index = int(np.argmax(np.abs(lag_values)))
        correlation_max = float(abs(lag_values[best_index]))
        best_lag = int(best_index - max_lag)
        cosine_sq = min(1.0, correlation_max * correlation_max / (reference_energy * prediction_energy))
        score = float(10.0 * np.log10(max(1.0 - cosine_sq, 1e-20)))
        values.append(score)
        per_sample.append({
            "sample_id": sample.sample_id,
            "score": score,
            "reference_energy": reference_energy,
            "prediction_energy": prediction_energy,
            "best_lag_samples": best_lag,
            "projection_cosine_sq": cosine_sq,
            "unit": "dB",
        })
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


def _fit_decay_duration(edc_db: np.ndarray, sample_rate: int, decay_db: float) -> float:
    """Return the measured time between -5 dB and -(5+decay_db) dB."""
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
    return float(end - start) / float(sample_rate)


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


def _rt30(signal: np.ndarray, sample_rate: int) -> float:
    x = np.ascontiguousarray(np.asarray(signal, dtype="float64").reshape(-1))
    if x.size < 4:
        return float("nan")
    energy = np.cumsum((x**2)[::-1])[::-1]
    edc_db = 10.0 * np.log10(energy / (energy[0] + _EPSILON) + _EPSILON)
    return _fit_decay_duration(edc_db, sample_rate, 30.0)


def _decay_metric(samples: list[RIRSample], *, metric_name: str, field_name: str, estimator) -> MetricResult:
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, rate = _pair(sample)
        errors = []
        channel_rows = []
        for channel in range(prediction.shape[1]):
            predicted, target = _acoustic_pair(prediction[:, channel], reference[:, channel], rate)
            pred = estimator(predicted, rate)
            ref = estimator(target, rate)
            if not (np.isfinite(pred) and np.isfinite(ref)):
                raise ValueError(f"{field_name.upper()} could not be estimated for {sample.sample_id}, channel {channel}")
            error = float(abs(pred - ref))
            errors.append(error)
            channel_rows.append({
                "channel": channel,
                f"prediction_{field_name}": float(pred),
                f"reference_{field_name}": float(ref),
                "score": error,
                "unit": "seconds",
            })
        score = float(np.mean(errors))
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score, "channels": channel_rows, "unit": "seconds"})
    return MetricResult(metric_name, _mean(values), {"per_sample": per_sample, "higher_is_better": False, "unit": "seconds"})


def rt30_mae(samples: list[RIRSample]) -> MetricResult:
    return _decay_metric(samples, metric_name="rt30_mae", field_name="rt30", estimator=_rt30)


def rt60_mae(samples: list[RIRSample]) -> MetricResult:
    values = []
    per_sample = []
    for sample in samples:
        prediction, reference, rate = _pair(sample)
        errors = []
        predictions = []
        references = []
        channel_rows = []
        for channel in range(prediction.shape[1]):
            predicted, target = _acoustic_pair(prediction[:, channel], reference[:, channel], rate)
            pred = _rt60(predicted, rate)
            ref = _rt60(target, rate)
            if not (np.isfinite(pred) and np.isfinite(ref)):
                raise ValueError(f"RT60 could not be estimated for {sample.sample_id}, channel {channel}")
            errors.append(abs(pred - ref))
            predictions.append(pred)
            references.append(ref)
            channel_rows.append({"channel": channel, "prediction_rt60": float(pred), "reference_rt60": float(ref), "score": float(abs(pred - ref)), "unit": "seconds"})
        score = float(np.mean(errors))
        values.append(score)
        per_sample.append({
            "sample_id": sample.sample_id,
            "score": score,
            "prediction_rt60": float(np.mean(predictions)),
            "reference_rt60": float(np.mean(references)),
            "channels": channel_rows,
            "unit": "seconds",
        })
    return MetricResult(
        "rt60_mae",
        _mean(values),
        {"per_sample": per_sample, "higher_is_better": False, "unit": "seconds"},
    )


def _edc_curve_stats(prediction: np.ndarray, reference: np.ndarray, floor_db: float = -40.0) -> tuple[float, float, float, int]:
    def curve(signal: np.ndarray) -> np.ndarray:
        energy = np.cumsum((signal**2)[::-1])[::-1]
        return 10.0 * np.log10(energy / (energy[0] + _EPSILON) + _EPSILON)

    predicted = curve(np.asarray(prediction, dtype="float64"))
    target = curve(np.asarray(reference, dtype="float64"))
    length = min(predicted.size, target.size)
    if length < 4:
        return float("nan"), float("nan"), float("nan"), 0
    predicted = predicted[:length]
    target = target[:length]
    valid = np.isfinite(predicted) & np.isfinite(target) & (target >= floor_db)
    if int(valid.sum()) < 4:
        return float("nan"), float("nan"), float("nan"), int(valid.sum())
    difference = predicted[valid] - target[valid]
    correlation = float("nan")
    if difference.size >= 2 and np.std(predicted[valid]) > _EPSILON and np.std(target[valid]) > _EPSILON:
        correlation = float(np.corrcoef(predicted[valid], target[valid])[0, 1])
    return (
        float(np.mean(np.abs(difference))),
        float(np.sqrt(np.mean(difference**2))),
        correlation,
        int(valid.sum()),
    )


def rir_edc_mae(samples: list[RIRSample]) -> MetricResult:
    values = []
    per_sample = []
    rmses = []
    correlations = []
    for sample in samples:
        prediction, reference, rate = _pair(sample)
        channel_errors = []
        channel_rows = []
        for channel in range(prediction.shape[1]):
            predicted, target = _acoustic_pair(prediction[:, channel], reference[:, channel], rate)
            mae, rmse, correlation, valid_points = _edc_curve_stats(predicted, target)
            if not np.isfinite(mae):
                raise ValueError(f"EDC could not be estimated for {sample.sample_id}, channel {channel}")
            channel_errors.append(mae)
            rmses.append(rmse)
            correlations.append(correlation)
            channel_rows.append({
                "channel": channel,
                "edc_mae_db": mae,
                "edc_rmse_db": rmse,
                "edc_correlation": correlation if np.isfinite(correlation) else None,
                "valid_points": valid_points,
                "unit": "dB",
            })
        score = float(np.mean(channel_errors))
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score, "channels": channel_rows, "unit": "dB"})
    finite_rmse = [value for value in rmses if np.isfinite(value)]
    finite_corr = [value for value in correlations if np.isfinite(value)]
    return MetricResult(
        "rir_edc_mae",
        _mean(values),
        {
            "per_sample": per_sample,
            "higher_is_better": False,
            "unit": "dB",
            "edc_rmse_db": float(np.mean(finite_rmse)) if finite_rmse else float("nan"),
            "edc_correlation": float(np.mean(finite_corr)) if finite_corr else None,
        },
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
        channel_rows = []
        for channel in range(prediction.shape[1]):
            predicted, target = _acoustic_pair(prediction[:, channel], reference[:, channel], rate)
            pred_early, pred_late = _early_late(predicted, rate, window_ms)
            ref_early, ref_late = _early_late(target, rate, window_ms)
            if not np.isfinite(pred_early + pred_late + ref_early + ref_late):
                raise ValueError(f"C{int(window_ms)} could not be estimated for {sample.sample_id}, channel {channel}")
            prediction_value = 10.0 * np.log10(pred_early / pred_late)
            reference_value = 10.0 * np.log10(ref_early / ref_late)
            channel_score = abs(prediction_value - reference_value)
            channel_errors.append(channel_score)
            channel_rows.append({"channel": channel, "prediction_c": prediction_value, "reference_c": reference_value, "score": channel_score, "unit": "dB"})
        score = float(np.mean(channel_errors))
        values.append(score)
        per_sample.append({"sample_id": sample.sample_id, "score": score, "channels": channel_rows, "unit": "dB"})
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
        channel_rows = []
        for channel in range(prediction.shape[1]):
            predicted, target = _acoustic_pair(prediction[:, channel], reference[:, channel], rate)
            pred_early, pred_late = _early_late(predicted, rate, 2.5)
            ref_early, ref_late = _early_late(target, rate, 2.5)
            if not np.isfinite(pred_early + pred_late + ref_early + ref_late):
                raise ValueError(f"DRR could not be estimated for {sample.sample_id}, channel {channel}")
            pred_drr = 10.0 * np.log10(pred_early / pred_late)
            ref_drr = 10.0 * np.log10(ref_early / ref_late)
            channel_score = abs(pred_drr - ref_drr)
            channel_errors.append(channel_score)
            channel_rows.append({"channel": channel, "prediction_drr": pred_drr, "reference_drr": ref_drr, "score": channel_score, "unit": "dB"})
        score = float(np.mean(channel_errors))
        values.append(score)
        rows.append({"sample_id": sample.sample_id, "score": score, "channels": channel_rows, "unit": "dB"})
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
    left = np.asarray(left, dtype="float64").reshape(-1)
    right = np.asarray(right, dtype="float64").reshape(-1)
    length = min(left.size, right.size)
    if length < 2:
        return np.array([], dtype="int64"), np.array([], dtype="float64")
    left = left[:length] - float(np.mean(left[:length]))
    right = right[:length] - float(np.mean(right[:length]))
    # FFT correlation keeps the same full-correlation ordering as
    # ``np.correlate`` while avoiding its quadratic cost on 1-second RIRs.
    fft_length = left.size + right.size - 1
    correlation_fft = np.fft.irfft(
        np.fft.rfft(left, fft_length) * np.conj(np.fft.rfft(right, fft_length)),
        fft_length,
    )
    correlation = np.concatenate((correlation_fft[-(right.size - 1) :], correlation_fft[: left.size]))
    lags = np.arange(-len(left) + 1, len(left))
    max_lag = max(0, int(round(sample_rate * max_lag_ms / 1000.0)))
    mask = np.abs(lags) <= max_lag
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPSILON:
        raise ValueError("cross-correlation requires nonzero centered energy in both channels")
    return lags[mask], correlation[mask] / denominator


def _iacc(signal: np.ndarray, sample_rate: int, window: str = "full") -> float:
    left, right = signal[:, 0], signal[:, 1]
    length = min(left.size, right.size)
    if length < 2:
        return float("nan")
    left = left[:length]
    right = right[:length]
    anchor = int(np.argmax(np.abs(left)))
    early_end = min(length, anchor + int(round(sample_rate * 80.0 / 1000.0)))
    if window == "early":
        left, right = left[anchor:early_end], right[anchor:early_end]
    elif window == "late":
        left, right = left[early_end:], right[early_end:]
    _, correlation = _cross_correlation(left, right, sample_rate)
    return float(np.max(np.abs(correlation))) if correlation.size else float("nan")


def _itd(signal: np.ndarray, sample_rate: int) -> float:
    left, right = signal[:, 0], signal[:, 1]
    lags, correlation = _cross_correlation(left, right, sample_rate)
    if not correlation.size or not np.any(np.abs(correlation) > _EPSILON):
        raise ValueError("ITD could not be estimated from the selected correlation window")
    return float(lags[int(np.argmax(np.abs(correlation)))]) / sample_rate


def _rir_iacc_mae(samples: list[RIRSample], window: str, metric_name: str) -> MetricResult:
    values = []
    rows = []
    for sample in samples:
        prediction, reference, rate = _binaural_pair(sample)
        prediction_iacc = _iacc(prediction, rate, window)
        reference_iacc = _iacc(reference, rate, window)
        if not (np.isfinite(prediction_iacc) and np.isfinite(reference_iacc)):
            raise ValueError(f"IACC {window} could not be estimated for {sample.sample_id}")
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
        metric_name, _mean(values), {"per_sample": rows, "higher_is_better": False, "unit": "coefficient"}
    )


def rir_iacc_full_mae(samples: list[RIRSample]) -> MetricResult:
    return _rir_iacc_mae(samples, "full", "rir_iacc_full_mae")


def rir_iacc_early_mae(samples: list[RIRSample]) -> MetricResult:
    return _rir_iacc_mae(samples, "early", "rir_iacc_early_mae")


def rir_iacc_late_mae(samples: list[RIRSample]) -> MetricResult:
    return _rir_iacc_mae(samples, "late", "rir_iacc_late_mae")


def rir_iacc_mae(samples: list[RIRSample]) -> MetricResult:
    """Compatibility alias for the full-window IACC error."""
    result = rir_iacc_full_mae(samples)
    return MetricResult("rir_iacc_mae", result.score, result.details)


def rir_itd_mae(samples: list[RIRSample]) -> MetricResult:
    values = []
    rows = []
    for sample in samples:
        prediction, reference, rate = _spatial_pair(sample)
        pair_rows = []
        for channel, prediction_pair in _channel_pairs(prediction):
            reference_pair = reference[:, [0, channel]]
            prediction_itd = _itd(prediction_pair, rate) * 1000.0
            reference_itd = _itd(reference_pair, rate) * 1000.0
            score = float(abs(prediction_itd - reference_itd))
            pair_rows.append({
                "channel": channel,
                "score": score,
                "prediction_itd": prediction_itd,
                "reference_itd": reference_itd,
                "unit": "ms",
            })
        sample_score = float(np.mean([row["score"] for row in pair_rows]))
        values.append(sample_score)
        rows.append({
            "sample_id": sample.sample_id,
            "score": sample_score,
            "pair_count": len(pair_rows),
            "pairs": pair_rows,
            "unit": "ms",
        })
    return MetricResult(
        "rir_itd_mae",
        _mean(values),
        {"per_sample": rows, "higher_is_better": False, "unit": "ms"},
    )


def _spectral_pair_error(samples: list[RIRSample], phase: bool) -> MetricResult:
    values = []
    rows = []
    unit = "radians" if phase else "dB"
    for sample in samples:
        prediction, reference, _ = _spatial_pair(sample)
        pair_rows = []
        for channel, prediction_pair in _channel_pairs(prediction):
            reference_pair = reference[:, [0, channel]]
            pred_fft = np.fft.rfft(prediction_pair, axis=0)
            ref_fft = np.fft.rfft(reference_pair, axis=0)
            pred_magnitude = np.abs(pred_fft)
            ref_magnitude = np.abs(ref_fft)
            mask = (
                (pred_magnitude[:, 0] > _EPSILON)
                & (pred_magnitude[:, 1] > _EPSILON)
                & (ref_magnitude[:, 0] > _EPSILON)
                & (ref_magnitude[:, 1] > _EPSILON)
            )
            if not np.any(mask):
                raise ValueError(f"no valid spectral bins for {sample.sample_id}, channel {channel}")
            if phase:
                pred_value = np.angle(pred_fft[mask, 0] / pred_fft[mask, 1])
                ref_value = np.angle(ref_fft[mask, 0] / ref_fft[mask, 1])
                delta = np.angle(np.exp(1j * (pred_value - ref_value)))
                absolute_errors = np.abs(delta)
                prediction_value = float(np.mean(pred_value))
                reference_value = float(np.mean(ref_value))
            else:
                pred_value = 20.0 * np.log10(pred_magnitude[mask, 0] / pred_magnitude[mask, 1])
                ref_value = 20.0 * np.log10(ref_magnitude[mask, 0] / ref_magnitude[mask, 1])
                absolute_errors = np.abs(pred_value - ref_value)
                prediction_value = float(np.mean(pred_value))
                reference_value = float(np.mean(ref_value))
            absolute_error_sum = float(np.sum(absolute_errors))
            valid_bin_count = int(mask.sum())
            score = absolute_error_sum / valid_bin_count
            pair_rows.append({
                "channel": channel,
                "score": score,
                "prediction_ild" if not phase else "prediction_ipd": prediction_value,
                "reference_ild" if not phase else "reference_ipd": reference_value,
                "absolute_error_sum": absolute_error_sum,
                "valid_bin_count": valid_bin_count,
                "unit": unit,
            })
        name = "rir_ipd_mae" if phase else "rir_ild_mae"
        sample_score = float(np.mean([row["score"] for row in pair_rows]))
        values.append(sample_score)
        rows.append({
            "sample_id": sample.sample_id,
            "score": sample_score,
            "pair_count": len(pair_rows),
            "pairs": pair_rows,
            "unit": unit,
        })
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
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError(f"microphone_spacing_m is required for {sample.sample_id}")
        sound_speed = float(sample.metadata.get("sound_speed_mps", 343.0))
        if not np.isfinite(sound_speed) or sound_speed <= 0:
            raise ValueError(f"sound_speed_mps must be finite and positive for {sample.sample_id}")
        prediction, reference, rate = _spatial_pair(sample)
        pair_rows = []
        pair_spacings = sample.metadata.get("channel_spacings_m")
        if pair_spacings is not None and (
            not isinstance(pair_spacings, (list, tuple)) or len(pair_spacings) != prediction.shape[1] - 1
        ):
            raise ValueError(f"channel_spacings_m must contain one spacing per channel-0 pair for {sample.sample_id}")
        for channel, prediction_pair in _channel_pairs(prediction):
            reference_pair = reference[:, [0, channel]]
            pair_spacing = float(pair_spacings[channel - 1]) if pair_spacings else spacing * channel
            if not np.isfinite(pair_spacing) or pair_spacing <= 0:
                raise ValueError(f"invalid channel spacing for {sample.sample_id}, channel {channel}")
            pred_lag = _itd(prediction_pair, rate)
            ref_lag = _itd(reference_pair, rate)
            pred_angle = np.degrees(np.arcsin(np.clip(pred_lag * sound_speed / pair_spacing, -1.0, 1.0)))
            ref_angle = np.degrees(np.arcsin(np.clip(ref_lag * sound_speed / pair_spacing, -1.0, 1.0)))
            score = float(abs(pred_angle - ref_angle))
            pair_rows.append({
                "channel": channel,
                "score": score,
                "prediction_doa": float(pred_angle),
                "reference_doa": float(ref_angle),
                "prediction_itd_ms": float(pred_lag * 1000.0),
                "reference_itd_ms": float(ref_lag * 1000.0),
                "unit": "degrees",
            })
        sample_score = float(np.mean([row["score"] for row in pair_rows]))
        values.append(sample_score)
        rows.append({
            "sample_id": sample.sample_id,
            "score": sample_score,
            "pair_count": len(pair_rows),
            "pairs": pair_rows,
            "unit": "degrees",
        })
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
    "rir_edc_mae",
    "rir_iacc_mae",
    "rir_ild_mae",
    "rir_ipd_mae",
    "rir_itd_mae",
    "rir_npm",
    "rt30_mae",
    "rt60_mae",
    "iacc",
    "itd",
    "validate_rir_samples",
]
