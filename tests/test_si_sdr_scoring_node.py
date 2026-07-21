from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    sf.write(str(path), data.astype("float32"), sr)


def test_si_sdr_identical_signals_yield_infinity(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    signal = np.random.RandomState(42).randn(16000)
    pred = tmp_path / "pred.wav"
    ref = tmp_path / "ref.wav"
    _write_wav(pred, signal)
    _write_wav(ref, signal)

    result = score_si_sdr([("utt1", str(pred), str(ref))])
    score = result.details["result"]["per_sample"][0]["si_sdr"]
    assert math.isinf(score), f"expected inf for identical signals, got {score}"


def test_si_sdr_different_signals_yield_finite_score(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    rng = np.random.RandomState(42)
    ref_signal = rng.randn(16000)
    pred_signal = rng.randn(16000)
    pred = tmp_path / "pred.wav"
    ref = tmp_path / "ref.wav"
    _write_wav(pred, pred_signal)
    _write_wav(ref, ref_signal)

    result = score_si_sdr([("utt1", str(pred), str(ref))])
    score = result.details["result"]["per_sample"][0]["si_sdr"]
    assert not math.isinf(score), "expected finite score for different signals"
    assert not math.isnan(score), "expected non-NaN score"
    assert -200 < score < 200, f"score {score} outside expected range"


def test_si_sdr_lengths_aligned_by_truncation(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    short_signal = np.random.RandomState(7).randn(8000)
    long_signal = np.random.RandomState(42).randn(16000)
    long_signal[:8000] = short_signal
    pred = tmp_path / "pred.wav"
    ref = tmp_path / "ref.wav"
    _write_wav(pred, long_signal)
    _write_wav(ref, short_signal)

    result = score_si_sdr([("utt1", str(pred), str(ref))])
    score = result.details["result"]["per_sample"][0]["si_sdr"]
    assert math.isinf(score), f"expected inf when overlapping portion is identical, got {score}"


def test_si_sdr_mean_aggregation_over_multiple_rows(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    rng = np.random.RandomState(42)
    rows = []
    expected_scores = []
    for i in range(3):
        ref = rng.randn(8000)
        noise = rng.randn(8000) * 0.1
        pred = ref + noise
        pred_path = tmp_path / f"pred_{i}.wav"
        ref_path = tmp_path / f"ref_{i}.wav"
        _write_wav(pred_path, pred)
        _write_wav(ref_path, ref)
        rows.append((f"utt{i}", str(pred_path), str(ref_path)))
        from sure_eval.evaluation.nodes.scoring.si_sdr.node import _si_sdr

        expected_scores.append(_si_sdr(ref, pred))

    result = score_si_sdr(rows)
    per_sample_scores = [row["si_sdr"] for row in result.details["result"]["per_sample"]]
    mean_score = result.details["result"]["score"]
    expected_mean = sum(per_sample_scores) / len(per_sample_scores)
    assert abs(mean_score - expected_mean) < 1e-6
    assert len(result.details["result"]["per_sample"]) == 3
    assert result.details["result"]["num_samples"] == 3


def test_si_sdr_node_result_fields(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    signal = np.random.RandomState(42).randn(8000)
    pred = tmp_path / "pred.wav"
    ref = tmp_path / "ref.wav"
    _write_wav(pred, signal)
    _write_wav(ref, signal)

    result = score_si_sdr([("utt1", str(pred), str(ref))])
    assert result.stage == "scoring"
    assert result.node_id == "scoring/si_sdr"
    assert result.version == "v1"
    assert result.details["metric"] == "si_sdr"
    assert result.details["keys"] == ["utt1"]
    assert result.internal_stages == ("audio_load", "length_align", "si_sdr_compute", "mean_aggregation")


def test_si_sdri_with_mixture_matches_difference(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    rng = np.random.RandomState(42)
    ref = rng.randn(16000)
    pred = ref + rng.randn(16000) * 0.1
    mixed = rng.randn(16000)

    pred_path = tmp_path / "pred.wav"
    ref_path = tmp_path / "ref.wav"
    mixed_path = tmp_path / "mixed.wav"
    _write_wav(pred_path, pred)
    _write_wav(ref_path, ref)
    _write_wav(mixed_path, mixed)

    pred_only = score_si_sdr([("utt1", str(pred_path), str(ref_path))])
    mixed_only = score_si_sdr([("m", str(mixed_path), str(ref_path))])
    pred_si_sdr = pred_only.details["result"]["per_sample"][0]["si_sdr"]
    mixed_si_sdr = mixed_only.details["result"]["per_sample"][0]["si_sdr"]
    expected_si_sdri = pred_si_sdr - mixed_si_sdr

    result = score_si_sdr(
        [("utt1", str(pred_path), str(ref_path))],
        mixed_paths=[str(mixed_path)],
    )
    per_sample = result.details["result"]["per_sample"][0]
    assert set(per_sample.keys()) == {"si_sdr", "si_sdri"}
    assert per_sample["si_sdr"] == pred_si_sdr
    assert abs(per_sample["si_sdri"] - expected_si_sdri) < 1e-6


def test_si_sdri_infinite_when_prediction_perfect(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    ref = np.random.RandomState(42).randn(16000)
    mixed = np.random.RandomState(7).randn(16000)

    pred_path = tmp_path / "pred.wav"
    ref_path = tmp_path / "ref.wav"
    mixed_path = tmp_path / "mixed.wav"
    _write_wav(pred_path, ref)
    _write_wav(ref_path, ref)
    _write_wav(mixed_path, mixed)

    result = score_si_sdr(
        [("utt1", str(pred_path), str(ref_path))],
        mixed_paths=[str(mixed_path)],
    )
    per_sample = result.details["result"]["per_sample"][0]
    assert math.isinf(per_sample["si_sdr"]) and per_sample["si_sdr"] > 0
    assert math.isinf(per_sample["si_sdri"]) and per_sample["si_sdri"] > 0


def test_si_sdri_zero_when_prediction_and_mixture_both_perfect(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    ref = np.random.RandomState(42).randn(16000)

    pred_path = tmp_path / "pred.wav"
    ref_path = tmp_path / "ref.wav"
    mixed_path = tmp_path / "mixed.wav"
    _write_wav(pred_path, ref)
    _write_wav(ref_path, ref)
    _write_wav(mixed_path, ref)

    result = score_si_sdr(
        [("utt1", str(pred_path), str(ref_path))],
        mixed_paths=[str(mixed_path)],
    )
    per_sample = result.details["result"]["per_sample"][0]
    assert math.isinf(per_sample["si_sdr"]) and per_sample["si_sdr"] > 0
    assert per_sample["si_sdri"] == 0.0


def test_si_sdri_absent_without_mixed_paths(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    rng = np.random.RandomState(42)
    ref = rng.randn(8000)
    pred = ref + rng.randn(8000) * 0.1
    pred_path = tmp_path / "pred.wav"
    ref_path = tmp_path / "ref.wav"
    _write_wav(pred_path, pred)
    _write_wav(ref_path, ref)

    result = score_si_sdr([("utt1", str(pred_path), str(ref_path))])
    assert set(result.details["result"]["per_sample"][0].keys()) == {"si_sdr"}
    assert "si_sdri" not in result.details["result"]


def test_si_sdri_mixed_paths_length_mismatch_raises(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    try:
        score_si_sdr([("utt1", "pred.wav", "ref.wav")], mixed_paths=[])
        assert False, "should have raised"
    except ValueError as exc:
        assert "mixed_paths" in str(exc)


def test_si_sdri_aggregated_mean(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.si_sdr.node import score_si_sdr

    rng = np.random.RandomState(42)
    rows = []
    mixed_paths = []
    for i in range(3):
        ref = rng.randn(8000)
        pred = ref + rng.randn(8000) * 0.1
        mixed = rng.randn(8000)
        pred_path = tmp_path / f"pred_{i}.wav"
        ref_path = tmp_path / f"ref_{i}.wav"
        mixed_path = tmp_path / f"mixed_{i}.wav"
        _write_wav(pred_path, pred)
        _write_wav(ref_path, ref)
        _write_wav(mixed_path, mixed)
        rows.append((f"utt{i}", str(pred_path), str(ref_path)))
        mixed_paths.append(str(mixed_path))

    result = score_si_sdr(rows, mixed_paths=mixed_paths)
    per_sample_si_sdri = [row["si_sdri"] for row in result.details["result"]["per_sample"]]
    expected = sum(per_sample_si_sdri) / len(per_sample_si_sdri)
    assert abs(result.details["result"]["si_sdri"] - expected) < 1e-6