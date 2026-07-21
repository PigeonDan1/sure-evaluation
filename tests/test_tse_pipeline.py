from __future__ import annotations

import numpy as np
from pathlib import Path
import math
import soundfile as sf


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    sf.write(str(path), data.astype("float32"), sr)


def test_tse_si_sdr_route_with_real_audio(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.tse.pipeline import evaluate_tse_samples
    from sure_eval.evaluation.tasks.tse.types import TSESample

    rng = np.random.RandomState(42)
    ref = rng.randn(16000)
    pred = ref.copy()

    pred_path = tmp_path / "pred.wav"
    ref_path = tmp_path / "ref.wav"
    _write_wav(pred_path, pred)
    _write_wav(ref_path, ref)

    report = evaluate_tse_samples(
        [
            TSESample(
                prediction_audio=str(pred_path),
                reference_audio=str(ref_path),
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("si_sdr",),
    )

    assert report.task == "TSE"
    assert report.language == "en"
    assert report.metric == "si_sdr"
    assert report.pipeline_id == "tse.en.si_sdr.si_sdr"
    assert math.isinf(report.score)
    assert "si_sdr" in report.details["results"]


def test_tse_sim_route_with_mock_provider() -> None:
    from sure_eval.evaluation.tasks.tse.pipeline import evaluate_tse_samples
    from sure_eval.evaluation.tasks.tse.types import TSESample

    report = evaluate_tse_samples(
        [
            TSESample(
                prediction_audio="pred.wav",
                reference_audio="ref.wav",
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("sim/wavlm-large",),
        speaker_providers={
            "wavlm-large": lambda prediction, reference, **kwargs: {"ASV": 0.85},
        },
    )

    assert report.task == "TSE"
    assert report.metric == "sim/wavlm-large"
    assert report.pipeline_id == "tse.en.multi.audio_metric_nodes"
    assert report.score == 0.85
    assert report.details["results"]["sim/wavlm-large"]["score"] == 0.85


def test_tse_mos_route_with_mock_provider() -> None:
    from sure_eval.evaluation.tasks.tse.pipeline import evaluate_tse_samples
    from sure_eval.evaluation.tasks.tse.types import TSESample

    report = evaluate_tse_samples(
        [
            TSESample(
                prediction_audio="pred.wav",
                reference_audio="ref.wav",
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("dnsmos",),
        mos_providers={
            "dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.5},
        },
    )

    assert report.task == "TSE"
    assert report.metric == "dnsmos"
    assert report.pipeline_id == "tse.en.multi.audio_metric_nodes"
    assert report.score == 3.5


def test_tse_multi_metric_route(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.tse.pipeline import evaluate_tse_samples
    from sure_eval.evaluation.tasks.tse.types import TSESample

    rng = np.random.RandomState(42)
    ref = rng.randn(8000)
    pred = ref.copy()

    pred_path = tmp_path / "pred.wav"
    ref_path = tmp_path / "ref.wav"
    _write_wav(pred_path, pred)
    _write_wav(ref_path, ref)

    report = evaluate_tse_samples(
        [
            TSESample(
                prediction_audio=str(pred_path),
                reference_audio=str(ref_path),
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("si_sdr", "sim/wavlm-large", "dnsmos"),
        speaker_providers={
            "wavlm-large": lambda prediction, reference, **kwargs: {"ASV": 0.9},
        },
        mos_providers={
            "dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 4.0},
        },
    )

    assert report.task == "TSE"
    assert report.metric == "multi"
    assert report.pipeline_id == "tse.en.multi.audio_metric_nodes"
    assert "si_sdr" in report.details["results"]
    assert "sim/wavlm-large" in report.details["results"]
    assert "dnsmos" in report.details["results"]


def test_tse_unsupported_metric_raises() -> None:
    from sure_eval.evaluation.tasks.tse.pipeline import evaluate_tse_samples
    from sure_eval.evaluation.tasks.tse.types import TSESample

    try:
        evaluate_tse_samples(
            [
                TSESample(
                    prediction_audio="pred.wav",
                    reference_audio="ref.wav",
                    language="en",
                    sample_id="utt1",
                )
            ],
            metrics=("nonexistent_metric",),
        )
        assert False, "should have raised"
    except ValueError as exc:
        assert "nonexistent_metric" in str(exc)


def test_tse_empty_samples_raises() -> None:
    from sure_eval.evaluation.tasks.tse.pipeline import evaluate_tse_samples

    try:
        evaluate_tse_samples([], metrics=("si_sdr",))
        assert False, "should have raised"
    except ValueError as exc:
        assert "at least one" in str(exc)


def test_tse_si_sdri_reported_when_mixed_audio_provided(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.tse.pipeline import evaluate_tse_samples
    from sure_eval.evaluation.tasks.tse.types import TSESample

    rng = np.random.RandomState(42)
    ref = rng.randn(16000)
    pred = ref.copy()
    mixed = rng.randn(16000)

    pred_path = tmp_path / "pred.wav"
    ref_path = tmp_path / "ref.wav"
    mixed_path = tmp_path / "mixed.wav"
    _write_wav(pred_path, pred)
    _write_wav(ref_path, ref)
    _write_wav(mixed_path, mixed)

    report = evaluate_tse_samples(
        [
            TSESample(
                prediction_audio=str(pred_path),
                reference_audio=str(ref_path),
                mixed_audio=str(mixed_path),
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("si_sdr",),
    )

    assert report.metric == "si_sdr"
    si_sdr_block = report.details["results"]["si_sdr"]
    assert "si_sdri" in si_sdr_block
    assert "si_sdri" in report.details["rows"][0]["signal"]["si_sdr"]


def test_tse_si_sdri_absent_without_mixed_audio(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.tse.pipeline import evaluate_tse_samples
    from sure_eval.evaluation.tasks.tse.types import TSESample

    rng = np.random.RandomState(42)
    ref = rng.randn(16000)
    pred = ref.copy()

    pred_path = tmp_path / "pred.wav"
    ref_path = tmp_path / "ref.wav"
    _write_wav(pred_path, pred)
    _write_wav(ref_path, ref)

    report = evaluate_tse_samples(
        [
            TSESample(
                prediction_audio=str(pred_path),
                reference_audio=str(ref_path),
                language="en",
                sample_id="utt1",
            )
        ],
        metrics=("si_sdr",),
    )

    si_sdr_block = report.details["results"]["si_sdr"]
    assert "si_sdri" not in si_sdr_block
    assert "si_sdri" not in report.details["rows"][0]["signal"]["si_sdr"]