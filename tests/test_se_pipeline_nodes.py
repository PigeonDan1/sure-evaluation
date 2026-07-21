from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_se_task_route_scores_full_reference_and_mos_nodes() -> None:
    from sure_eval.evaluation.tasks.se.pipeline import evaluate_se_samples
    from sure_eval.evaluation.tasks.se.types import SESample

    report = evaluate_se_samples(
        [
            SESample(
                enhanced_audio="enhanced.wav",
                noisy_audio="noisy.wav",
                reference_audio="clean.wav",
                sample_id="utt1",
            )
        ],
        metrics=("si-sdr", "stoi", "pesq", "dnsmos"),
        reference_providers={
            "si-sdr": lambda prediction, reference, **kwargs: {"si_sdr": 9.0},
            "stoi": lambda prediction, reference, **kwargs: {"stoi": 0.92},
            "pesq": lambda prediction, reference, **kwargs: {"pesq": 2.7},
        },
        mos_providers={"dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.4, "SIG": 3.6}},
    )

    assert report.task == "SE"
    assert report.language == "n/a"
    assert report.metric == "multi"
    assert report.score == 9.0
    assert report.pipeline_id == "se.multi.enhancement_quality_nodes"
    assert [node.node_id for node in report.pipeline_trace] == [
        "scoring/si_sdr",
        "scoring/stoi",
        "scoring/pesq",
        "scoring/dnsmos",
    ]
    assert report.input_contract is not None
    assert report.input_contract.required_roles == ("enhanced_audio", "reference_audio")
    assert report.details["results"]["stoi"]["score"] == 0.92
    assert report.details["results"]["pesq"]["score"] == 2.7
    assert report.details["results"]["dnsmos"]["score"] == 3.4
    assert report.details["rows"][0]["full_reference"]["si-sdr"]["si_sdr"] == 9.0
    assert report.details["rows"][0]["mos"]["dnsmos"]["OVRL"] == 3.4


def test_se_si_sdr_scores_generated_enhanced_audio(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import calculate_si_sdr, read_audio_mono
    from sure_eval.evaluation.tasks.se.pipeline import evaluate_se_samples
    from sure_eval.evaluation.tasks.se.types import SESample

    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    clean = 0.4 * np.sin(2.0 * np.pi * 440.0 * time)
    noise = 0.15 * np.sin(2.0 * np.pi * 2100.0 * time)
    noisy = clean + noise
    enhanced = clean + 0.25 * noise

    clean_path = tmp_path / "clean.wav"
    noisy_path = tmp_path / "noisy.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    _write_wav(clean_path, clean, sample_rate)
    _write_wav(noisy_path, noisy, sample_rate)
    _write_wav(enhanced_path, enhanced, sample_rate)

    report = evaluate_se_samples(
        [
            SESample(
                enhanced_audio=str(enhanced_path),
                noisy_audio=str(noisy_path),
                reference_audio=str(clean_path),
                sample_id="synthetic",
            )
        ],
        metrics=("si-sdr",),
    )

    clean_audio, _ = read_audio_mono(str(clean_path))
    noisy_audio, _ = read_audio_mono(str(noisy_path))
    enhanced_audio, _ = read_audio_mono(str(enhanced_path))
    assert report.pipeline_id == "se.si_sdr.si_sdr"
    assert report.details["results"]["si-sdr"]["score"] == report.score
    assert calculate_si_sdr(clean_audio, enhanced_audio) > calculate_si_sdr(clean_audio, noisy_audio)


def test_se_full_reference_metrics_require_reference_for_every_sample() -> None:
    from sure_eval.evaluation.tasks.se.pipeline import evaluate_se_samples
    from sure_eval.evaluation.tasks.se.types import SESample

    samples = [
        SESample(enhanced_audio="enhanced-a.wav", reference_audio="clean-a.wav", sample_id="utt-a"),
        SESample(enhanced_audio="enhanced-b.wav", sample_id="utt-b"),
    ]

    try:
        evaluate_se_samples(
            samples,
            metrics=("si-sdr",),
            reference_providers={"si-sdr": lambda prediction, reference, **kwargs: {"si_sdr": 1.0}},
        )
    except ValueError as exc:
        assert "requires reference_audio for every sample" in str(exc)
        assert "utt-b" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_se_samples_jsonl_loader_validates_full_reference_roles(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_se_samples_jsonl

    enhanced = tmp_path / "enhanced.wav"
    enhanced.write_bytes(b"fake")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps({"sample_id": "utt1", "enhanced_audio": "enhanced.wav"}) + "\n",
        encoding="utf-8",
    )

    try:
        load_se_samples_jsonl(samples, metrics=("si-sdr",))
    except SampleJsonlError as exc:
        assert "reference_audio is required" in str(exc)
    else:
        raise AssertionError("expected SampleJsonlError")

    loaded = load_se_samples_jsonl(samples, metrics=("dnsmos",))
    assert loaded[0].enhanced_audio == str(enhanced.resolve())


def test_se_metric_definitions_are_registry_metrics() -> None:
    from sure_eval.evaluation.registry import MetricRegistry
    from sure_eval.evaluation.tasks.se.metrics import PESQMetric, SISDRMetric, STOIMetric

    assert isinstance(MetricRegistry.get_metric("si-sdr"), SISDRMetric)
    assert isinstance(MetricRegistry.get_metric("sisdr"), SISDRMetric)
    assert isinstance(MetricRegistry.get_metric("stoi"), STOIMetric)
    assert isinstance(MetricRegistry.get_metric("pesq"), PESQMetric)
