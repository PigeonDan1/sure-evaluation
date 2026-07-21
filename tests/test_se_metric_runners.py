from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from sure_eval.cli import app


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _write_synthetic_se_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    clean = 0.4 * np.sin(2.0 * np.pi * 440.0 * time)
    noise = 0.12 * np.sin(2.0 * np.pi * 1900.0 * time)
    noisy = clean + noise
    enhanced = clean + 0.3 * noise
    clean_path = tmp_path / "clean.wav"
    noisy_path = tmp_path / "noisy.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    samples_path = tmp_path / "samples.jsonl"
    _write_wav(clean_path, clean, sample_rate)
    _write_wav(noisy_path, noisy, sample_rate)
    _write_wav(enhanced_path, enhanced, sample_rate)
    samples_path.write_text(
        json.dumps(
            {
                "sample_id": "synthetic",
                "enhanced_audio": "enhanced.wav",
                "noisy_audio": "noisy.wav",
                "reference_audio": "clean.wav",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return clean_path, noisy_path, enhanced_path, samples_path


def test_se_pipeline_cli_runs_with_stub_backend(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_se_metric_pipeline.py",
            "--enhanced-audio",
            "enhanced.wav",
            "--noisy-audio",
            "noisy.wav",
            "--reference-audio",
            "clean.wav",
            "--metrics",
            "si-sdr,stoi,pesq,dnsmos,wv-mos,utmos",
            "--stub",
            "--output",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["report"]["pipeline_id"] == "se.multi.enhancement_quality_nodes"
    assert report["metrics"]["si-sdr"]["score"] == 8.0
    assert report["metrics"]["stoi"]["score"] == 0.91
    assert report["metrics"]["pesq"]["score"] == 2.8
    assert report["metrics"]["dnsmos"]["score"] == 3.1


def test_se_script_describe_and_run_with_injected_providers(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts.se import describe_pipeline, run
    from sure_eval.evaluation.tasks.se.types import SESample

    description = describe_pipeline(metrics=("si-sdr", "dnsmos"))
    assert description.task == "SE"
    assert description.pipeline_id == "se.multi.enhancement_quality_nodes"
    assert description.node_ids == ("scoring/si_sdr", "scoring/dnsmos")
    assert description.required_roles == ("enhanced_audio", "reference_audio")

    report = run(
        [
            SESample(
                enhanced_audio="enhanced.wav",
                noisy_audio="noisy.wav",
                reference_audio="clean.wav",
                sample_id="utt1",
            )
        ],
        metrics=("si-sdr", "dnsmos"),
        reference_providers={"si-sdr": lambda prediction, reference, **kwargs: {"si_sdr": 7.5}},
        mos_providers={"dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.0}},
        output_dir=str(tmp_path / "se_out"),
    )

    assert report.pipeline_id == "se.multi.enhancement_quality_nodes"
    assert (tmp_path / "se_out" / "report.json").exists()
    assert (tmp_path / "se_out" / "pipeline_description.json").exists()


def test_sure_eval_metric_describe_run_se_si_sdr(tmp_path: Path) -> None:
    _clean, _noisy, _enhanced, samples_path = _write_synthetic_se_files(tmp_path)
    pipeline_path = tmp_path / "se_pipeline.json"
    output_dir = tmp_path / "se_eval"
    runner = CliRunner()

    describe = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "se",
            "--metrics",
            "si-sdr",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe.exit_code == 0, describe.stdout
    payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert payload["pipeline_id"] == "se.si_sdr.si_sdr"
    assert payload["required_roles"] == ["samples_jsonl"]

    run = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--samples-jsonl",
            str(samples_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--json",
        ],
    )

    assert run.exit_code == 0, run.stdout
    summary = json.loads(run.stdout)
    assert summary["task"] == "SE"
    assert summary["metric"] == "si-sdr"
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["pipeline_id"] == "se.si_sdr.si_sdr"
    assert report["details"]["results"]["si-sdr"]["score"] == report["score"]


def test_metric_describe_run_se_default_metrics_with_injected_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    import sure_eval.evaluation.audio_runtime as audio_runtime
    from sure_eval.evaluation.cli_adapters import build_pipeline_spec, run_pipeline_spec

    expected_metrics = ["si-sdr", "stoi", "pesq", "dnsmos", "wv-mos", "utmos"]
    enhanced = tmp_path / "enhanced.wav"
    reference = tmp_path / "reference.wav"
    samples_jsonl = tmp_path / "samples.jsonl"
    enhanced.write_bytes(b"fake")
    reference.write_bytes(b"fake")
    samples_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "utt1",
                "enhanced_audio": "enhanced.wav",
                "reference_audio": "reference.wav",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_build_se_runtime(*, metrics, device="cuda", cache_dir=None):
        assert list(metrics) == expected_metrics
        return {
            "reference_providers": {
                "si-sdr": lambda prediction, reference, **kwargs: {"si_sdr": 8.0},
                "stoi": lambda prediction, reference, **kwargs: {"stoi": 0.91},
                "pesq": lambda prediction, reference, **kwargs: {"pesq": 2.8},
            },
            "mos_providers": {
                "dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.1},
                "wv-mos": lambda prediction, reference="", **kwargs: {"mos": 3.4},
                "utmos": lambda prediction, reference="", **kwargs: {"utmos": 3.2},
            },
        }

    monkeypatch.setattr(audio_runtime, "build_se_runtime", fake_build_se_runtime)

    pipeline = build_pipeline_spec("se")
    assert pipeline["metric"] == "multi"
    assert pipeline["metrics"] == expected_metrics

    summary = run_pipeline_spec(
        pipeline,
        samples_jsonl=str(samples_jsonl),
        output_dir=str(tmp_path / "out"),
        device="cpu",
    )

    assert summary["task"] == "SE"
    assert summary["metric"] == "multi"
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert list(report["details"]["results"]) == expected_metrics
