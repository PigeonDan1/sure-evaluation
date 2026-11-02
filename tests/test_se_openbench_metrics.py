"""Route identity and score-definition checks for opt-in SE metrics."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from sure_eval.cli import app
from sure_eval.evaluation.cli_adapters import build_pipeline_spec, list_metric_routes
from sure_eval.evaluation.nodes.scoring._score_components import ScoreComponentProvider
from sure_eval.evaluation.nodes.scoring._full_reference_audio import (
    ESTOIProvider, LSDProvider,
)
from sure_eval.evaluation.nodes.scoring.lps.node import LPSProvider, levenshtein


NEW_METRICS = (
    "estoi", "lsd", "nsrmr", "lps", "dnsmos_p808", "dnsmos_sig",
    "dnsmos_bak", "squim_stoi", "squim_pesq", "squim_si_sdr",
    "nisqa_mos", "nisqa_noi", "nisqa_dis", "nisqa_col", "nisqa_loud",
)


def _wav(path: Path, signal: np.ndarray, sample_rate: int = 16000) -> None:
    samples = (np.clip(signal, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def test_new_atomic_routes_and_original_default_bundle() -> None:
    for metric in NEW_METRICS:
        routes = list_metric_routes("se", metric=metric)
        assert routes["count"] == 1
        route = routes["routes"][0]
        spec = build_pipeline_spec("se", pipeline_id=route["pipeline_id"])
        assert spec["pipeline_id"] == route["pipeline_id"]
        assert spec["metric"] == metric
        assert spec["computation_node_ids"] == route["computation_node_ids"]
        assert route["default"] is True
    default = build_pipeline_spec("se")
    assert default["metrics"] == ["si-sdr", "stoi", "pesq", "dnsmos", "wv-mos", "utmos"]
    assert default["pipeline_id"].endswith("utmos.utmos_v1")


def test_estoi_and_lsd_match_local_score_definitions(tmp_path: Path) -> None:
    pystoi = pytest.importorskip("pystoi")
    librosa = pytest.importorskip("librosa")
    time = np.arange(32000, dtype=np.float32) / 16000
    clean = 0.3 * np.sin(2 * np.pi * 440 * time) + 0.1 * np.sin(2 * np.pi * 530 * time)
    enhanced = clean + 0.05 * np.sin(2 * np.pi * 1700 * time)
    clean_path, enhanced_path = tmp_path / "clean.wav", tmp_path / "enhanced.wav"
    _wav(clean_path, clean)
    _wav(enhanced_path, enhanced)
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import read_audio_mono

    reference, _ = read_audio_mono(str(clean_path))
    estimate, _ = read_audio_mono(str(enhanced_path))
    expected_estoi = float(pystoi.stoi(reference, estimate, 16000, extended=True))
    assert ESTOIProvider()(str(enhanced_path), str(clean_path))["estoi"] == pytest.approx(expected_estoi)
    est_db = 20 * np.log10(np.maximum(np.abs(librosa.stft(estimate, n_fft=512, hop_length=128, win_length=512)), 1e-7))
    ref_db = 20 * np.log10(np.maximum(np.abs(librosa.stft(reference, n_fft=512, hop_length=128, win_length=512)), 1e-7))
    expected_lsd = float(np.sqrt(np.mean((est_db - ref_db) ** 2)))
    assert LSDProvider()(str(enhanced_path), str(clean_path))["lsd"] == pytest.approx(expected_lsd)
    assert LSDProvider()(str(clean_path), str(clean_path))["lsd"] == pytest.approx(0.0)


def test_nsrmr_matches_normalized_srmrpy_mode(tmp_path: Path) -> None:
    srmrpy = pytest.importorskip("srmrpy")
    from sure_eval.evaluation.nodes.scoring.nsrmr.node import NSRMRProvider
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import read_audio_mono

    time = np.arange(64000, dtype=np.float32) / 16000
    path = tmp_path / "speech.wav"
    _wav(path, 0.25 * np.sin(2 * np.pi * (140 * time + 60 * time * time)))
    waveform, _ = read_audio_mono(str(path))
    expected = float(srmrpy.srmr(waveform, 16000, fast=True, norm=True)[0])
    result = NSRMRProvider()(str(path))
    assert result["score"] == pytest.approx(expected)
    assert result["unit"] == "ratio"


def test_lsd_resamples_non_16k_audio_like_local_leaderboard(tmp_path: Path) -> None:
    librosa = pytest.importorskip("librosa")
    from sure_eval.evaluation.nodes.scoring._full_reference_audio import read_audio_mono

    time = np.arange(48000, dtype=np.float32) / 48000
    clean_path, enhanced_path = tmp_path / "clean48.wav", tmp_path / "enhanced48.wav"
    _wav(clean_path, 0.25 * np.sin(2 * np.pi * 400 * time), 48000)
    _wav(enhanced_path, 0.25 * np.sin(2 * np.pi * 400 * time) + 0.03 * np.sin(2 * np.pi * 1800 * time), 48000)
    clean, rate = read_audio_mono(str(clean_path))
    enhanced, _ = read_audio_mono(str(enhanced_path))
    clean = librosa.resample(clean, orig_sr=rate, target_sr=16000)
    enhanced = librosa.resample(enhanced, orig_sr=rate, target_sr=16000)
    spec_clean = np.abs(librosa.stft(clean, n_fft=512, hop_length=128, win_length=512))
    spec_enhanced = np.abs(librosa.stft(enhanced, n_fft=512, hop_length=128, win_length=512))
    expected = float(np.sqrt(np.mean((20 * np.log10(np.maximum(spec_clean, 1e-7)) - 20 * np.log10(np.maximum(spec_enhanced, 1e-7))) ** 2)))
    assert LSDProvider()(str(enhanced_path), str(clean_path))["lsd"] == pytest.approx(expected)


def test_component_selection_rejects_missing_and_nonfinite_fields() -> None:
    rows = [("a", "a.wav"), ("b", "b.wav")]

    class Provider:
        def score_batch(self, values, *, metric_name):
            assert values == rows
            return [{"P808_MOS": 3.0}, {"P808_MOS": 4.0}]

    from sure_eval.evaluation.nodes.scoring.dnsmos_p808 import score_dnsmos_p808

    result = score_dnsmos_p808(rows, provider=Provider())
    assert result.details["result"]["score"] == 3.5
    assert result.details["result"]["num_samples"] == 2
    with pytest.raises(KeyError, match="SIG"):
        ScoreComponentProvider(lambda *_: {"BAK": 3}, field="SIG")("a.wav")
    with pytest.raises(ValueError, match="non-finite"):
        ScoreComponentProvider(lambda *_: {"SIG": math.nan}, field="SIG")("a.wav")


def test_lps_denominator_and_empty_reference(monkeypatch) -> None:
    assert levenshtein("abc", "adc") == 1
    assert levenshtein("", "abc") == 3
    provider = LPSProvider(model_dir="unused")
    monkeypatch.setattr(provider, "_wave", lambda path: np.array([0.1], dtype=np.float32))
    phonemes = iter(("ab", "abcd"))
    monkeypatch.setattr(provider, "_phonemes", lambda wave: next(phonemes))
    assert provider("enhanced.wav", "clean.wav")["lps"] == 0.5
    phonemes = iter(("ab", ""))
    monkeypatch.setattr(provider, "_phonemes", lambda wave: next(phonemes))
    with pytest.raises(ValueError, match="empty reference"):
        provider("enhanced.wav", "clean.wav")


def test_new_routes_require_reference_only_when_needed(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_se_samples_jsonl

    enhanced = tmp_path / "enhanced.wav"
    enhanced.write_bytes(b"test")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"sample_id": "a", "enhanced_audio": "enhanced.wav"}) + "\n")
    for metric in ("estoi", "lsd", "lps"):
        with pytest.raises(SampleJsonlError, match="reference_audio is required"):
            load_se_samples_jsonl(samples, metrics=(metric,))
    assert len(load_se_samples_jsonl(samples, metrics=("nisqa_mos", "squim_stoi", "nsrmr"))) == 1


def test_real_runtime_selects_shared_model_providers_without_loading_assets() -> None:
    from sure_eval.evaluation.audio_runtime import build_se_runtime

    runtime = build_se_runtime(
        metrics=("estoi", "lsd", "lps", "dnsmos_p808", "dnsmos_sig",
                 "squim_pesq", "squim_stoi", "nisqa_mos", "nisqa_noi", "nsrmr"),
        device="cpu",
    )
    assert set(runtime["reference_providers"]) == {"estoi", "lsd", "lps"}
    mos = runtime["mos_providers"]
    assert set(mos) >= {"dnsmos_p808", "dnsmos_sig", "squim_pesq", "squim_stoi", "nisqa_mos", "nisqa_noi", "nsrmr"}
    assert mos["dnsmos_p808"] is mos["dnsmos_sig"]
    assert mos["squim_pesq"] is mos["squim_stoi"]
    assert mos["nisqa_mos"] is mos["nisqa_noi"]


def test_cli_describe_run_report_preserves_identity(monkeypatch, tmp_path: Path) -> None:
    import sure_eval.evaluation.audio_runtime as runtime

    enhanced, reference = tmp_path / "enhanced.wav", tmp_path / "reference.wav"
    enhanced.write_bytes(b"provider fixture")
    reference.write_bytes(b"provider fixture")
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps({"sample_id": "a", "enhanced_audio": "enhanced.wav", "reference_audio": "reference.wav"}) + "\n")

    def providers(*, metrics, device="cuda", cache_dir=None):
        assert tuple(metrics) == ("estoi", "dnsmos_p808")
        return {
            "reference_providers": {"estoi": lambda *_: {"estoi": 0.83}},
            "mos_providers": {"dnsmos_p808": lambda *_: {"P808_MOS": 3.5}},
        }

    monkeypatch.setattr(runtime, "build_se_runtime", providers)
    runner = CliRunner()
    routes = runner.invoke(app, ["metric", "routes", "se", "--metric", "dnsmos_p808", "--json"])
    assert routes.exit_code == 0, routes.stdout
    assert json.loads(routes.stdout)["routes"][0]["pipeline_id"] == "se.any.dnsmos_p808.dnsmos_p808_v1"
    pipeline = tmp_path / "pipeline.json"
    describe = runner.invoke(app, ["metric", "describe", "se", "--metrics", "estoi,dnsmos_p808", "--output", str(pipeline), "--json"])
    assert describe.exit_code == 0, describe.stdout
    spec = json.loads(pipeline.read_text())
    expected = "se.any.multi.estoi.estoi_v1__dnsmos_p808.dnsmos_p808_v1"
    assert spec["pipeline_id"] == expected
    output = tmp_path / "out"
    run = runner.invoke(app, ["metric", "run", "--pipeline", str(pipeline), "--samples-jsonl", str(samples), "--output-dir", str(output), "--device", "cpu", "--json"])
    assert run.exit_code == 0, run.stdout
    report = json.loads((output / "report.json").read_text())
    description = json.loads((output / "pipeline_description.json").read_text())
    assert report["pipeline_id"] == description["pipeline_id"] == expected
    assert report["details"]["results"]["estoi"]["score"] == 0.83
    assert report["details"]["results"]["dnsmos_p808"]["score"] == 3.5
    assert report["details"]["results"]["estoi"]["higher_is_better"] is True
    assert report["details"]["results"]["dnsmos_p808"]["unit"] == "predicted_mos"
    assert report["computation_node_ids"] == description["computation_node_ids"]


def test_script_stub_reports_new_atomic_identity(tmp_path: Path) -> None:
    output = tmp_path / "stub.json"
    subprocess.run([
        sys.executable, "scripts/run_se_metric_pipeline.py", "--enhanced-audio", "enhanced.wav",
        "--reference-audio", "clean.wav", "--metrics", "lsd,nsrmr", "--stub",
        "--output", str(output),
    ], check=True)
    payload = json.loads(output.read_text())
    assert payload["report"]["pipeline_id"] == "se.any.multi.lsd.lsd_v1__nsrmr.nsrmr_v1"
    assert payload["metrics"]["lsd"]["score"] == 5.2
    assert payload["metrics"]["nsrmr"]["score"] == 4.1
    assert payload["metrics"]["lsd"]["higher_is_better"] is False
    assert payload["metrics"]["nsrmr"]["unit"] == "ratio"
