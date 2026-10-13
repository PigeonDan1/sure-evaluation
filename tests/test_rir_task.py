from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from sure_eval.cli import app
from sure_eval.evaluation.scripts import describe_pipeline
from sure_eval.evaluation.scripts import run_task
from sure_eval.evaluation.tasks.rir.pipeline import evaluate_rir_samples
from sure_eval.evaluation.tasks.rir.types import RIRSample


def _write_rir(path: Path, scale: float = 1.0) -> None:
    signal = np.zeros((256, 2), dtype=np.float64)
    signal[8, 0] = scale
    signal[8, 1] = scale * 0.8
    signal[40:, 0] = scale * np.exp(-np.arange(216) / 40.0)
    signal[40:, 1] = scale * 0.8 * np.exp(-np.arange(216) / 40.0)
    sf.write(path, signal, 16000)


def test_rir_pipeline_describes_registered_route() -> None:
    description = describe_pipeline("rir", metric="rir_nmse")
    assert description.pipeline_id == "rir.any.rir_nmse.rir_nmse_v1"
    assert description.required_roles == ("prediction_rir", "reference_rir")


def test_rir_script_describe_and_run_writes_report(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    prediction = tmp_path / "prediction.wav"
    _write_rir(reference)
    _write_rir(prediction, 0.9)
    sample = RIRSample(
        prediction_rir=str(prediction),
        reference_rir=str(reference),
        sample_id="rir-001",
        sample_rate=16000,
    )

    description = describe_pipeline("rir", metric="rir_nmse")
    report = run_task(
        "rir",
        samples=[sample],
        metrics=("rir_nmse",),
        output_dir=str(tmp_path / "script_run"),
    )

    assert description.pipeline_id == report.pipeline_id
    assert json.loads((tmp_path / "script_run" / "report.json").read_text())["pipeline_id"] == (
        "rir.any.rir_nmse.rir_nmse_v1"
    )


def test_rir_cli_routes_describe_and_run_preserve_pipeline_id(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    prediction = tmp_path / "prediction.wav"
    _write_rir(reference)
    _write_rir(prediction, 0.9)
    samples_jsonl = tmp_path / "samples.jsonl"
    samples_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "rir-001",
                "prediction_rir": prediction.name,
                "reference_rir": reference.name,
                "sample_rate": 16000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pipeline_json = tmp_path / "pipeline.json"
    runner = CliRunner()

    routes = runner.invoke(app, ["metric", "routes", "rir", "--metric", "rir_nmse", "--json"])
    assert routes.exit_code == 0, routes.stdout
    routes_payload = json.loads(routes.stdout)
    assert routes_payload["default_pipeline_id"] == "rir.any.rir_nmse.rir_nmse_v1"

    described = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "rir",
            "--metric",
            "rir_nmse",
            "--output",
            str(pipeline_json),
            "--json",
        ],
    )
    assert described.exit_code == 0, described.stdout
    assert json.loads(pipeline_json.read_text())["pipeline_id"] == (
        "rir.any.rir_nmse.rir_nmse_v1"
    )

    run = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_json),
            "--samples-jsonl",
            str(samples_jsonl),
            "--output-dir",
            str(tmp_path / "cli_run"),
            "--validate-env",
            "--json",
        ],
    )
    assert run.exit_code == 0, run.stdout
    assert json.loads(run.stdout)["pipeline_id"] == "rir.any.rir_nmse.rir_nmse_v1"
    report = json.loads((tmp_path / "cli_run" / "report.json").read_text())
    description = json.loads(
        (tmp_path / "cli_run" / "pipeline_description.json").read_text()
    )
    assert report["pipeline_id"] == description["pipeline_id"]
    assert report["details"]["results"]["rir_nmse"]["score"] > 0


def test_rir_metrics_run_on_small_fixture(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    prediction = tmp_path / "prediction.wav"
    _write_rir(reference)
    _write_rir(prediction, 0.9)
    sample = RIRSample(
        prediction_rir=str(prediction),
        reference_rir=str(reference),
        sample_id="rir-001",
        sample_rate=16000,
        channel_layout="binaural_lr",
        metadata={"microphone_spacing_m": 0.18},
    )
    report = evaluate_rir_samples([sample], metrics=("rir_nmse", "rir_corr", "rir_drr_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"))
    assert report.pipeline_kind == "bundle"
    assert report.score is not None
    assert set(report.details["results"]) == {"rir_nmse", "rir_corr", "rir_drr_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"}


def test_rir_metrics_accept_reference_npy_in_channel_first_layout(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.wav"
    reference = tmp_path / "reference.npy"
    signal = np.zeros((2, 256), dtype=np.float64)
    signal[0, 8] = 1.0
    signal[1, 8] = 0.8
    np.save(reference, signal)
    sf.write(prediction, signal.T * 0.9, 16000)
    sample = RIRSample(
        prediction_rir=str(prediction),
        reference_rir=str(reference),
        sample_id="rir-npy-001",
        sample_rate=16000,
    )
    result = evaluate_rir_samples([sample], metrics=("rir_nmse",))
    assert result.details["results"]["rir_nmse"]["score"] > 0


def test_rir_samples_jsonl_contract_rejects_missing_reference(tmp_path: Path) -> None:
    fixture = tmp_path / "samples.jsonl"
    fixture.write_text(json.dumps({"sample_id": "rir-001", "prediction_rir": "prediction.wav"}) + "\n", encoding="utf-8")
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_rir_samples_jsonl

    try:
        load_rir_samples_jsonl(fixture)
    except SampleJsonlError as exc:
        assert "reference_rir is required" in str(exc)
    else:
        raise AssertionError("missing reference_rir was accepted")


def test_rir_samples_jsonl_contract_rejects_malformed_metadata(tmp_path: Path) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_rir_samples_jsonl

    prediction = tmp_path / "prediction.wav"
    reference = tmp_path / "reference.wav"
    _write_rir(prediction)
    _write_rir(reference)
    fixture = tmp_path / "samples.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "sample_id": "rir-001",
                "prediction_rir": prediction.name,
                "reference_rir": reference.name,
                "metadata": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_rir_samples_jsonl(fixture)
    except SampleJsonlError as exc:
        assert "metadata must be an object" in str(exc)
    else:
        raise AssertionError("malformed metadata was accepted")


def test_rir_nodes_are_in_process_environment_checks() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    checker = NodeEnvChecker()
    for metric in (
        "rir_nmse",
        "rir_corr",
        "rt60_mae",
        "rir_drr_mae",
        "rir_c50_mae",
        "rir_c80_mae",
        "rir_ild_mae",
        "rir_ipd_mae",
        "rir_doa_mae",
    ):
        result = checker.check_node(f"scoring/{metric}")
        assert result.status == "ok"
        assert result.runtime == "in_process"
        assert result.required is False
