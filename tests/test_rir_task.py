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


def _write_mono_rir(path: Path, scale: float = 1.0) -> None:
    signal = np.zeros(256, dtype=np.float64)
    signal[8] = scale
    signal[40:] = scale * np.exp(-np.arange(216) / 40.0)
    sf.write(path, signal, 16000)


def _write_delayed_binaural_rir(path: Path, right_delay: int) -> None:
    signal = np.zeros((256, 2), dtype=np.float64)
    signal[20, 0] = 1.0
    signal[20 + right_delay, 1] = 0.8
    signal[40:, 0] += 0.01 * np.exp(-np.arange(216) / 40.0)
    signal[40:, 1] += 0.008 * np.exp(-np.arange(216) / 40.0)
    sf.write(path, signal, 16000)


def test_rir_pipeline_describes_registered_route() -> None:
    description = describe_pipeline("rir", metric="rir_npm")
    assert description.pipeline_id == "rir.any.rir_npm.rir_npm_v1"
    assert description.required_roles == ("prediction_rir", "reference_rir")

    npm = describe_pipeline("rir", metric="npm")
    assert npm.pipeline_id == "rir.any.rir_npm.rir_npm_v1"
    assert npm.metric == "rir_npm"

    default = describe_pipeline("rir")
    assert default.pipeline_id == "rir.any.multi.rir_npm.rir_npm_v1__rt60_mae.rt60_mae_v1"

    base_bundle = describe_pipeline(
        "rir",
        metrics=("rir_npm", "rt60_mae", "rir_drr_mae", "rir_c50_mae", "rir_c80_mae"),
    )
    assert base_bundle.pipeline_kind == "bundle"
    assert base_bundle.execution_metrics == (
        "rir_npm", "rt60_mae", "rir_drr_mae", "rir_c50_mae", "rir_c80_mae"
    )

    iacc = describe_pipeline("rir", metric="iacc")
    assert iacc.pipeline_id == "rir.any.rir_iacc_mae.rir_iacc_mae_v1"
    assert iacc.metric == "rir_iacc_mae"


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

    description = describe_pipeline("rir", metric="rir_npm")
    report = run_task(
        "rir",
        samples=[sample],
        metrics=("rir_npm",),
        output_dir=str(tmp_path / "script_run"),
    )

    assert description.pipeline_id == report.pipeline_id
    assert json.loads((tmp_path / "script_run" / "report.json").read_text())["pipeline_id"] == (
        "rir.any.rir_npm.rir_npm_v1"
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

    routes = runner.invoke(app, ["metric", "routes", "rir", "--metric", "rir_npm", "--json"])
    assert routes.exit_code == 0, routes.stdout
    routes_payload = json.loads(routes.stdout)
    assert routes_payload["default_pipeline_id"] == "rir.any.rir_npm.rir_npm_v1"

    described = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "rir",
            "--metric",
            "rir_npm",
            "--output",
            str(pipeline_json),
            "--json",
        ],
    )
    assert described.exit_code == 0, described.stdout
    assert json.loads(pipeline_json.read_text())["pipeline_id"] == (
        "rir.any.rir_npm.rir_npm_v1"
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
    assert json.loads(run.stdout)["pipeline_id"] == "rir.any.rir_npm.rir_npm_v1"
    report = json.loads((tmp_path / "cli_run" / "report.json").read_text())
    description = json.loads(
        (tmp_path / "cli_run" / "pipeline_description.json").read_text()
    )
    assert report["pipeline_id"] == description["pipeline_id"]
    assert report["details"]["results"]["rir_npm"]["score"] < 0


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
    report = evaluate_rir_samples([sample], metrics=("rir_npm", "rir_drr_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"))
    assert report.pipeline_kind == "bundle"
    assert report.score is not None
    assert set(report.details["results"]) == {"rir_npm", "rir_drr_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"}


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
    result = evaluate_rir_samples([sample], metrics=("rir_npm",))
    assert result.details["results"]["rir_npm"]["score"] < 0


def test_rir_npm_is_scale_invariant_and_reports_db(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    prediction = tmp_path / "prediction.wav"
    _write_rir(reference)
    _write_rir(prediction, 0.9)
    sample = RIRSample(
        prediction_rir=str(prediction),
        reference_rir=str(reference),
        sample_id="rir-npm-001",
        sample_rate=16000,
    )
    report = evaluate_rir_samples([sample], metrics=("npm",))
    result = report.details["results"]["rir_npm"]
    assert result["score"] < -80
    assert result["unit"] == "dB"


def test_rir_channel_branches_and_binaural_metrics(tmp_path: Path) -> None:
    mono_reference = tmp_path / "mono_reference.wav"
    mono_prediction = tmp_path / "mono_prediction.wav"
    _write_mono_rir(mono_reference)
    _write_mono_rir(mono_prediction, 0.9)
    mono = RIRSample(
        prediction_rir=str(mono_prediction),
        reference_rir=str(mono_reference),
        sample_id="rir-mono-001",
        channel_layout="mono",
    )
    single_report = evaluate_rir_samples([mono], metrics=("rir_npm",), channel_mode="single")
    assert single_report.details["channel_mode"] == "single"
    try:
        evaluate_rir_samples([mono], metrics=("rir_npm",), channel_mode="multi")
    except ValueError as exc:
        assert "multi-channel" in str(exc)
    else:
        raise AssertionError("single-channel RIR was accepted by the multi-channel chain")

    reference = tmp_path / "binaural_reference.wav"
    prediction = tmp_path / "binaural_prediction.wav"
    _write_delayed_binaural_rir(reference, right_delay=3)
    _write_delayed_binaural_rir(prediction, right_delay=5)
    binaural = RIRSample(
        prediction_rir=str(prediction),
        reference_rir=str(reference),
        sample_id="rir-binaural-001",
        channel_layout="binaural_lr",
        metadata={"microphone_spacing_m": 0.18},
    )
    report = evaluate_rir_samples(
        [binaural],
        metrics=("iacc", "itd", "rir_ild_mae", "rir_ipd_mae"),
        channel_mode="multi",
    )
    assert set(report.details["results"]) == {
        "rir_iacc_mae",
        "rir_itd_mae",
        "rir_ild_mae",
        "rir_ipd_mae",
    }
    assert report.details["results"]["rir_iacc_mae"]["per_sample"][0]["prediction_iacc"] >= 0
    assert report.details["results"]["rir_itd_mae"]["score"] > 0
    assert report.details["results"]["rir_itd_mae"]["unit"] == "ms"


def test_binaural_metric_rejects_three_channel_rir(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npy"
    prediction = tmp_path / "prediction.npy"
    signal = np.zeros((3, 64), dtype=np.float64)
    signal[:, 4] = 1.0
    np.save(reference, signal)
    np.save(prediction, signal)
    sample = RIRSample(
        prediction_rir=str(prediction),
        reference_rir=str(reference),
        sample_id="rir-three-channel-001",
        sample_rate=16000,
        channel_layout="multi_channel",
        metadata={"microphone_spacing_m": 0.08},
    )
    try:
        evaluate_rir_samples([sample], metrics=("rir_iacc_mae",), channel_mode="multi")
    except ValueError as exc:
        assert "exactly two" in str(exc)
    else:
        raise AssertionError("IACC accepted a non-binaural multi-channel RIR")
    spatial_report = evaluate_rir_samples(
        [sample],
        metrics=("rir_itd_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"),
        channel_mode="multi",
    )
    for result in spatial_report.details["results"].values():
        row = result["per_sample"][0]
        assert row["pair_count"] == 2
        assert len(row["pairs"]) == 2


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
        "rir_npm",
        "rt60_mae",
        "rir_drr_mae",
        "rir_c50_mae",
        "rir_c80_mae",
        "rir_iacc_mae",
        "rir_itd_mae",
        "rir_ild_mae",
        "rir_ipd_mae",
        "rir_doa_mae",
    ):
        result = checker.check_node(f"scoring/{metric}")
        assert result.status == "ok"
        assert result.runtime == "in_process"
        assert result.required is False
