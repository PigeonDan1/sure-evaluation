from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from sure_eval.cli import app
from sure_eval.evaluation.scripts import describe_pipeline
from sure_eval.evaluation.scripts import run_task
from sure_eval.evaluation.tasks.rir.pipeline import METRICS, evaluate_rir_samples
from sure_eval.evaluation.tasks.rir.types import RIRSample


def _write_rir(path: Path, scale: float = 1.0) -> None:
    signal = np.zeros((256, 2), dtype=np.float64)
    signal[8, 0] = scale
    signal[8, 1] = scale * 0.8
    signal[40:, 0] = scale * np.exp(-np.arange(216) / 40.0)
    signal[40:, 1] = scale * 0.8 * np.exp(-np.arange(216) / 40.0)
    sf.write(path, signal, 16000)


def _write_long_rir(path: Path, scale: float = 1.0) -> None:
    signal = np.zeros((4096, 2), dtype=np.float64)
    signal[8, 0] = scale
    signal[8, 1] = scale * 0.8
    tail = np.exp(-np.arange(4088) / 600.0)
    signal[8:, 0] += scale * 0.05 * tail
    signal[8:, 1] += scale * 0.04 * tail
    sf.write(path, signal, 16000)


def _write_mono_rir(path: Path, scale: float = 1.0) -> None:
    signal = np.zeros(256, dtype=np.float64)
    signal[8] = scale
    signal[40:] = scale * np.exp(-np.arange(216) / 40.0)
    sf.write(path, signal, 16000)


def _write_delayed_binaural_rir(path: Path, right_delay: int) -> None:
    signal = np.zeros((4096, 2), dtype=np.float64)
    signal[20, 0] = 1.0
    signal[20 + right_delay, 1] = 0.8
    signal[40:, 0] += 0.01 * np.exp(-np.arange(4056) / 600.0)
    signal[40:, 1] += 0.008 * np.exp(-np.arange(4056) / 600.0)
    sf.write(path, signal, 16000, subtype="FLOAT")


def test_rir_pipeline_describes_registered_route() -> None:
    description = describe_pipeline("rir", metric="rir_npm")
    assert description.pipeline_id == "rir.any.rir_npm.rir_npm_v1"
    assert description.required_roles == ("prediction_rir", "reference_rir")

    npm = describe_pipeline("rir", metric="npm")
    assert npm.pipeline_id == "rir.any.rir_npm.rir_npm_v1"
    assert npm.metric == "rir_npm"

    default = describe_pipeline("rir")
    assert default.pipeline_id == "rir.any.multi.rir_npm.rir_npm_v1__rt30_mae.rt30_mae_v1__rt60_mae.rt60_mae_v1__rir_drr_mae.rir_drr_mae_v1__rir_c50_mae.rir_c50_mae_v1__rir_c80_mae.rir_c80_mae_v1__rir_edc_mae.rir_edc_mae_v1"

    base_bundle = describe_pipeline(
        "rir",
        metrics=("rir_npm", "rt30_mae", "rt60_mae", "rir_drr_mae", "rir_c50_mae", "rir_c80_mae", "rir_edc_mae"),
    )
    assert base_bundle.pipeline_kind == "bundle"
    assert base_bundle.execution_metrics == (
        "rir_npm", "rt30_mae", "rt60_mae", "rir_drr_mae", "rir_c50_mae", "rir_c80_mae", "rir_edc_mae"
    )

    iacc = describe_pipeline("rir", metric="iacc")
    assert iacc.pipeline_id == "rir.any.rir_iacc_mae.rir_iacc_mae_v1"
    assert iacc.metric == "rir_iacc_mae"
    for metric in ("rir_iacc_full_mae", "rir_iacc_early_mae", "rir_iacc_late_mae"):
        window = describe_pipeline("rir", metric=metric)
        assert window.metric == metric
        assert window.pipeline_id == f"rir.any.{metric}.{metric}_v1"

    rt30 = describe_pipeline("rir", metric="rt30_mae")
    assert rt30.pipeline_id == "rir.any.rt30_mae.rt30_mae_v1"
    edc = describe_pipeline("rir", metric="rir_edc_mae")
    assert edc.pipeline_id == "rir.any.rir_edc_mae.rir_edc_mae_v1"


def test_rir_script_describe_and_run_writes_report(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    prediction = tmp_path / "prediction.wav"
    _write_long_rir(reference)
    _write_long_rir(prediction, 0.9)
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
    _write_long_rir(reference)
    _write_long_rir(prediction, 0.9)
    sample = RIRSample(
        prediction_rir=str(prediction),
        reference_rir=str(reference),
        sample_id="rir-001",
        sample_rate=16000,
        channel_layout="binaural_lr",
        metadata={"microphone_spacing_m": 0.18},
    )
    report = evaluate_rir_samples([sample], metrics=("rir_npm", "rt30_mae", "rt60_mae", "rir_drr_mae", "rir_c50_mae", "rir_c80_mae", "rir_edc_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"))
    assert report.pipeline_kind == "bundle"
    assert report.score is not None
    assert set(report.details["results"]) == {"rir_npm", "rt30_mae", "rt60_mae", "rir_drr_mae", "rir_c50_mae", "rir_c80_mae", "rir_edc_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"}
    npm_row = report.details["results"]["rir_npm"]["per_sample"][0]
    assert "reference_energy" in npm_row and "prediction_energy" in npm_row
    rt60_row = report.details["results"]["rt60_mae"]["per_sample"][0]
    assert rt60_row["channels"] and "reference_rt60" in rt60_row["channels"][0]
    rt30_row = report.details["results"]["rt30_mae"]["per_sample"][0]
    assert rt30_row["channels"] and "reference_rt30" in rt30_row["channels"][0]
    drr_row = report.details["results"]["rir_drr_mae"]["per_sample"][0]
    assert drr_row["channels"][0]["reference_drr"] is not None
    c50_row = report.details["results"]["rir_c50_mae"]["per_sample"][0]
    assert c50_row["channels"][0]["reference_c"] is not None
    edc_row = report.details["results"]["rir_edc_mae"]["per_sample"][0]
    assert edc_row["channels"][0]["valid_points"] >= 4
    ild_row = report.details["results"]["rir_ild_mae"]["per_sample"][0]
    assert "reference_ild" in ild_row["pairs"][0]
    assert np.isclose(ild_row["pairs"][0]["score"], ild_row["pairs"][0]["absolute_error_sum"] / ild_row["pairs"][0]["valid_bin_count"])
    ipd_row = report.details["results"]["rir_ipd_mae"]["per_sample"][0]
    assert "reference_ipd" in ipd_row["pairs"][0]
    assert np.isclose(ipd_row["pairs"][0]["score"], ipd_row["pairs"][0]["absolute_error_sum"] / ipd_row["pairs"][0]["valid_bin_count"])
    doa_row = report.details["results"]["rir_doa_mae"]["per_sample"][0]
    assert "reference_doa" in doa_row["pairs"][0]


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


def test_rir_minimal_examples_cover_every_registered_metric(tmp_path: Path) -> None:
    acoustic_metrics = ("rir_npm", "rt30_mae", "rt60_mae", "rir_drr_mae", "rir_c50_mae", "rir_c80_mae", "rir_edc_mae")
    mono_reference = tmp_path / "mono_reference.wav"
    mono_prediction = tmp_path / "mono_prediction.wav"
    mono_reference_stereo = tmp_path / "mono_reference_stereo.wav"
    mono_prediction_stereo = tmp_path / "mono_prediction_stereo.wav"
    _write_long_rir(mono_reference_stereo)
    _write_long_rir(mono_prediction_stereo, 0.9)
    reference_audio, reference_rate = sf.read(mono_reference_stereo, always_2d=True)
    prediction_audio, prediction_rate = sf.read(mono_prediction_stereo, always_2d=True)
    sf.write(mono_reference, reference_audio[:, 0], reference_rate)
    sf.write(mono_prediction, prediction_audio[:, 0], prediction_rate)
    mono = RIRSample(
        prediction_rir=str(mono_prediction), reference_rir=str(mono_reference),
        sample_id="minimal-mono", channel_layout="mono",
    )
    binaural_reference = tmp_path / "binaural_reference.wav"
    binaural_prediction = tmp_path / "binaural_prediction.wav"
    _write_long_rir(binaural_reference)
    _write_long_rir(binaural_prediction, 0.9)
    binaural = RIRSample(
        prediction_rir=str(binaural_prediction), reference_rir=str(binaural_reference),
        sample_id="minimal-binaural", channel_layout="binaural_lr",
        metadata={"microphone_spacing_m": 0.18},
    )
    reference = np.zeros((4096, 4), dtype=np.float64)
    prediction = np.zeros((4096, 4), dtype=np.float64)
    for channel in range(4):
        reference[8 + channel * 2, channel] = 1.0 / (channel + 1)
        prediction[9 + channel * 2, channel] = 0.9 / (channel + 1)
        reference_tail = np.exp(-np.arange(4088 - channel * 2) / 600.0) * 0.05 / (channel + 1)
        prediction_tail = np.exp(-np.arange(4087 - channel * 2) / 600.0) * 0.045 / (channel + 1)
        reference[8 + channel * 2 :, channel] += reference_tail
        prediction[9 + channel * 2 :, channel] += prediction_tail
    np.save(tmp_path / "four_reference.npy", reference)
    np.save(tmp_path / "four_prediction.npy", prediction)
    four = RIRSample(
        prediction_rir=str(tmp_path / "four_prediction.npy"), reference_rir=str(tmp_path / "four_reference.npy"),
        sample_id="minimal-four-channel", sample_rate=16000, channel_layout="multi_channel",
        metadata={"microphone_spacing_m": 0.08},
    )
    mono_report = evaluate_rir_samples([mono], metrics=acoustic_metrics, channel_mode="single")
    binaural_report = evaluate_rir_samples(
        [binaural], metrics=acoustic_metrics + ("rir_iacc_mae", "rir_iacc_full_mae", "rir_iacc_early_mae", "rir_iacc_late_mae", "rir_itd_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"), channel_mode="binaural"
    )
    four_report = evaluate_rir_samples(
        [four], metrics=acoustic_metrics + ("rir_itd_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"), channel_mode="multi"
    )
    covered = set(mono_report.details["results"]) | set(binaural_report.details["results"]) | set(four_report.details["results"])
    assert covered == set(METRICS)
    assert binaural_report.details["results"]["rir_iacc_mae"]["score"] == binaural_report.details["results"]["rir_iacc_full_mae"]["score"]
    assert four_report.details["results"]["rir_doa_mae"]["per_sample"][0]["pair_count"] == 3


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
        metrics=("iacc_full", "iacc_early", "iacc_late", "itd", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"),
        channel_mode="multi",
    )
    assert set(report.details["results"]) == {
        "rir_iacc_full_mae",
        "rir_iacc_early_mae",
        "rir_iacc_late_mae",
        "rir_itd_mae",
        "rir_ild_mae",
        "rir_ipd_mae",
        "rir_doa_mae",
    }
    assert report.details["results"]["rir_iacc_full_mae"]["per_sample"][0]["prediction_iacc"] >= 0
    assert report.details["results"]["rir_itd_mae"]["score"] > 0
    assert report.details["results"]["rir_itd_mae"]["unit"] == "ms"


def test_binaural_metric_rejects_four_channel_rir(tmp_path: Path) -> None:
    reference = tmp_path / "reference.npy"
    prediction = tmp_path / "prediction.npy"
    signal = np.zeros((4, 64), dtype=np.float64)
    signal[:, 4] = 1.0
    np.save(reference, signal)
    np.save(prediction, signal)
    sample = RIRSample(
        prediction_rir=str(prediction),
        reference_rir=str(reference),
        sample_id="rir-four-channel-001",
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
        assert row["pair_count"] == 3
        assert len(row["pairs"]) == 3


def test_multichannel_spatial_scores_weight_samples_equally(tmp_path: Path) -> None:
    two_reference = np.zeros((64, 2), dtype=np.float64)
    two_prediction = np.zeros((64, 2), dtype=np.float64)
    two_reference[10, :] = 1.0
    two_prediction[10, 0] = 1.0
    two_prediction[12, 1] = 0.7
    three_reference = np.zeros((64, 3), dtype=np.float64)
    three_reference[10, :] = 1.0
    for name, array in (
        ("two_reference", two_reference),
        ("two_prediction", two_prediction),
        ("three_reference", three_reference),
        ("three_prediction", three_reference),
    ):
        np.save(tmp_path / f"{name}.npy", array)
    samples = [
        RIRSample(
            prediction_rir=str(tmp_path / "two_prediction.npy"),
            reference_rir=str(tmp_path / "two_reference.npy"),
            sample_id="two-channel",
            sample_rate=16000,
            channel_layout="multi_channel",
            metadata={"microphone_spacing_m": 0.18},
        ),
        RIRSample(
            prediction_rir=str(tmp_path / "three_prediction.npy"),
            reference_rir=str(tmp_path / "three_reference.npy"),
            sample_id="three-channel",
            sample_rate=16000,
            channel_layout="multi_channel",
            metadata={"microphone_spacing_m": 0.18},
        ),
    ]
    report = evaluate_rir_samples(
        samples,
        metrics=("rir_itd_mae", "rir_ild_mae", "rir_ipd_mae", "rir_doa_mae"),
        channel_mode="multi",
    )
    for result in report.details["results"].values():
        first, second = result["per_sample"]
        assert first["pair_count"] == 1
        assert second["pair_count"] == 2
        assert first["score"] > 0
        assert second["score"] == 0
        assert np.isclose(result["score"], first["score"] / 2)


def test_rir_rejects_unscorable_silent_pairs(tmp_path: Path) -> None:
    silence = np.zeros((64, 2), dtype=np.float64)
    np.save(tmp_path / "silence.npy", silence)
    sample = RIRSample(
        prediction_rir=str(tmp_path / "silence.npy"),
        reference_rir=str(tmp_path / "silence.npy"),
        sample_id="silent-pair",
        sample_rate=16000,
        channel_layout="binaural_lr",
    )
    for metric, message in (
        ("rir_npm", "nonzero RIR energy"),
        ("rir_ild_mae", "no valid spectral bins"),
        ("rir_ipd_mae", "no valid spectral bins"),
    ):
        try:
            evaluate_rir_samples([sample], metrics=(metric,))
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"{metric} accepted a silent RIR as valid")


@pytest.mark.parametrize("missing_field", ("sample_id", "prediction_rir", "reference_rir"))
def test_rir_samples_jsonl_contract_rejects_missing_required_roles(tmp_path: Path, missing_field: str) -> None:
    fixture = tmp_path / "samples.jsonl"
    row = {"sample_id": "rir-001", "prediction_rir": "prediction.wav", "reference_rir": "reference.wav"}
    del row[missing_field]
    fixture.write_text(json.dumps(row) + "\n", encoding="utf-8")
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_rir_samples_jsonl

    with pytest.raises(SampleJsonlError, match=f"{missing_field} is required"):
        load_rir_samples_jsonl(fixture)


@pytest.mark.parametrize("field", ("prediction_rir", "reference_rir"))
@pytest.mark.parametrize("invalid_value", (42, ["rir.wav"], {"path": "rir.wav"}))
def test_rir_samples_jsonl_contract_rejects_malformed_required_paths(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    from sure_eval.evaluation.audio_samples import SampleJsonlError, load_rir_samples_jsonl

    fixture = tmp_path / "samples.jsonl"
    row = {"sample_id": "rir-001", "prediction_rir": "prediction.wav", "reference_rir": "reference.wav"}
    row[field] = invalid_value
    fixture.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(SampleJsonlError, match=f"{field} must be a path string"):
        load_rir_samples_jsonl(fixture)


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
        "rt30_mae",
        "rt60_mae",
        "rir_drr_mae",
        "rir_c50_mae",
        "rir_c80_mae",
        "rir_edc_mae",
        "rir_iacc_mae",
        "rir_iacc_full_mae",
        "rir_iacc_early_mae",
        "rir_iacc_late_mae",
        "rir_itd_mae",
        "rir_ild_mae",
        "rir_ipd_mae",
        "rir_doa_mae",
    ):
        result = checker.check_node(f"scoring/{metric}")
        assert result.status == "ok"
        assert result.runtime == "in_process"
        assert result.required is False


@pytest.mark.parametrize("metric", ("rt30_mae", "rt60_mae", "rir_drr_mae", "rir_c50_mae", "rir_c80_mae", "rir_edc_mae"))
@pytest.mark.parametrize("invalid_role", ("prediction", "reference"))
def test_rir_acoustic_metrics_do_not_drop_unscorable_channels(tmp_path: Path, metric: str, invalid_role: str) -> None:
    tail = np.random.default_rng(42).normal(size=16000) * np.exp(-np.arange(16000) / 1500.0) * 0.05
    tail[20] += 1.0
    valid = np.column_stack((tail, tail))
    invalid = valid.copy()
    invalid[:, 1] = 0.0
    for role in ("prediction", "reference"):
        np.save(tmp_path / f"{role}.npy", invalid if role == invalid_role else valid)
    sample = RIRSample(str(tmp_path / "prediction.npy"), str(tmp_path / "reference.npy"), sample_rate=16000, sample_id="missing-channel")
    if metric == "rir_edc_mae" and invalid_role == "prediction":
        # EDC MAE can quantify a silent prediction against a valid decay;
        # the undefined optional correlation must not remove this channel.
        result = evaluate_rir_samples([sample], metrics=(metric,)).details["results"][metric]
        assert len(result["per_sample"][0]["channels"]) == 2
        assert result["score"] > 1.0
        assert result["per_sample"][0]["channels"][1]["edc_correlation"] is None
        json.dumps(result, allow_nan=False)
    else:
        with pytest.raises(ValueError, match="missing-channel, channel 1"):
            evaluate_rir_samples([sample], metrics=(metric,))


@pytest.mark.parametrize("metric", ("rir_iacc_full_mae", "rir_iacc_early_mae", "rir_iacc_late_mae", "rir_itd_mae", "rir_doa_mae"))
@pytest.mark.parametrize("invalid_role", ("prediction", "reference"))
def test_rir_spatial_metrics_reject_undefined_channel_correlation(tmp_path: Path, metric: str, invalid_role: str) -> None:
    tail = np.random.default_rng(7).normal(size=4096) * np.exp(-np.arange(4096) / 600.0)
    valid = np.column_stack((tail, tail))
    invalid = valid.copy()
    invalid[:, 1] = 0.0
    for role in ("prediction", "reference"):
        np.save(tmp_path / f"{role}.npy", invalid if role == invalid_role else valid)
    sample = RIRSample(str(tmp_path / "prediction.npy"), str(tmp_path / "reference.npy"), sample_rate=16000, metadata={"microphone_spacing_m": 0.18})
    with pytest.raises(ValueError, match="nonzero centered energy"):
        evaluate_rir_samples([sample], metrics=(metric,))


def test_rir_iacc_rejects_empty_late_window(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    _write_rir(path)
    with pytest.raises(ValueError, match="IACC late could not be estimated"):
        evaluate_rir_samples([RIRSample(str(path), str(path))], metrics=("rir_iacc_late_mae",))


def test_rir_report_preserves_geometry_and_truncation_inputs(tmp_path: Path) -> None:
    reference = np.zeros((4096, 2))
    reference[20] = 1.0
    np.save(tmp_path / "reference.npy", reference)
    np.save(tmp_path / "prediction.npy", reference[:2048])
    metadata = {"microphone_spacing_m": 0.18, "sound_speed_mps": 343.0, "channel_spacings_m": [0.18]}
    sample = RIRSample(str(tmp_path / "prediction.npy"), str(tmp_path / "reference.npy"), sample_rate=16000, sample_id="cropped", channel_layout="binaural_lr", metadata=metadata)
    report = evaluate_rir_samples([sample], metrics=("rir_npm",))
    row = report.details["rows"][0]
    assert row["prediction_rir"] == sample.prediction_rir
    assert row["reference_rir"] == sample.reference_rir
    assert row["metadata"] == metadata
    assert row["sample_rate"] == 16000
    assert row["channel_layout"] == "binaural_lr"
    assert row["channel_count"] == 2
    assert (row["prediction_samples"], row["reference_samples"], row["paired_samples"]) == (2048, 4096, 2048)
    assert row["length_policy"] == "truncate_to_shorter"
    assert report.details["results"]["rir_npm"]["score"] < -100


@pytest.mark.parametrize("metadata", (
    {"microphone_spacing_m": float("nan")},
    {"microphone_spacing_m": 0.18, "sound_speed_mps": 0.0},
    {"microphone_spacing_m": 0.18, "channel_spacings_m": []},
    {"microphone_spacing_m": 0.18, "channel_spacings_m": [float("inf")]},
))
def test_rir_doa_rejects_invalid_geometry(tmp_path: Path, metadata: dict) -> None:
    path = tmp_path / "rir.wav"
    _write_rir(path)
    with pytest.raises(ValueError, match="spacing|sound_speed_mps"):
        evaluate_rir_samples([RIRSample(str(path), str(path), metadata=metadata)], metrics=("rir_doa_mae",))


@pytest.mark.parametrize("spacing", (0.09, 0.18))
def test_rir_known_itd_and_doa_within_supported_window(tmp_path: Path, spacing: float) -> None:
    reference = np.zeros((256, 2))
    reference[20] = 1.0
    prediction = np.zeros_like(reference)
    prediction[20, 0] = 1.0
    prediction[24, 1] = 1.0
    np.save(tmp_path / "reference.npy", reference)
    np.save(tmp_path / "prediction.npy", prediction)
    sample = RIRSample(str(tmp_path / "prediction.npy"), str(tmp_path / "reference.npy"), sample_rate=16000, metadata={"microphone_spacing_m": spacing})
    results = evaluate_rir_samples([sample], metrics=("rir_itd_mae", "rir_doa_mae")).details["results"]
    assert results["rir_itd_mae"]["score"] == pytest.approx(0.25)
    expected_angle = np.degrees(np.arcsin(4 / 16000 * 343 / spacing))
    assert results["rir_doa_mae"]["score"] == pytest.approx(expected_angle)


@pytest.mark.parametrize("value", (0.0, 1.0))
def test_rir_itd_rejects_silent_or_constant_pairs(tmp_path: Path, value: float) -> None:
    path = tmp_path / "constant.npy"
    np.save(path, np.full((256, 2), value))
    with pytest.raises(ValueError, match="nonzero centered energy"):
        evaluate_rir_samples([RIRSample(str(path), str(path), sample_rate=16000)], metrics=("rir_itd_mae",))
