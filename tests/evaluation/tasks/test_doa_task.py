from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(frame_index: int, azimuth: float) -> dict:
    return {
        "recording_id": "rec-1",
        "frame_index": frame_index,
        "timestamp": frame_index * 0.01,
        "num_channels": 2,
        "coordinate_system": "array_local_spherical",
        "angle_unit": "degree",
        "sources": [{"source_id": "s0", "azimuth": azimuth, "activity": True}],
    }


def _canonical_files(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [_row(0, 10.0), _row(1, 20.0)])
    _write_jsonl(prediction, [_row(0, 12.0), _row(1, 22.0)])
    return reference, prediction


def test_doa_routes_describe_canonical() -> None:
    from sure_eval.evaluation.scripts import describe_pipeline

    canonical = describe_pipeline("doa", metric="mae")
    assert canonical.task == "DOA"
    assert canonical.pipeline_id == "doa.any.mae.doa_jsonl_v1.doa_localization_v1"
    assert canonical.node_ids == ("normalization/doa_jsonl", "scoring/doa_localization")


def test_doa_nodes_are_in_process_and_require_no_node_environment() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    checker = NodeEnvChecker()
    for node_id in ("normalization/doa_jsonl", "scoring/doa_localization"):
        result = checker.check_node(node_id)
        assert result.status == "ok"
        assert result.runtime == "in_process"
        assert result.required is False


def test_doa_generated_catalog_and_atlas_contain_formal_routes() -> None:
    catalog_path = Path("docs/pipeline_catalog.jsonl")
    rows = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines()]
    doa_rows = [row for row in rows if row["task"] == "DOA"]

    assert len(doa_rows) == 8
    assert {row["metric"] for row in doa_rows} == {
        "mae",
        "rmse",
        "acc_10",
        "acc_15",
        "acc_30",
        "localization_recall",
        "localization_precision",
        "localization_f1",
    }
    assert all(
        row["nodes"] == ["normalization/doa_jsonl", "scoring/doa_localization"] for row in doa_rows
    )
    assert "doa.any.mae.doa_jsonl_v1.doa_localization_v1" in Path(
        "docs/atlas/index.html"
    ).read_text(encoding="utf-8")


def test_doa_run_writes_trace_and_report(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts import run_task

    reference, prediction = _canonical_files(tmp_path)
    output_dir = tmp_path / "out"
    report = run_task(
        "doa",
        reference_jsonl=str(reference),
        sample_output=str(prediction),
        metric="mae",
        output_dir=str(output_dir),
    )

    assert report.task == "DOA"
    assert report.score == pytest.approx(2.0)
    assert report.pipeline_id == "doa.any.mae.doa_jsonl_v1.doa_localization_v1"
    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert [item["node_id"] for item in payload["pipeline_trace"]] == [
        "normalization/doa_jsonl",
        "scoring/doa_localization",
    ]
    assert payload["details"]["evaluation_config"] == {
        "dimension": "2d",
        "threshold_deg": 10.0,
        "configured_threshold_deg": 10.0,
        "aggregation": "micro",
        "timestamp_tolerance_sec": 1e-6,
        "prediction_activity_policy": "presence",
    }
    assert set(payload["details"]["input_sha256"]) == {
        "reference_jsonl",
        "sample_output",
    }
    assert all(len(value) == 64 for value in payload["details"]["input_sha256"].values())


def test_doa_cli_describe_and_run_preserve_route_identity(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sure_eval.cli import app

    reference, prediction = _canonical_files(tmp_path)
    pipeline_path = tmp_path / "pipeline.json"
    output_dir = tmp_path / "cli-out"
    runner = CliRunner()
    describe = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "doa",
            "--metric",
            "mae",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe.exit_code == 0, describe.stdout
    run = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--reference-jsonl",
            str(reference),
            "--sample-output",
            str(prediction),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )
    assert run.exit_code == 0, run.stdout
    assert json.loads(run.stdout)["pipeline_id"] == "doa.any.mae.doa_jsonl_v1.doa_localization_v1"


def test_doa_acc_route_reports_its_fixed_effective_threshold(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts import run_task

    reference, prediction = _canonical_files(tmp_path)
    report = run_task(
        "doa",
        reference_jsonl=str(reference),
        sample_output=str(prediction),
        metric="acc_15",
        output_dir=str(tmp_path / "out"),
    )

    assert report.pipeline_id == "doa.any.acc_15.doa_jsonl_v1.doa_localization_v1"
    assert report.details["evaluation_config"]["threshold_deg"] == 15.0
    assert report.details["scoring"]["micro"]["true_positive_15"] == 2


def test_doa_pipeline_id_helper_rejects_unknown_metric() -> None:
    from sure_eval.evaluation.tasks.doa.pipeline import pipeline_id_for_metric

    with pytest.raises(ValueError, match="Unsupported DOA metric"):
        pipeline_id_for_metric("not_a_metric")
