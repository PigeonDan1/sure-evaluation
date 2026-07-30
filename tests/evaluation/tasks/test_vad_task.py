from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    reference_jsonl = tmp_path / "ref.jsonl"
    sample_output = tmp_path / "pred.jsonl"
    _write_jsonl(
        reference_jsonl,
        [
            {
                "key": "utt1",
                "duration": 1.0,
                "speech_segments": [{"start": 0.2, "end": 0.6}],
            }
        ],
    )
    _write_jsonl(
        sample_output,
        [
            {
                "key": "utt1",
                "speech_segments": [{"start": 0.2, "end": 0.6}],
                "frame_scores": [
                    {"start": 0.0, "end": 0.2, "score": 0.1},
                    {"start": 0.2, "end": 0.6, "score": 0.9},
                    {"start": 0.6, "end": 1.0, "score": 0.2},
                ],
                "audio_duration": 1.0,
            }
        ],
    )
    return reference_jsonl, sample_output


def test_vad_route_describe_contains_expected_nodes() -> None:
    from sure_eval.evaluation.scripts.vad import describe_pipeline

    desc = describe_pipeline(metric="f1")

    assert desc.task == "VAD"
    assert desc.language == "n/a"
    assert desc.metric == "f1"
    assert desc.pipeline_id == (
        "vad.any.f1.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1"
    )
    assert desc.node_ids == (
        "validation/vad_contract",
        "normalization/vad_timebase",
        "scoring/vad_detection_duration",
    )
    assert desc.required_roles == ("reference_jsonl", "sample_output")
    assert desc.execution_metrics == ("f1",)


def test_vad_run_task_report_shape(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts import run_task

    reference_jsonl, sample_output = _fixture_files(tmp_path)
    output_dir = tmp_path / "vad_out"

    report = run_task(
        "vad",
        reference_jsonl=str(reference_jsonl),
        sample_output=str(sample_output),
        metric="f1",
        output_dir=str(output_dir),
    )

    assert report.task == "VAD"
    assert report.metric == "f1"
    assert report.score == 1.0
    assert report.pipeline_id == (
        "vad.any.f1.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1"
    )
    assert (output_dir / "report.json").exists()
    assert (output_dir / "pipeline_description.json").exists()

    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    description_payload = json.loads(
        (output_dir / "pipeline_description.json").read_text(encoding="utf-8")
    )
    assert report_payload["details"]["results"]["f1"]["score"] == 1.0
    assert report_payload["details"]["auxiliary"]["num_detection_samples"] == 1
    assert [node["node_id"] for node in report_payload["pipeline_trace"]] == [
        "validation/vad_contract",
        "normalization/vad_timebase",
        "scoring/vad_detection_duration",
    ]
    assert description_payload["pipeline_id"] == report_payload["pipeline_id"]
    assert description_payload["required_roles"] == ["reference_jsonl", "sample_output"]


def test_vad_cli_describe_run_preserves_pipeline_id(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sure_eval.cli import app

    reference_jsonl, sample_output = _fixture_files(tmp_path)
    pipeline_path = tmp_path / "vad_pipeline.json"
    output_dir = tmp_path / "vad_cli_out"
    pipeline_id = "vad.any.f1.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1"
    runner = CliRunner()

    describe_result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "vad",
            "--metric",
            "f1",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe_result.exit_code == 0, describe_result.stdout

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--reference-jsonl",
            str(reference_jsonl),
            "--sample-output",
            str(sample_output),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )
    assert run_result.exit_code == 0, run_result.stdout

    summary = json.loads(run_result.stdout)
    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert summary["pipeline_id"] == pipeline_id
    assert report_payload["pipeline_id"] == pipeline_id
    assert report_payload["metric"] == "f1"


def test_pipeline_catalog_contains_vad_routes() -> None:
    from scripts.generate_pipeline_catalog import COMBINATIONS
    from sure_eval.evaluation.scripts import describe_pipeline

    vad_combinations = [kwargs for task, kwargs in COMBINATIONS if task == "vad"]
    rows = [describe_pipeline("vad", **kwargs) for kwargs in vad_combinations]

    assert {row.metric for row in rows} == {"f1", "p_fa", "p_miss", "dcf_nist", "auc_roc"}
    assert {row.pipeline_id for row in rows} == {
        "vad.any.f1.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1",
        "vad.any.p_fa.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1",
        "vad.any.p_miss.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1",
        "vad.any.dcf_nist.vad_contract_v1.vad_timebase_strict_v1.vad_detection_duration_v1",
        "vad.any.auc_roc.vad_contract_v1.vad_timebase_strict_v1.vad_auc_roc_v1",
    }
