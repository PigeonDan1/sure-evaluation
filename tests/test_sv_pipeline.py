from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sure_eval.evaluation.cli import app
from sure_eval.evaluation.scripts.sv import describe_pipeline
from sure_eval.evaluation.tasks.sv.pipeline import evaluate_sv_files


def _write_sv_fixture(tmp_path: Path) -> tuple[Path, Path]:
    sample_output = tmp_path / "embeddings.jsonl"
    rows = [
        {"key": "demo__a", "result": {"embedding": [1.0, 0.0], "dimension": 2}},
        {"key": "demo__b", "result": {"embedding": [0.9, 0.1], "dimension": 2}},
        {"key": "demo__c", "result": {"embedding": [0.0, 1.0], "dimension": 2}},
        {"key": "demo__d", "result": {"embedding": [0.1, 0.9], "dimension": 2}},
        {"key": "demo__unused", "result": {"embedding": [1.0, 1.0], "dimension": 2}},
    ]
    sample_output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    trials = tmp_path / "trials.tsv"
    trials.write_text(
        "enroll_key\ttest_key\tlabel\tcondition\n"
        "demo__a\tdemo__b\ttarget\tdefault\n"
        "demo__c\tdemo__d\ttarget\tdefault\n"
        "demo__a\tdemo__c\tnontarget\tdefault\n"
        "demo__b\tdemo__d\tnontarget\tdefault\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "trial_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "sure.sv.trial_manifest.v1",
                "dataset_id": "opensvbench-demo-sv-v1",
                "testset_id": "demo",
                "source_dataset_id": "demo_source",
                "trials_file": trials.name,
                "trials_sha256": hashlib.sha256(trials.read_bytes()).hexdigest(),
                "trial_count": 4,
                "target_count": 2,
                "nontarget_count": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return sample_output, manifest


def test_sv_bundle_scores_testset_trials(tmp_path: Path) -> None:
    sample_output, manifest = _write_sv_fixture(tmp_path)
    report = evaluate_sv_files(str(sample_output), str(manifest), work_dir=tmp_path / "work")

    assert report.task == "SV"
    assert report.metric == "multi"
    assert report.pipeline_kind == "bundle"
    assert report.score == 0.0
    assert report.details["results"]["eer"]["eer_percent"] == 0.0
    assert report.details["results"]["min_dcf"]["min_dcf"] == 0.0
    assert report.details["dataset"]["extra_embedding_count"] == 1
    assert report.details["dataset"]["dataset_id"] == "opensvbench-demo-sv-v1"
    assert report.details["dataset"]["testset_id"] == "demo"
    assert report.details["dataset"]["source_dataset_id"] == "demo_source"
    assert report.computation_node_ids == (
        "scoring/cosine_trial_scores",
        "scoring/det_eer",
        "scoring/min_dcf_p005",
    )


def test_sv_atomic_pipeline_identity(tmp_path: Path) -> None:
    sample_output, manifest = _write_sv_fixture(tmp_path)
    report = evaluate_sv_files(
        str(sample_output),
        str(manifest),
        metrics=("min_dcf",),
        work_dir=tmp_path / "work",
    )
    assert report.pipeline_id == "sv.any.min_dcf.cosine_trial_scores_v1.min_dcf_p005_v1"
    assert report.pipeline_kind == "atomic"
    assert report.member_pipeline_ids == ()


def test_sv_describe_default_bundle() -> None:
    description = describe_pipeline()
    assert description.pipeline_kind == "bundle"
    assert description.required_roles == ("sample_output", "trial_manifest")
    assert description.execution_metrics == ("eer", "min_dcf")


def test_sv_rejects_duplicate_embedding_key(tmp_path: Path) -> None:
    sample_output, manifest = _write_sv_fixture(tmp_path)
    with sample_output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": "demo__a", "result": {"embedding": [1.0, 0.0]}}) + "\n")
    with pytest.raises(ValueError, match="Duplicate SV embedding key"):
        evaluate_sv_files(str(sample_output), str(manifest), work_dir=tmp_path / "work")


def test_sv_rejects_missing_embedding(tmp_path: Path) -> None:
    sample_output, manifest = _write_sv_fixture(tmp_path)
    lines = sample_output.read_text(encoding="utf-8").splitlines()
    sample_output.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required embedding key"):
        evaluate_sv_files(str(sample_output), str(manifest), work_dir=tmp_path / "work")


def test_sv_rejects_trial_fingerprint_mismatch(tmp_path: Path) -> None:
    sample_output, manifest = _write_sv_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["trials_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluate_sv_files(str(sample_output), str(manifest), work_dir=tmp_path / "work")


def test_sv_metric_cli_describe_and_run(tmp_path: Path) -> None:
    runner = CliRunner()
    sample_output, manifest = _write_sv_fixture(tmp_path)
    pipeline = tmp_path / "sv.json"
    output_dir = tmp_path / "report"
    describe_result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "sv",
            "--metrics",
            "eer,min_dcf",
            "--output",
            str(pipeline),
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
            str(pipeline),
            "--sample-output",
            str(sample_output),
            "--trial-manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )
    assert run_result.exit_code == 0, run_result.stdout
    payload = json.loads(run_result.stdout)
    assert payload["task"] == "SV"
    assert payload["score"] == 0.0
    assert (output_dir / "report.json").is_file()
