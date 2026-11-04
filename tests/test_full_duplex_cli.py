from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sure_eval.cli import app

_METRICS = ("ird", "fsed", "ert", "eit")
_FIXTURES = Path(__file__).parent / "fixtures" / "full_duplex"
_DEFAULT_PIPELINE_ID = "full_duplex.any.ird.duplex_timeline_v1.fdbench_timing_v1"


def _run_fixture_case(
    tmp_path: Path, *, case: str, metric: str, measurement_protocol: dict | None = None,
    json_output: bool = True,
) -> tuple[dict | str, dict]:
    paths = {}
    for name in ("reference", "sample_output"):
        rows = [
            json.loads(line)
            for line in (_FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        path = tmp_path / f"{name}.jsonl"
        row = next(row for row in rows if row["key"] == case)
        if name == "sample_output" and measurement_protocol is not None:
            row["metadata"] = {"measurement_protocol": measurement_protocol}
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        paths[name] = path
    runner = CliRunner()
    pipeline_path = tmp_path / "pipeline.json"
    describe = runner.invoke(
        app,
        [
            "metric", "describe", "full_duplex", "--metric", metric,
            "--output", str(pipeline_path), "--json",
        ],
    )
    assert describe.exit_code == 0, describe.stdout
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "metric", "run", "--pipeline", str(pipeline_path),
            "--reference-jsonl", str(paths["reference"]),
            "--sample-output", str(paths["sample_output"]),
            "--output-dir", str(output_dir),
        ] + (["--json"] if json_output else []),
        env={"COLUMNS": "160"},
    )
    assert result.exit_code == 0, result.stdout
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    return (json.loads(result.stdout) if json_output else result.stdout), report


@pytest.mark.parametrize("metric", ["ert", "eit"])
def test_full_duplex_cli_empty_pool_summary_distinguishes_unmeasured_zero(
    tmp_path: Path, metric: str
) -> None:
    summary, report = _run_fixture_case(tmp_path, case="case-I", metric=metric)

    assert summary["score"] == report["score"] == 0.0
    assert summary["sample_count"] == 0
    assert summary["status"] == "no_events"
    assert summary["score_is_measured"] is False
    warning = summary["warning"].lower()
    assert "compatibility" in warning
    assert "not a measured 0 ms" in warning
    assert summary["warnings"]
    assert summary["warning"] == " ".join(summary["warnings"])


@pytest.mark.parametrize(
    "case,expected_score", [("case-J", 0.0), ("case-A", 400.0)]
)
def test_full_duplex_cli_measured_summary_uses_event_count(
    tmp_path: Path, case: str, expected_score: float
) -> None:
    summary, report = _run_fixture_case(tmp_path, case=case, metric="fsed")

    assert summary["score"] == report["score"] == expected_score
    assert summary["sample_count"] == 1
    assert summary["status"] == "ok"
    assert summary["score_is_measured"] is True
    assert not any("not a measured 0 ms" in warning for warning in summary["warnings"])


def test_full_duplex_cli_missing_protocol_warns_and_still_scores(tmp_path: Path) -> None:
    summary, report = _run_fixture_case(tmp_path, case="case-A", metric="fsed")

    assert summary["score"] == report["score"] == 400.0
    assert summary["measurement_protocol_status"] == "missing"
    assert any("measurement protocol" in warning.lower() for warning in summary["warnings"])
    assert "missing" in summary["warning"].lower()


@pytest.mark.parametrize(
    "case,metric,expected_status,expected_measured",
    [("case-I", "ert", "no_events", "False"), ("case-J", "fsed", "ok", "True")],
)
def test_full_duplex_cli_human_summary_discloses_measurement_status(
    tmp_path: Path, case: str, metric: str, expected_status: str, expected_measured: str
) -> None:
    output, report = _run_fixture_case(tmp_path, case=case, metric=metric, json_output=False)

    assert report["score"] == 0.0
    normalized = " ".join(output.split())
    assert expected_status in normalized
    assert "sample_count" in normalized
    assert "score_is_measured" in normalized
    assert expected_measured in normalized
    assert "measurement_protocol_status" in normalized
    assert "Warning:" in normalized
    if expected_status == "no_events":
        assert "not a measured 0 ms delay" in normalized
    else:
        assert "not a measured 0 ms delay" not in normalized


def test_full_duplex_cli_partial_protocol_warns_and_still_scores(tmp_path: Path) -> None:
    summary, report = _run_fixture_case(
        tmp_path, case="case-A", metric="fsed",
        measurement_protocol={"vad": {"name": "manual-test"}},
    )

    assert summary["score"] == report["score"] == 400.0
    assert summary["measurement_protocol_status"] == "partial"
    assert any("measurement protocol" in warning.lower() for warning in summary["warnings"])
    assert "partial" in summary["warning"].lower()


def test_full_duplex_cli_complete_protocol_has_no_provenance_warning(tmp_path: Path) -> None:
    protocol = {
        "vad": {
            "name": "manual-test", "version": "1", "threshold": 0.5,
            "min_silence_ms": 1500, "padding_ms": 30,
        },
        "playback": {
            "queue_policy": "retain_all_received_audio", "clock": "shared_monotonic",
            "speaker_output_observed": False,
        },
        "reference": {"source": "upstream_machine_schedule"},
        "preamble": {"removed": False, "reason": "none"},
    }
    summary, report = _run_fixture_case(
        tmp_path, case="case-A", metric="fsed", measurement_protocol=protocol,
    )

    assert summary["score"] == report["score"] == 400.0
    assert summary["measurement_protocol_status"] == "recorded"
    assert summary["warnings"] == []
    assert "warning" not in summary


def test_full_duplex_summary_fields_leave_other_task_summaries_unchanged(tmp_path: Path) -> None:
    from sure_eval.evaluation.cli_adapters import build_pipeline_spec, run_pipeline_spec

    pipeline = build_pipeline_spec(
        "asr", pipeline_id="asr.en.wer.whisper_norm_english_v1.wenet_wer_v1"
    )
    reference = tmp_path / "reference.txt"
    prediction = tmp_path / "prediction.txt"
    reference.write_text("utt1\thello world\n", encoding="utf-8")
    prediction.write_text("utt1\thello world\n", encoding="utf-8")
    summary = run_pipeline_spec(
        pipeline, output_dir=str(tmp_path / "output"),
        ref_file=str(reference), hyp_file=str(prediction),
    )

    assert summary["status"] == "ok"
    assert summary["score"] == 0.0
    assert set(summary) == {
        "status", "task", "metric", "score", "pipeline_id", "output_dir", "report_path",
        "pipeline_description_path", "environment_note", "node_config_paths",
    }


def test_full_duplex_cli_routes_have_four_atomic_choices() -> None:
    result = CliRunner().invoke(app, ["metric", "routes", "full_duplex", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["count"] == 4
    assert {route["metric"] for route in payload["routes"]} == set(_METRICS)
    assert {route["pipeline_id"] for route in payload["routes"]} == {
        f"full_duplex.any.{metric}.duplex_timeline_v1.fdbench_timing_v1" for metric in _METRICS
    }
    assert payload["default_pipeline_id"] == _DEFAULT_PIPELINE_ID
    assert payload["default_pipeline_ids"] == [_DEFAULT_PIPELINE_ID]
    assert [route["metric"] for route in payload["routes"] if route["default"]] == ["ird"]
    assert all(route["execution_metrics"] == [route["metric"]] for route in payload["routes"])
    assert all(route["setup_node_ids"] == [] for route in payload["routes"])


def test_full_duplex_cli_filtered_routes_preserve_task_default() -> None:
    result = CliRunner().invoke(
        app, ["metric", "routes", "full_duplex", "--metric", "fsed", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["count"] == 1
    assert payload["default_pipeline_id"] == _DEFAULT_PIPELINE_ID
    assert payload["default_pipeline_ids"] == [_DEFAULT_PIPELINE_ID]
    route = payload["routes"][0]
    assert route["metric"] == "fsed"
    assert route["pipeline_id"] == "full_duplex.any.fsed.duplex_timeline_v1.fdbench_timing_v1"
    assert route["default"] is False
    assert route["execution_metrics"] == ["fsed"]


@pytest.mark.parametrize("metric", _METRICS)
def test_full_duplex_cli_describe_and_exact_pipeline_run(tmp_path: Path, metric: str) -> None:
    pipeline_id = f"full_duplex.any.{metric}.duplex_timeline_v1.fdbench_timing_v1"
    pipeline_path = tmp_path / "pipeline.json"
    output_dir = tmp_path / "output"
    runner = CliRunner()
    describe = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "full_duplex",
            "--pipeline-id",
            pipeline_id,
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe.exit_code == 0, describe.stdout
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert pipeline["metric"] == metric
    assert pipeline["requested_metric"] == metric
    assert pipeline["metrics"] == [metric]
    assert pipeline["pipeline_kind"] == "atomic"
    assert pipeline["required_roles"] == ["reference_jsonl", "sample_output"]
    run = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--reference-jsonl",
            str(_FIXTURES / "reference.jsonl"),
            "--sample-output",
            str(_FIXTURES / "sample_output.jsonl"),
            "--output-dir",
            str(output_dir),
            "--validate-env",
            "--json",
        ],
    )
    assert run.exit_code == 0, run.stdout
    summary = json.loads(run.stdout)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    expected = json.loads((_FIXTURES / "expected.json").read_text(encoding="utf-8"))
    assert summary["pipeline_id"] == report["pipeline_id"] == pipeline_id
    assert report["metric"] == metric
    assert report["score"] == expected["pooled"][metric]["score"]
    for reported_metric in _METRICS:
        actual = report["details"]["results"][reported_metric]
        assert actual["score"] == expected["pooled"][reported_metric]["score"]
        assert actual["sample_count"] == expected["pooled"][reported_metric]["sample_count"]


@pytest.mark.parametrize("field", ["computation_node_ids", "nodes", "pipeline_id"])
def test_full_duplex_pipeline_rejects_identity_tampering(tmp_path: Path, field: str) -> None:
    from sure_eval.evaluation.cli_adapters import build_pipeline_spec, validate_pipeline_identity

    payload = build_pipeline_spec("full_duplex", metric="fsed")
    if field == "pipeline_id":
        payload[field] = "full_duplex.any.fsed.duplex_timeline_v1.fdbench_timing_v999"
    else:
        payload[field] = []
    with pytest.raises((ValueError, KeyError)):
        validate_pipeline_identity(payload)


def test_full_duplex_cli_rejects_unregistered_metric() -> None:
    result = CliRunner().invoke(
        app, ["metric", "describe", "full_duplex", "--metric", "srr", "--json"]
    )
    assert result.exit_code == 1
    assert "Unsupported FULL_DUPLEX metric" in result.stdout


@pytest.mark.parametrize(
    "case,role,error_tokens",
    [
        ("missing_file", "reference_jsonl", ("reference.jsonl",)),
        ("missing_file", "sample_output", ("sample_output.jsonl",)),
        ("bad_json", "reference_jsonl", ("reference.jsonl:1", "Expecting")),
        ("bad_json", "sample_output", ("sample_output.jsonl:1", "Expecting")),
        ("missing_duration", "reference_jsonl", ("reference.jsonl:1", "duration")),
        ("missing_user_segments", "reference_jsonl", ("reference.jsonl:1", "user_segments")),
        ("missing_system_segments", "sample_output", ("sample_output.jsonl:1", "system_segments")),
        ("missing_key", "reference_jsonl", ("reference.jsonl:1", "key")),
        ("missing_key", "sample_output", ("sample_output.jsonl:1", "key")),
        ("duplicate_key", "reference_jsonl", ("reference.jsonl:2", "duplicate", "case-invalid")),
        ("duplicate_key", "sample_output", ("sample_output.jsonl:2", "duplicate", "case-invalid")),
        ("reversed_segment", "reference_jsonl", ("case-invalid", "user_segments[0]")),
        ("reversed_segment", "sample_output", ("case-invalid", "system_segments[0]")),
        ("nan_duration", "reference_jsonl", ("reference.jsonl:1", "NaN")),
        ("infinity_duration", "reference_jsonl", ("reference.jsonl:1", "Infinity")),
        ("off_grid", "sample_output", ("case-invalid", "system segment", "a1", "grid")),
        ("key_set_mismatch", "sample_output", ("key mismatch", "case-invalid", "case-extra")),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_full_duplex_cli_malformed_inputs_are_locatable_and_write_no_report(
    tmp_path: Path, case: str, role: str, error_tokens: tuple[str, ...]
) -> None:
    shared = {
        "schema_version": 1,
        "key": "case-invalid",
        "time_unit": "seconds",
        "time_origin": "conversation_start",
    }
    rows = {
        "reference_jsonl": {
            **shared,
            "duration": 10,
            "user_segments": [{"id": "u1", "start": 1, "end": 3}],
        },
        "sample_output": {
            **shared,
            "system_segments": [{"id": "a1", "start": 3.5, "end": 4}],
        },
    }
    invalid_row = rows[role]
    if case.startswith("missing_") and case != "missing_file":
        del invalid_row[case.removeprefix("missing_")]
    elif case == "reversed_segment":
        channel = "user_segments" if role == "reference_jsonl" else "system_segments"
        invalid_row[channel][0].update(start=4, end=3)
    elif case == "nan_duration":
        invalid_row["duration"] = float("nan")
    elif case == "infinity_duration":
        invalid_row["duration"] = float("inf")
    elif case == "off_grid":
        invalid_row["system_segments"][0]["start"] = 3.00001
    elif case == "key_set_mismatch":
        invalid_row["key"] = "case-extra"

    paths = {
        "reference_jsonl": tmp_path / "reference.jsonl",
        "sample_output": tmp_path / "sample_output.jsonl",
    }
    for name, row in rows.items():
        if case == "missing_file" and role == name:
            continue
        contents = json.dumps(row) + "\n"
        if role == name:
            if case == "bad_json":
                contents = "{invalid JSON}\n"
            elif case == "duplicate_key":
                contents += contents
        paths[name].write_text(contents, encoding="utf-8")

    runner = CliRunner()
    pipeline_path = tmp_path / "pipeline.json"
    describe = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "full_duplex",
            "--metric",
            "fsed",
            "--output",
            str(pipeline_path),
            "--json",
        ],
        env={"COLUMNS": "160"},
    )
    assert describe.exit_code == 0, describe.stdout
    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--reference-jsonl",
            str(paths["reference_jsonl"]),
            "--sample-output",
            str(paths["sample_output"]),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        env={"COLUMNS": "160"},
    )
    assert result.exit_code != 0, result.stdout
    error = json.loads(result.stdout)
    assert error["status"] == "error"
    for token in error_tokens:
        assert token in error["message"], error
    assert not (output_dir / "report.json").exists()
    assert not (output_dir / "pipeline_description.json").exists()


def test_full_duplex_catalog_discovery_and_generated_atlas() -> None:
    from scripts.generate_pipeline_catalog import iter_atomic_pipeline_ids

    expected_ids = {
        f"full_duplex.any.{metric}.duplex_timeline_v1.fdbench_timing_v1" for metric in _METRICS
    }
    discovered = {
        pipeline_id for task, pipeline_id in iter_atomic_pipeline_ids() if task == "full_duplex"
    }
    assert discovered == expected_ids
    root = Path(__file__).resolve().parents[1]
    rows = [
        json.loads(line)
        for line in (root / "docs/pipeline_catalog.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    generated = [row for row in rows if row["task_alias"] == "full_duplex"]
    assert {row["pipeline_id"] for row in generated} == expected_ids
    assert all(row["pipeline_kind"] == "atomic" for row in generated)
    assert all(row["execution_metrics"] == [row["metric"]] for row in generated)
    for filename in ("index.html", "pipeline_atlas.svg", "pipeline_atlas_dark.svg"):
        atlas = (root / "docs/atlas" / filename).read_text(encoding="utf-8")
        assert "FULL_DUPLEX" in atlas
