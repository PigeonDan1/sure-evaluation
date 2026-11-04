from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

_METRICS = ("ird", "fsed", "ert", "eit")
_NODES = ("normalization/duplex_timeline", "scoring/fdbench_timing")


def _fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    shared = {
        "schema_version": 1,
        "key": "case-A",
        "time_unit": "seconds",
        "time_origin": "conversation_start",
    }
    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    reference.write_text(
        json.dumps(
            {
                **shared,
                "duration": 10,
                "user_segments": [
                    {"id": "u1", "start": 1, "end": 3},
                    {"id": "u2", "start": 6, "end": 8},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prediction.write_text(
        json.dumps(
            {
                **shared,
                "system_segments": [
                    {"id": "a1", "start": 2.75, "end": 6.25},
                    {"id": "a2", "start": 8.4, "end": 9},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return reference, prediction


@pytest.mark.parametrize("metric", _METRICS)
def test_full_duplex_describe_atomic_selector_and_contract(metric: str) -> None:
    from sure_eval.evaluation.scripts import describe_pipeline
    from sure_eval.evaluation.tasks.full_duplex.pipeline import pipeline_id_for_metric

    description = describe_pipeline("full_duplex", metric=metric)
    assert description.task == "FULL_DUPLEX"
    assert description.language == "n/a"
    assert description.metric == metric
    assert description.pipeline_id == pipeline_id_for_metric(metric)
    assert (
        description.pipeline_id == f"full_duplex.any.{metric}.duplex_timeline_v1.fdbench_timing_v1"
    )
    assert description.pipeline_kind == "atomic"
    assert description.member_pipeline_ids == ()
    assert description.node_ids == _NODES
    assert description.computation_node_ids == _NODES
    assert description.execution_metrics == (metric,)
    assert description.required_roles == ("reference_jsonl", "sample_output")
    assert description.optional_roles == ()
    assert description.contracts[0]["aggregation"] == "pooled_event_median_ms"


def test_full_duplex_default_metric_matches_manifest_and_adapter() -> None:
    from sure_eval.evaluation.cli_adapters import build_pipeline_spec
    from sure_eval.evaluation.scripts import describe_pipeline
    from sure_eval.evaluation.scripts.contracts import load_task_manifest

    manifest, _ = load_task_manifest("full_duplex")
    assert manifest["default_metric"] == "ird"
    assert describe_pipeline("full_duplex").metric == "ird"
    assert build_pipeline_spec("full_duplex")["requested_metric"] == "ird"


@pytest.mark.parametrize("metric,score", [("ird", 250), ("fsed", 400), ("ert", 250), ("eit", 0)])
def test_full_duplex_run_preserves_selected_route_and_serializable_results(
    tmp_path: Path, metric: str, score: float
) -> None:
    from sure_eval.evaluation.scripts import run_task

    reference, prediction = _fixture_files(tmp_path)
    pipeline_id = f"full_duplex.any.{metric}.duplex_timeline_v1.fdbench_timing_v1"
    output_dir = tmp_path / "output"
    report = run_task(
        "full_duplex",
        pipeline_id=pipeline_id,
        reference_jsonl=str(reference),
        sample_output=str(prediction),
        output_dir=str(output_dir),
    )
    assert report.task == "FULL_DUPLEX"
    assert report.metric == metric
    assert report.score == score
    assert report.pipeline_id == pipeline_id
    assert report.computation_node_ids == _NODES
    assert [(node.node_id, node.version) for node in report.pipeline_trace] == [
        (_NODES[0], "v1"),
        (_NODES[1], "v1"),
    ]
    assert set(report.details["results"]) == set(_METRICS)
    assert report.details["results"][metric]["score"] == report.score
    assert report.details["results"]["eit"]["sample_count"] == 0
    assert report.details["results"]["eit"]["status"] == "no_events"
    assert report.details["input_sha256"] == {
        "reference_jsonl": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "sample_output": hashlib.sha256(prediction.read_bytes()).hexdigest(),
    }
    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    description = json.loads((output_dir / "pipeline_description.json").read_text(encoding="utf-8"))
    assert payload["score"] == score
    assert payload["pipeline_id"] == description["pipeline_id"] == pipeline_id
    assert description["execution_metrics"] == [metric]
    assert (
        payload["details"]["evaluation_config"]
        == payload["pipeline_trace"][1]["details"]["evaluation_config"]
    )
    assert payload["pipeline_trace"][1]["details"]["protocol"] == "fdbench_8a4b7df_compat"


def test_full_duplex_report_keeps_large_diagnostics_once_and_compact_traces(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts import run_task
    from sure_eval.evaluation.scripts.contracts import load_task_manifest

    fixtures = Path(__file__).parents[2] / "fixtures" / "full_duplex"
    output_dir = tmp_path / "output"
    run_task(
        "full_duplex",
        reference_jsonl=str(fixtures / "reference.jsonl"),
        sample_output=str(fixtures / "sample_output.jsonl"),
        output_dir=str(output_dir),
    )
    report_text = (output_dir / "report.json").read_text(encoding="utf-8")
    payload = json.loads(report_text)
    details = payload["details"]
    manifest, _ = load_task_manifest("full_duplex")
    assert set(details) == set(manifest["reported_fields"]) == {
        "results", "evaluation_config", "input_sha256", "per_conversation", "events",
        "measurement_protocol", "input_contract", "input_files",
    }
    assert report_text.count('"per_conversation":') == 1
    assert report_text.count('"events":') == 1
    assert report_text.count('"raw_slots_ticks":') == len(details["per_conversation"])
    assert report_text.count('"event_id":') == len(details["events"])
    assert details["measurement_protocol"]["status"] == "missing"
    assert "conversations" not in payload["pipeline_trace"][0]["details"]["input_summary"]

    trace = payload["pipeline_trace"][1]
    assert trace["node_id"] == _NODES[1]
    assert trace["version"] == "v1"
    assert set(trace["details"]) == {
        "score", "metric", "unit", "protocol", "evaluation_config", "conversation_count",
        "user_segment_count", "system_segment_count", "event_count", "results",
    }
    assert trace["details"]["conversation_count"] == len(details["per_conversation"])
    assert trace["details"]["event_count"] == len(details["events"])
    expected = json.loads((fixtures / "expected.json").read_text(encoding="utf-8"))
    for metric in _METRICS:
        result = details["results"][metric]
        assert result["score"] == expected["pooled"][metric]["score"]
        assert result["sample_count"] == expected["pooled"][metric]["sample_count"]
        for name, value in trace["details"]["results"][metric].items():
            assert value == result[name]
            assert not isinstance(value, (dict, list))
        assert set(trace["details"]["results"][metric]) == {
            "score", "unit", "sample_count", "positive_tick_count", "zero_ms_count",
            "status", "reason",
        }
    for row in details["per_conversation"]:
        assert row["samples_ms"] == expected["per_conversation"][row["key"]]


def test_full_duplex_report_preserves_metadata_and_measurement_protocol_once(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts import run_task

    reference, prediction = _fixture_files(tmp_path)
    protocol = {
        "vad": {
            "name": "synthetic_vad", "version": "1", "threshold": 0.5,
            "min_silence_ms": 100, "padding_ms": 30,
        },
        "playback": {
            "queue_policy": "retain_all_received_audio", "clock": "shared_monotonic",
            "speaker_output_observed": False,
        },
        "reference": {"source": "human_annotation"},
        "preamble": {"removed": False, "reason": "kept"},
        "notes": "protocol-provenance-once",
    }
    for path, metadata in (
        (reference, {
            "marker": "reference-metadata-once", "nested": {"offset": 0.125},
            "measurement_protocol": protocol,
        }),
        (prediction, {"marker": "prediction-metadata-once"}),
    ):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["metadata"] = metadata
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    output_dir = tmp_path / "output"
    report = run_task(
        "full_duplex", reference_jsonl=str(reference), sample_output=str(prediction),
        output_dir=str(output_dir),
    )
    report_text = (output_dir / "report.json").read_text(encoding="utf-8")
    details = json.loads(report_text)["details"]
    assert report.score == 250
    assert details["measurement_protocol"]["status"] == "recorded"
    assert details["measurement_protocol"]["conversations"]["case-A"]["reference"] == protocol
    assert details["per_conversation"][0]["metadata"] == {
        "reference": {"marker": "reference-metadata-once", "nested": {"offset": "0.125"}},
        "prediction": {"marker": "prediction-metadata-once"},
    }
    for marker in (
        "reference-metadata-once", "prediction-metadata-once", "protocol-provenance-once",
    ):
        assert report_text.count(marker) == 1


def test_full_duplex_contract_protocol_and_route_declarations_agree() -> None:
    from sure_eval.evaluation.core.types import MetricInputContract
    from sure_eval.evaluation.nodes.scoring.fdbench_timing import PROTOCOL, SUPPORTED_METRICS
    from sure_eval.evaluation.scripts import describe_pipeline
    from sure_eval.evaluation.scripts.contracts import load_task_manifest, load_task_routes
    from sure_eval.evaluation.tasks.full_duplex.pipeline import (
        _CONTRACT,
        _PROTOCOL,
        evaluate_full_duplex_files,
        pipeline_id_for_metric,
    )

    manifest, _ = load_task_manifest("full_duplex")
    configured, _ = load_task_routes("full_duplex")
    contract_key = "full_duplex_jsonl"
    contract = manifest["input_contracts"][contract_key]
    assert MetricInputContract(**{
        **contract,
        "required_roles": tuple(contract["required_roles"]),
        "optional_roles": tuple(contract["optional_roles"]),
    }) == _CONTRACT
    assert _PROTOCOL == PROTOCOL
    assert inspect.signature(evaluate_full_duplex_files).parameters["protocol"].default == PROTOCOL
    assert tuple(manifest["metrics"]) == SUPPORTED_METRICS == _METRICS
    assert set(manifest["routes"]) == set(_METRICS)
    assert len(configured["routes"]) == len(_METRICS)
    assert {route["metric"] for route in configured["routes"]} == set(_METRICS)
    for route in configured["routes"]:
        metric = route["metric"]
        expected_id = f"full_duplex.any.{metric}.duplex_timeline_v1.fdbench_timing_v1"
        assert route["pipeline_id"] == manifest["routes"][metric]["pipeline_id"] == expected_id
        assert pipeline_id_for_metric(metric) == expected_id
        assert route["input_contract"] == contract_key
        assert route["protocol"] == PROTOCOL
        assert tuple(route["nodes"]) == tuple(route["computation_nodes"]) == _NODES
        assert manifest["routes"][metric]["normalization"] == _NODES[0]
        assert manifest["routes"][metric]["scoring"] == _NODES[1]
        description = describe_pipeline("full_duplex", pipeline_id=expected_id)
        assert description.contracts == (contract,)
        assert description.execution_metrics == (metric,)


@pytest.mark.parametrize("metric,expected_metric,score", [("fsed", "fsed", 400), (None, "ird", 250)])
def test_full_duplex_run_metric_selector_and_default_preserve_route(
    tmp_path: Path, metric: str | None, expected_metric: str, score: float
) -> None:
    from sure_eval.evaluation.scripts import run_task

    reference, prediction = _fixture_files(tmp_path)
    output_dir = tmp_path / "output"
    selection = {"metric": metric} if metric is not None else {}
    report = run_task(
        "full_duplex",
        reference_jsonl=str(reference),
        sample_output=str(prediction),
        output_dir=str(output_dir),
        **selection,
    )
    pipeline_id = f"full_duplex.any.{expected_metric}.duplex_timeline_v1.fdbench_timing_v1"
    assert report.metric == expected_metric
    assert report.score == score
    assert report.pipeline_id == pipeline_id
    assert report.pipeline_kind == "atomic"
    assert report.member_pipeline_ids == ()
    assert report.computation_node_ids == _NODES
    payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    description = json.loads((output_dir / "pipeline_description.json").read_text(encoding="utf-8"))
    assert payload["metric"] == description["metric"] == expected_metric
    assert payload["pipeline_id"] == description["pipeline_id"] == pipeline_id
    assert payload["pipeline_kind"] == description["pipeline_kind"] == "atomic"
    assert payload["member_pipeline_ids"] == description["member_pipeline_ids"] == []
    assert payload["computation_node_ids"] == description["computation_node_ids"] == list(_NODES)
    assert description["execution_metrics"] == [expected_metric]
    assert payload["score"] == payload["details"]["results"][expected_metric]["score"] == score


@pytest.mark.parametrize("metric", ["wer", "srr", "ird,fsed", "unknown"])
def test_full_duplex_rejects_unsupported_primary_metric(metric: str) -> None:
    from sure_eval.evaluation.scripts import describe_pipeline

    with pytest.raises(ValueError, match="Unsupported FULL_DUPLEX metric"):
        describe_pipeline("full_duplex", metric=metric)


@pytest.mark.parametrize("role", ["reference_jsonl", "sample_output"])
def test_full_duplex_missing_roles_are_contract_errors(role: str) -> None:
    from sure_eval.evaluation.tasks.full_duplex import evaluate_full_duplex_files

    with pytest.raises(ValueError, match="Missing required evaluation input role"):
        evaluate_full_duplex_files(**{role: "unused.jsonl"})


def test_full_duplex_rejects_unregistered_protocol() -> None:
    from sure_eval.evaluation.tasks.full_duplex import evaluate_full_duplex_files

    with pytest.raises(ValueError, match="Unsupported FULL_DUPLEX protocol"):
        evaluate_full_duplex_files(protocol="intuitive")


def test_full_duplex_node_environments_are_in_process() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker
    from sure_eval.evaluation.scripts.contracts import NODES_ROOT, load_yaml

    for node_id in _NODES:
        environment = load_yaml(NODES_ROOT / node_id / "node_env.yaml")
        assert environment["runtime"]["type"] == "in_process"
        assert environment["runtime"]["gpu"] is False
        assert NodeEnvChecker().check_node(node_id).status == "ok"
