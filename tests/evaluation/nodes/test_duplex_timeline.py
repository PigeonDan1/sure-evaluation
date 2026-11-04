"""Input-contract regressions: never silently repair the physical conversation clock."""

from __future__ import annotations

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

from sure_eval.evaluation.nodes.normalization.duplex_timeline import normalize_duplex_timeline
from sure_eval.evaluation.nodes.normalization.duplex_timeline.measurement_protocol import (
    summarize_measurement_protocol,
)
from sure_eval.evaluation.nodes.normalization.duplex_timeline.types import DuplexTimelineBundle
from sure_eval.evaluation.nodes.scoring.fdbench_timing import score_fdbench_timing


def _rows():
    common = {
        "schema_version": 1,
        "key": "case",
        "time_unit": "seconds",
        "time_origin": "conversation_start",
    }
    return (
        {
            **common,
            "duration": 10,
            "user_segments": [{"id": "u1", "start": 1, "end": 3}],
        },
        {**common, "system_segments": [{"id": "a1", "start": 2.75, "end": 4}]},
    )


def _write(tmp_path: Path, references, predictions):
    paths = tmp_path / "ref.jsonl", tmp_path / "pred.jsonl"
    for path, rows in zip(paths, (references, predictions)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return paths


def test_cross_channel_overlap_and_decimal_precision_are_preserved(tmp_path):
    ref, pred = _rows()
    paths = _write(tmp_path, [ref], [pred])
    paths[1].write_text(
        paths[1].read_text(encoding="utf-8").replace("2.75", "2.7499999999999999999999999999999"),
        encoding="utf-8",
    )
    bundle, trace = normalize_duplex_timeline(*paths)
    row = bundle.conversations[0]
    assert row.system_segments[0].start == Decimal("2.7499999999999999999999999999999")
    assert row.user_segments[0].end > row.system_segments[0].start
    assert trace.node_id == "normalization/duplex_timeline"
    assert trace.version == "v1"
    json.dumps(bundle.as_dict(), allow_nan=False)


def test_file_order_is_aligned_by_key_but_segments_are_not_sorted(tmp_path):
    ref, pred = _rows()
    ref_b, pred_b = copy.deepcopy(ref), copy.deepcopy(pred)
    ref_b["key"], pred_b["key"] = "b", "b"
    bundle, _ = normalize_duplex_timeline(*_write(tmp_path, [ref_b, ref], [pred, pred_b]))
    assert [row.key for row in bundle.conversations] == ["b", "case"]


def test_explicit_silence_and_adjacent_segments_are_valid(tmp_path):
    ref, pred = _rows()
    ref["user_segments"].append({"id": "u2", "start": 3, "end": 4})
    pred["system_segments"] = []
    bundle, _ = normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))
    assert bundle.conversations[0].system_segments == ()
    assert len(bundle.conversations[0].user_segments) == 2


@pytest.mark.parametrize("role", ["reference", "prediction"])
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema_version", True, "schema_version"),
        ("schema_version", 1.0, "schema_version"),
        ("schema_version", 2, "schema_version"),
        ("key", "  ", "non-empty"),
        ("time_unit", "milliseconds", "time_unit"),
        ("time_origin", "wall_clock", "time_origin"),
        ("metadata", [], "metadata"),
        ("unexpected", 1, "unknown"),
    ],
)
def test_strict_row_contract(tmp_path, role, field, value, error):
    ref, pred = _rows()
    (ref if role == "reference" else pred)[field] = value
    with pytest.raises(ValueError, match=error):
        normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))


@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf")])
def test_non_numbers_are_not_timestamps(tmp_path, value):
    ref, pred = _rows()
    pred["system_segments"][0]["start"] = value
    with pytest.raises(ValueError, match="finite"):
        normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))


@pytest.mark.parametrize("duration", [0, -1, True, "10", None])
def test_duration_is_positive_finite_number(tmp_path, duration):
    ref, pred = _rows()
    ref["duration"] = duration
    with pytest.raises(ValueError, match="duration"):
        normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))


@pytest.mark.parametrize(
    ("segments", "error"),
    [
        ([{"id": "a", "start": -1, "end": 2}], "0 <= start"),
        ([{"id": "a", "start": 2, "end": 2}], "0 <= start"),
        ([{"id": "a", "start": 2, "end": 11}], "0 <= start"),
        ([{"id": "", "start": 2, "end": 3}], "non-empty"),
        ([{"id": "a", "start": 2}], "missing required"),
        ([{"id": "a", "start": 2, "end": 3, "text": "x"}], "unknown"),
        ([{"id": "a", "start": 4, "end": 5}, {"id": "b", "start": 2, "end": 3}], "increasing"),
        ([{"id": "a", "start": 2, "end": 4}, {"id": "b", "start": 3, "end": 5}], "overlap"),
        ([{"id": "a", "start": 2, "end": 3}, {"id": "a", "start": 4, "end": 5}], "duplicate"),
        ({}, "array"),
        ([None], "object"),
    ],
)
def test_invalid_channel_segments_are_rejected(tmp_path, segments, error):
    ref, pred = _rows()
    pred["system_segments"] = segments
    with pytest.raises(ValueError, match=error):
        normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))


def test_missing_extra_duplicate_and_empty_rows_are_errors(tmp_path):
    ref, pred = _rows()
    for references, predictions, error in [
        ([ref], [{**pred, "key": "other"}], "key mismatch"),
        ([ref], [pred, pred], "duplicate conversation"),
        ([ref, ref], [pred], "duplicate conversation"),
        ([ref], [], "at least one"),
        ([], [pred], "at least one"),
        ([{**ref, "user_segments": []}], [pred], "must not be empty"),
    ]:
        with pytest.raises(ValueError, match=error):
            normalize_duplex_timeline(*_write(tmp_path, references, predictions))


def test_duplicate_json_fields_are_rejected_instead_of_last_value_wins(tmp_path):
    ref, pred = _rows()
    paths = _write(tmp_path, [ref], [pred])
    paths[1].write_text('{"key":"one","key":"two"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        normalize_duplex_timeline(*paths)


def test_metadata_is_extensible_json_safe_and_does_not_change_segments(tmp_path):
    ref, pred = _rows()
    ref["metadata"] = {"dataset": "synthetic", "nested": {"threshold": 0.125}}
    pred["metadata"] = {"model_id": "offline-fixture", "preamble_removed": False}
    bundle, _ = normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))
    row = bundle.as_dict()["conversations"][0]
    assert row["metadata"]["reference"]["nested"]["threshold"] == "0.125"
    assert row["metadata"]["prediction"]["preamble_removed"] is False
    assert row["system_segments"] == [{"id": "a1", "start": "2.75", "end": "4"}]


@pytest.mark.parametrize("channel", ["user_segments", "system_segments"])
@pytest.mark.parametrize(
    ("segments", "keyword", "segment_ids"),
    [
        ([{"id": "outside", "start": -1, "end": 2}], "0 <= start", ["outside"]),
        ([{"id": "after-end", "start": 9, "end": 11}], "0 <= start", ["after-end"]),
        ([{"id": "zero", "start": 2, "end": 2}], "0 <= start", ["zero"]),
        ([{"id": "reverse", "start": 3, "end": 2}], "0 <= start", ["reverse"]),
        (
            [{"id": "previous", "start": 4, "end": 5}, {"id": "current", "start": 2, "end": 3}],
            "increasing",
            ["current", "previous"],
        ),
        (
            [{"id": "previous", "start": 2, "end": 4}, {"id": "current", "start": 3, "end": 5}],
            "overlap",
            ["current", "previous"],
        ),
    ],
)
def test_segment_errors_locate_conversation_channel_and_ids(
    tmp_path, channel, segments, keyword, segment_ids
):
    ref, pred = _rows()
    (ref if channel == "user_segments" else pred)[channel] = segments
    with pytest.raises(ValueError, match=keyword) as caught:
        normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))
    message = str(caught.value)
    assert "'case'" in message
    assert channel in message
    assert "segment_id=" in message
    for segment_id in segment_ids:
        assert repr(segment_id) in message


@pytest.mark.parametrize(
    ("role", "field"),
    [
        ("reference", "duration"),
        ("reference", "user_segments"),
        ("prediction", "system_segments"),
        ("reference", "key"),
        ("prediction", "key"),
    ],
)
def test_missing_fields_locate_jsonl_file_and_line(tmp_path, role, field):
    ref, pred = _rows()
    del (ref if role == "reference" else pred)[field]
    paths = _write(tmp_path, [ref], [pred])
    with pytest.raises(ValueError, match="missing required") as caught:
        normalize_duplex_timeline(*paths)
    path = paths[0 if role == "reference" else 1]
    assert f"{path}:1" in str(caught.value)
    assert field in str(caught.value)


@pytest.mark.parametrize("value", ["{bad", "NaN", "Infinity", "-Infinity"])
def test_malformed_json_or_nonfinite_duration_locates_source(tmp_path, value):
    ref, pred = _rows()
    paths = _write(tmp_path, [ref], [pred])
    text = value if value == "{bad" else paths[0].read_text(encoding="utf-8").replace(
        '"duration": 10', f'"duration": {value}'
    )
    paths[0].write_text(text + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        normalize_duplex_timeline(*paths)
    assert f"{paths[0]}:1" in str(caught.value)
    keyword = "Expecting property" if value == "{bad" else "non-finite"
    assert keyword in str(caught.value)


def test_missing_file_preserves_filename_in_error(tmp_path):
    missing = tmp_path / "missing-reference.jsonl"
    with pytest.raises(FileNotFoundError) as caught:
        normalize_duplex_timeline(missing, tmp_path / "prediction.jsonl")
    assert caught.value.filename == str(missing)
    assert missing.name in str(caught.value)


def _complete_measurement_protocol():
    return {
        "vad": {
            "name": "fixture-vad",
            "version": "1",
            "threshold": 0.6,
            "min_silence_ms": 120.5,
            "padding_ms": 0,
        },
        "playback": {
            "queue_policy": "retain_all_received_audio",
            "clock": "speaker_output",
            "speaker_output_observed": True,
        },
        "reference": {"source": "human_annotation"},
        "preamble": {"removed": False, "reason": "no preamble present"},
    }


def test_measurement_protocol_preserves_both_roles_and_numeric_report_values(tmp_path):
    ref, pred = _rows()
    reference_protocol = _complete_measurement_protocol()
    reference_protocol["capture"] = {
        "model_id": "fixture-model",
        "endpoint": "fixture://offline",
        "sample_rates": {"user": 16000, "system": 24000},
    }
    reference_protocol["notes"] = "synthetic recording"
    prediction_protocol = _complete_measurement_protocol()
    prediction_protocol["vad"]["threshold"] = 0.2
    prediction_protocol["playback"]["queue_policy"] = "flush_on_server_speech_started"
    prediction_protocol["reference"]["source"] = "upstream_machine_schedule"
    ref["metadata"] = {
        "measurement_protocol": reference_protocol,
        "unrelated": {"threshold": 0.125},
    }
    pred["metadata"] = {"measurement_protocol": prediction_protocol}
    original = copy.deepcopy((ref, pred))
    bundle, _ = normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))
    before = copy.deepcopy(bundle.conversations[0].metadata)
    report = summarize_measurement_protocol(bundle)
    assert report == {
        "status": "recorded",
        "conversations": {
            "case": {
                "status": "recorded",
                "reference": reference_protocol,
                "prediction": prediction_protocol,
            }
        },
    }
    assert type(report["conversations"]["case"]["reference"]["vad"]["threshold"]) is float
    assert type(report["conversations"]["case"]["reference"]["vad"]["padding_ms"]) is int
    assert bundle.conversations[0].metadata == before
    assert (ref, pred) == original
    assert bundle.as_dict()["conversations"][0]["metadata"]["reference"]["unrelated"] == {
        "threshold": "0.125"
    }
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("role", ["reference", "prediction"])
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("", None, "object"),
        ("", [], "object"),
        ("", "detector", "object"),
        ("vad", [], "object"),
        ("playback", None, "object"),
        ("reference", "human_annotation", "object"),
        ("preamble", True, "object"),
        ("capture", 16000, "object"),
        ("vad.name", 1, "string"),
        ("vad.version", " ", "string"),
        ("vad.threshold", True, "finite"),
        ("vad.threshold", "0.5", "finite"),
        ("vad.threshold", -0.01, "between"),
        ("vad.threshold", 1.01, "between"),
        ("vad.min_silence_ms", False, "finite"),
        ("vad.min_silence_ms", -1, "nonnegative"),
        ("vad.padding_ms", [], "finite"),
        ("vad.padding_ms", -0.5, "nonnegative"),
        ("playback.queue_policy", "drop_audio", "one of"),
        ("playback.clock", False, "string"),
        ("playback.speaker_output_observed", 1, "boolean"),
        ("reference.source", "model_guess", "one of"),
        ("preamble.removed", "false", "boolean"),
        ("preamble.reason", None, "string"),
        ("capture.model_id", "", "string"),
        ("capture.endpoint", [], "string"),
        ("capture.sample_rates", [], "object"),
        ("capture.sample_rates.user", True, "positive integer"),
        ("capture.sample_rates.user", 16000.0, "positive integer"),
        ("capture.sample_rates.user", 0, "positive integer"),
        ("capture.sample_rates.user", -16000, "positive integer"),
        ("capture.sample_rates.", 16000, "non-empty string"),
        ("notes", [], "string"),
        ("unexpected", 1, "unknown fields"),
        ("vad.unexpected", 1, "unknown fields"),
    ],
)
def test_invalid_measurement_protocol_locates_file_key_and_field(
    tmp_path, role, field, value, error
):
    ref, pred = _rows()
    protocol = {}
    if field:
        target = protocol
        parts = field.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    else:
        protocol = value
    (ref if role == "reference" else pred)["metadata"] = {"measurement_protocol": protocol}
    paths = _write(tmp_path, [ref], [pred])
    with pytest.raises(ValueError, match=error) as caught:
        normalize_duplex_timeline(*paths)
    message = str(caught.value)
    assert f"{paths[0 if role == 'reference' else 1]}:1" in message
    assert "'case'" in message
    assert "metadata.measurement_protocol" in message
    if field:
        # Unknown fields are named in a list after their containing object.
        assert all(part in message for part in field.split(".") if part)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["threshold", "min_silence_ms", "padding_ms"])
def test_nonfinite_measurement_numbers_locate_nested_field(tmp_path, field, value):
    ref, pred = _rows()
    pred["metadata"] = {"measurement_protocol": {"vad": {field: value}}}
    paths = _write(tmp_path, [ref], [pred])
    with pytest.raises(ValueError, match="finite") as caught:
        normalize_duplex_timeline(*paths)
    message = str(caught.value)
    assert f"{paths[1]}:1" in message
    assert "'case'" in message
    assert f"metadata.measurement_protocol.vad.{field}" in message


@pytest.mark.parametrize(
    ("field", "literal", "error"),
    [
        ("min_silence_ms", "1" + "0" * 400 + ".1", "finite JSON number"),
        ("threshold", "1e-1000", "underflow"),
        ("padding_ms", "1" + "0" * 400, "finite JSON number"),
        ("min_silence_ms", "1.1e1000", "finite JSON number"),
    ],
)
def test_unrepresentable_measurement_numbers_are_rejected_before_report(
    tmp_path, field, literal, error
):
    ref, pred = _rows()
    ref["metadata"] = {"measurement_protocol": {"vad": {field: "literal"}}}
    paths = _write(tmp_path, [ref], [pred])
    paths[0].write_text(
        paths[0].read_text(encoding="utf-8").replace('"literal"', literal),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=error) as caught:
        normalize_duplex_timeline(*paths)
    message = str(caught.value)
    assert f"{paths[0]}:1" in message
    assert "'case'" in message
    assert f"metadata.measurement_protocol.vad.{field}" in message


def test_representable_measurement_integers_keep_exact_json_value(tmp_path):
    ref, pred = _rows()
    exact = 9007199254740993
    ref["metadata"] = {"measurement_protocol": {"vad": {"min_silence_ms": exact}}}
    paths = _write(tmp_path, [ref], [pred])
    for literal in (str(exact), f"{exact}.0"):
        paths[0].write_text(
            json.dumps(ref).replace(str(exact), literal) + "\n", encoding="utf-8"
        )
        bundle, _ = normalize_duplex_timeline(*paths)
        protocol = summarize_measurement_protocol(bundle)["conversations"]["case"]["reference"]
        assert type(protocol["vad"]["min_silence_ms"]) is int
        assert protocol["vad"]["min_silence_ms"] == exact
        json.dumps(protocol, allow_nan=False)


def test_nonfinite_measurement_fallback_still_rejects_duplicate_json_fields(tmp_path):
    ref, pred = _rows()
    ref["metadata"] = {"measurement_protocol": {"vad": {"threshold": "literal"}}}
    paths = _write(tmp_path, [ref], [pred])
    paths[0].write_text(
        paths[0].read_text(encoding="utf-8").replace(
            '"threshold": "literal"', '"threshold": NaN, "threshold": 0.5'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON field 'threshold'") as caught:
        normalize_duplex_timeline(*paths)
    assert f"{paths[0]}:1" in str(caught.value)


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        (None, None, "missing"),
        ({}, None, "partial"),
        (None, {}, "partial"),
        ({"notes": "only notes"}, {}, "partial"),
        ({"vad": {"threshold": 0}}, {"reference": {"source": "other"}}, "partial"),
        (_complete_measurement_protocol(), None, "recorded"),
        ({}, _complete_measurement_protocol(), "recorded"),
    ],
)
def test_measurement_protocol_status_allows_missing_and_partial_objects(
    tmp_path, reference, prediction, expected
):
    ref, pred = _rows()
    for row, protocol in ((ref, reference), (pred, prediction)):
        if protocol is not None:
            row["metadata"] = {"measurement_protocol": protocol}
    bundle, _ = normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))
    report = summarize_measurement_protocol(bundle)
    assert report["status"] == expected
    assert report["conversations"]["case"] == {
        "status": expected,
        "reference": reference,
        "prediction": prediction,
    }


def test_measurement_protocol_completeness_requires_each_documented_field(tmp_path):
    for section in ("vad", "playback", "reference", "preamble"):
        for field in _complete_measurement_protocol()[section]:
            ref, pred = _rows()
            protocol = _complete_measurement_protocol()
            del protocol[section][field]
            ref["metadata"] = {"measurement_protocol": protocol}
            bundle, _ = normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))
            assert summarize_measurement_protocol(bundle)["status"] == "partial", (section, field)


def test_measurement_protocol_completeness_does_not_merge_roles(tmp_path):
    ref, pred = _rows()
    complete = _complete_measurement_protocol()
    ref["metadata"] = {
        "measurement_protocol": {name: complete[name] for name in ("vad", "playback")}
    }
    pred["metadata"] = {
        "measurement_protocol": {name: complete[name] for name in ("reference", "preamble")}
    }
    bundle, _ = normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))
    assert summarize_measurement_protocol(bundle)["status"] == "partial"


@pytest.mark.parametrize("second_status", ["missing", "partial", "recorded"])
def test_measurement_protocol_bundle_status_reports_incomplete_coverage(tmp_path, second_status):
    ref, pred = _rows()
    second_ref, second_pred = copy.deepcopy(ref), copy.deepcopy(pred)
    second_ref["key"] = second_pred["key"] = "second"
    ref["metadata"] = {"measurement_protocol": _complete_measurement_protocol()}
    if second_status != "missing":
        second_pred["metadata"] = {
            "measurement_protocol": (
                _complete_measurement_protocol() if second_status == "recorded" else {}
            )
        }
    bundle, _ = normalize_duplex_timeline(
        *_write(tmp_path, [ref, second_ref], [pred, second_pred])
    )
    report = summarize_measurement_protocol(bundle)
    assert report["status"] == ("recorded" if second_status == "recorded" else "partial")
    assert report["conversations"]["second"]["status"] == second_status


def test_empty_measurement_protocol_bundle_is_missing():
    assert summarize_measurement_protocol(DuplexTimelineBundle(())) == {
        "status": "missing",
        "conversations": {},
    }


def test_measurement_protocol_accepts_boundary_values_and_other_categories(tmp_path):
    ref, pred = _rows()
    protocol = _complete_measurement_protocol()
    protocol["vad"].update(threshold=1, min_silence_ms=0, padding_ms=0.25)
    protocol["playback"]["queue_policy"] = "other"
    protocol["reference"]["source"] = "other"
    protocol["notes"] = ""
    ref["metadata"] = {"measurement_protocol": protocol}
    bundle, _ = normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))
    assert summarize_measurement_protocol(bundle)["conversations"]["case"]["reference"] == protocol


def test_measurement_metadata_does_not_change_six_arrays_or_four_scores(tmp_path):
    ref, pred = _rows()
    ref["user_segments"].append({"id": "u2", "start": 6, "end": 8})
    pred["system_segments"] = [
        {"id": "a1", "start": 2.75, "end": 6.25},
        {"id": "a2", "start": 8.4, "end": 9},
    ]
    plain_bundle, _ = normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))
    plain = score_fdbench_timing(plain_bundle).details
    ref["metadata"] = {"measurement_protocol": _complete_measurement_protocol()}
    pred["metadata"] = {"measurement_protocol": {"preamble": {"removed": True}}}
    recorded_bundle, _ = normalize_duplex_timeline(*_write(tmp_path, [ref], [pred]))
    recorded = score_fdbench_timing(recorded_bundle).details
    expected_arrays = plain["per_conversation"][0]["raw_slots_ticks"]
    assert len(expected_arrays) == 6
    assert recorded["per_conversation"][0]["raw_slots_ticks"] == expected_arrays
    assert len(plain["results"]) == 4
    assert recorded["results"] == plain["results"]
    assert recorded_bundle.conversations[0].user_segments == plain_bundle.conversations[0].user_segments
    assert (
        recorded_bundle.conversations[0].system_segments
        == plain_bundle.conversations[0].system_segments
    )
