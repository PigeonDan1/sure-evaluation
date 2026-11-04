"""Independent hand examples and frozen upstream FD-Bench timing regressions."""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path
from statistics import median

import pytest

from sure_eval.evaluation.nodes.normalization.duplex_timeline.types import (
    ConversationTimeline,
    DuplexTimelineBundle,
    SpeechSegment,
)
from sure_eval.evaluation.nodes.normalization.duplex_timeline import normalize_duplex_timeline
from sure_eval.evaluation.nodes.scoring.fdbench_timing import (
    PROTOCOL,
    SUPPORTED_METRICS,
    score_fdbench_timing,
)


_ORACLE_FIXTURE = json.loads(
    (Path(__file__).parents[2] / "fixtures" / "full_duplex" / "oracle_cases.json").read_text(
        encoding="utf-8"
    )
)


def _conversation(users, systems, *, key="case", duration="20", metadata=None):
    def segments(items, prefix):
        return tuple(
            SpeechSegment(f"{prefix}{index + 1}", Decimal(str(start)), Decimal(str(end)))
            for index, (start, end) in enumerate(items)
        )

    return ConversationTimeline(
        key=key,
        duration=Decimal(duration),
        user_segments=segments(users, "u"),
        system_segments=segments(systems, "a"),
        metadata=metadata or {},
    )


def _score(users, systems, **kwargs):
    return score_fdbench_timing(DuplexTimelineBundle((_conversation(users, systems),)), **kwargs)


def test_frozen_oracle_provenance_and_case_count():
    provenance = _ORACLE_FIXTURE["provenance"]
    assert provenance["source_commit"] == "8a4b7df1b4dcb0fc50a7a5660247a2ffe8d394eb"
    assert provenance["source_sha256_lf"] == (
        "86a515a2c86fcd9ae89da9c96b2a4dd461b5d41a68dc752f5d3d719b114ee979"
    )
    assert provenance["protocol"] == PROTOCOL
    assert provenance["generated_on"] == "2026-11-04"
    assert provenance["ticks_per_second"] == 16000
    assert provenance["synthetic_inputs"] is True
    assert 20 <= len(_ORACLE_FIXTURE["cases"]) == provenance["case_count"] <= 40
    assert len({case["id"] for case in _ORACLE_FIXTURE["cases"]}) == len(
        _ORACLE_FIXTURE["cases"]
    )


@pytest.mark.parametrize(
    "case", _ORACLE_FIXTURE["cases"], ids=lambda case: case["id"]
)
def test_frozen_upstream_oracle_arrays_and_results(case, tmp_path):
    # Expected values are frozen upstream AST output, never computed from SURE.
    # The normal test run needs only this synthetic JSON, not an upstream cache.
    inputs = case["input"]
    assert inputs["key"] == case["id"]
    shared = {
        field: inputs[field]
        for field in ("schema_version", "key", "time_unit", "time_origin")
    }
    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "sample_output.jsonl"
    reference.write_text(
        json.dumps(
            {**shared, "duration": inputs["duration"], "user_segments": inputs["user_segments"]}
        )
        + "\n",
        encoding="utf-8",
    )
    prediction.write_text(
        json.dumps({**shared, "system_segments": inputs["system_segments"]}) + "\n",
        encoding="utf-8",
    )
    bundle, _ = normalize_duplex_timeline(reference, prediction)
    details = score_fdbench_timing(bundle).details
    expected = case["expected"]
    assert details["per_conversation"][0]["raw_slots_ticks"] == expected["raw_slots_ticks"]
    assert set(expected["raw_slots_ticks"]) == {
        "Interrupt Delay",
        "Response Delay",
        "Response Delay to Interruption",
        "Lead Times",
        "Lead Times to Interruption",
        "Lead Times Interrupt",
    }
    assert tuple(expected["results"]) == SUPPORTED_METRICS
    for metric, result in expected["results"].items():
        for field in ("score", "sample_count", "status", "samples_ms"):
            assert details["results"][metric][field] == result[field]
            assert details["per_conversation"][0]["results"][metric][field] == result[field]


# Deltas are manually obtained from the endpoints shown here. In particular B
# uses system duration for IRD, F uses the new response end, and H overwrites ERT.
@pytest.mark.parametrize(
    "users,systems,expected",
    [
        (
            [(1, 3), (6, 8)],
            [(2.75, 6.25), (8.4, 9)],
            {"ird": [250], "fsed": [400], "ert": [250], "eit": []},
        ),
        ([(1, 3)], [(2, 2.4)], {"ird": [400], "fsed": [], "ert": [], "eit": [1000]}),
        ([(1, 3)], [(2.5, 4)], {"ird": [], "fsed": [], "ert": [500], "eit": []}),
        ([(1, 3)], [("2.4999375", 4)], {"ird": [], "fsed": [], "ert": [], "eit": [500]}),
        ([(1, 3)], [(1, 4)], {"ird": [], "fsed": [], "ert": [], "eit": []}),
        ([(1, 3)], [(3, 4)], {"ird": [], "fsed": [], "ert": [], "eit": []}),
        ([(1, 3), (8, 9)], [(0, 4), (5, 6)], {"ird": [5000], "fsed": [2000], "ert": [], "eit": []}),
        ([(1, 3)], [(0, 4), (5, 6)], {"ird": [3000], "fsed": [2000], "ert": [], "eit": []}),
        ([(1, 3)], [(2.6, 2.7), (2.8, 4)], {"ird": [], "fsed": [], "ert": [200], "eit": []}),
        ([(1, 3)], [], {"ird": [], "fsed": [], "ert": [], "eit": []}),
        ([(1, 3)], [("3.0000625", 4)], {"ird": [], "fsed": [0], "ert": [], "eit": []}),
    ],
    ids=["A", "B", "C-500ms", "D-500ms-plus-tick", "E-start", "E-end", "F", "G", "H", "I", "J"],
)
def test_hand_calculated_golden(users, systems, expected):
    node = _score(users, systems)
    for metric, samples in expected.items():
        result = node.details["results"][metric]
        assert result["samples_ms"] == samples
        assert result["score"] == (float(median(samples)) if samples else 0.0)
        assert result["sample_count"] == len(samples)
        assert result["status"] == ("ok" if samples else "no_events")
    assert node.node_id == "scoring/fdbench_timing"
    assert node.version == "v1"
    assert node.stage == "scoring"
    json.dumps(node.details, allow_nan=False)


def test_overwritten_ert_candidate_remains_auditable():
    result = _score([(1, 3)], [(2.6, 2.7), (2.8, 4)]).details
    ert_events = [event for event in result["events"] if event["metric"] == "ert"]
    assert [event["action"] for event in ert_events] == ["set", "overwrite"]
    assert [event["selected"] for event in ert_events] == [False, True]
    assert ert_events[0]["exclusion_reason"] == "overwritten_by_later_candidate"
    assert ert_events[1]["raw_tick_delta"] == 3200
    assert ert_events[1]["difference_endpoints"] == ["user.end", "system.start"]
    assert result["per_conversation"][0]["raw_slots_ticks"]["Lead Times"] == [3200]


def test_multiple_eit_values_and_interrupted_early_response():
    details = _score([(1, 3)], [(1.5, 1.7), (2, 2.2), (2.7, 4)]).details
    assert details["results"]["eit"]["samples_ms"] == [1500, 1000]
    assert details["results"]["ird"]["samples_ms"] == [200]
    assert details["results"]["ert"]["samples_ms"] == [300]
    raw = details["per_conversation"][0]["raw_slots_ticks"]
    assert raw["Lead Times Interrupt"] == [[24000, 16000]]
    assert raw["Lead Times to Interruption"] == [4800]
    assert raw["Lead Times"] == [0]


def test_late_ordinary_response_keeps_first_candidate():
    details = _score([(1, 3)], [(3.1, 3.2), (3.5, 4)]).details
    assert details["results"]["fsed"]["samples_ms"] == [100]
    events = [event for event in details["events"] if event["metric"] == "fsed"]
    assert [(event["action"], event["selected"]) for event in events] == [
        ("set", True),
        ("skip", False),
    ]


def test_pooled_median_does_not_average_conversation_medians():
    first = _conversation([(1, 2), (4, 5), (7, 8)], [(2.1, 2.2), (5.2, 5.3), (8.3, 8.4)], key="a")
    second = _conversation([(1, 2)], [(3, 4)], key="b")
    details = score_fdbench_timing(DuplexTimelineBundle((first, second)), metric="fsed").details
    assert details["score"] == 250.0
    assert details["results"]["fsed"]["samples_ms"] == [100, 200, 300, 1000]
    assert [item["results"]["fsed"]["score"] for item in details["per_conversation"]] == [
        200.0,
        1000.0,
    ]


def test_truncation_precedes_even_median():
    bundle = DuplexTimelineBundle(
        (
            _conversation([(1, 2)], [("2.1009375", 3)], key="a"),
            _conversation([(1, 2)], [("2.1010625", 3)], key="b"),
        )
    )
    result = score_fdbench_timing(bundle).details["results"]["fsed"]
    assert result["samples_ticks"] == [1615, 1617]
    assert result["samples_ms"] == [100, 101]
    assert result["score"] == 100.5


def test_positive_submillisecond_sample_differs_from_no_events():
    details = _score([(1, 3)], [("3.0000625", 4)]).details
    measured = details["results"]["fsed"]
    empty = details["results"]["ird"]
    assert measured["score"] == empty["score"] == 0.0
    assert measured["sample_count"] == measured["positive_tick_count"] == 1
    assert measured["zero_ms_count"] == 1
    assert measured["status"] == "ok"
    assert empty["sample_count"] == 0
    assert empty["status"] == "no_events"


def test_next_user_start_is_exclusive_response_boundary():
    details = _score([(1, 3), (5, 7)], [(5, 8)]).details
    assert all(result["sample_count"] == 0 for result in details["results"].values())


def test_nonfinal_old_speech_end_at_user_end_is_included():
    details = _score([(1, 3), (5, 7)], [(0, 3)]).details
    assert details["results"]["ird"]["samples_ms"] == [2000]


def test_cross_user_system_segment_is_not_consumed():
    details = _score([(1, 3), (5, 7)], [(0, 8)]).details
    # The same system segment is considered for both users. Non-final user has
    # no qualifying old-speech stop; final user has no user-end upper bound.
    assert details["results"]["ird"]["samples_ms"] == [3000]
    contexts = [
        event for event in details["events"] if event["branch_id"] == "user_starts_during_system"
    ]
    assert [event["user_segment_id"] for event in contexts] == ["u1", "u2"]


@pytest.mark.parametrize("metric", SUPPORTED_METRICS)
def test_selected_primary_metric_does_not_change_joint_computation(metric):
    details = _score([(1, 3), (6, 8)], [(2.75, 6.25), (8.4, 9)], metric=metric).details
    assert details["score"] == details["results"][metric]["score"]
    assert tuple(details["results"]) == SUPPORTED_METRICS
    assert details["evaluation_config"]["protocol"] == PROTOCOL


def test_translation_row_order_and_metadata_do_not_change_scores():
    first = _conversation([(1, 3), (6, 8)], [(2.75, 6.25), (8.4, 9)], key="a")
    shifted = _conversation([(2, 4), (7, 9)], [(3.75, 7.25), (9.4, 10)], key="b")
    assert score_fdbench_timing(DuplexTimelineBundle((first,))).details["results"] == (
        score_fdbench_timing(DuplexTimelineBundle((shifted,))).details["results"]
    )
    assert score_fdbench_timing(DuplexTimelineBundle((first, shifted))).details == (
        score_fdbench_timing(DuplexTimelineBundle((shifted, first))).details
    )
    branded = _conversation([(1, 3), (6, 8)], [(2.75, 6.25), (8.4, 9)], metadata={"model": "Moshi"})
    assert score_fdbench_timing(DuplexTimelineBundle((branded,))).details["results"] == (
        score_fdbench_timing(DuplexTimelineBundle((first,))).details["results"]
    )


@pytest.mark.parametrize("endpoint", ["2.50001", "2.50000000000000000000000000000000001"])
def test_off_grid_endpoints_are_rejected_without_decimal_context_rounding(endpoint):
    with localcontext() as context:
        context.prec = 8
        with pytest.raises(ValueError, match="conversation 'case'.*system segment 'a1'.start.*off"):
            _score([(1, 3)], [(endpoint, 4)])


@pytest.mark.parametrize("metric", ["wer", "srr", "IRD", "", None])
def test_unregistered_metric_is_rejected(metric):
    with pytest.raises(ValueError, match="Unsupported full-duplex metric"):
        _score([(1, 3)], [], metric=metric)


def test_unknown_protocol_and_empty_bundle_are_rejected():
    with pytest.raises(ValueError, match="Unsupported FD-Bench protocol"):
        _score([(1, 3)], [], protocol="paper_intuition")
    with pytest.raises(ValueError, match="at least one conversation"):
        score_fdbench_timing(DuplexTimelineBundle(()))


def test_independent_fixture_matrix_and_corpus_expectations():
    fixture_dir = Path(__file__).parents[2] / "fixtures" / "full_duplex"
    expected = json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8"))
    bundle, _ = normalize_duplex_timeline(
        fixture_dir / "reference.jsonl", fixture_dir / "sample_output.jsonl"
    )
    details = score_fdbench_timing(bundle).details
    for row in details["per_conversation"]:
        assert row["samples_ms"] == expected["per_conversation"][row["key"]]
    for metric, expected_result in expected["pooled"].items():
        for key, value in expected_result.items():
            assert details["results"][metric][key] == value
