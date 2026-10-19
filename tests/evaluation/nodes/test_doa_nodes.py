from __future__ import annotations

import json
import math
import random
from itertools import permutations
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    frame_index: int,
    *,
    azimuth: float,
    recording_id: str = "rec-1",
    elevation: float | None = None,
    source_id: str = "s0",
    num_channels: int = 4,
    activity: bool = True,
) -> dict:
    source = {"source_id": source_id, "azimuth": azimuth, "activity": activity}
    if elevation is not None:
        source["elevation"] = elevation
    return {
        "recording_id": recording_id,
        "frame_index": frame_index,
        "timestamp": frame_index * 0.01,
        "num_channels": num_channels,
        "coordinate_system": "array_local_spherical",
        "angle_unit": "degree",
        "sources": [source],
    }


def test_doa_normalization_wraps_azimuth_and_records_trace(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [_row(0, azimuth=179.0)])
    _write_jsonl(prediction, [_row(0, azimuth=-179.0)])

    bundle, trace = normalize_doa_jsonl(reference, prediction)

    assert bundle.reference[0].sources[0].azimuth == pytest.approx(179.0)
    assert bundle.prediction[0].sources[0].azimuth == pytest.approx(-179.0)
    assert trace.node_id == "normalization/doa_jsonl"
    assert "exact_frame_alignment" in trace.internal_stages


def test_doa_normalization_rejects_duplicate_frame_and_contract_mismatch(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [_row(0, azimuth=0.0), _row(0, azimuth=1.0)])
    _write_jsonl(prediction, [_row(0, azimuth=0.0)])
    with pytest.raises(ValueError, match="duplicate recording_id/frame_index"):
        normalize_doa_jsonl(reference, prediction)

    _write_jsonl(reference, [_row(0, azimuth=0.0, num_channels=4)])
    _write_jsonl(prediction, [_row(0, azimuth=0.0, num_channels=2)])
    with pytest.raises(ValueError, match="num_channels mismatch"):
        normalize_doa_jsonl(reference, prediction)

    _write_jsonl(reference, [_row(0, azimuth=0.0, num_channels=1)])
    _write_jsonl(prediction, [_row(0, azimuth=0.0, num_channels=1)])
    with pytest.raises(ValueError, match="integer >= 2"):
        normalize_doa_jsonl(reference, prediction)

    _write_jsonl(reference, [_row(0, azimuth=0.0, elevation=1.0)])
    _write_jsonl(prediction, [_row(0, azimuth=0.0, elevation=1.0)])
    with pytest.raises(ValueError, match="must not include elevation"):
        normalize_doa_jsonl(reference, prediction)


def test_doa_normalization_requires_explicit_convention_and_aligns_by_key(
    tmp_path: Path,
) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    missing_convention = _row(0, azimuth=0.0)
    missing_convention.pop("coordinate_system")
    _write_jsonl(reference, [missing_convention])
    _write_jsonl(prediction, [_row(0, azimuth=0.0)])
    with pytest.raises(ValueError, match="coordinate_system"):
        normalize_doa_jsonl(reference, prediction)

    _write_jsonl(
        reference,
        [_row(1, azimuth=721.0), _row(0, azimuth=-721.0)],
    )
    _write_jsonl(
        prediction,
        [_row(0, azimuth=-1.0), _row(1, azimuth=1.0)],
    )
    bundle, trace = normalize_doa_jsonl(reference, prediction)
    assert [frame.frame_index for frame in bundle.reference] == [0, 1]
    assert [frame.sources[0].azimuth for frame in bundle.reference] == pytest.approx([-1.0, 1.0])
    assert trace.details["input_summary"]["input_order"] == (
        "normalized_by_recording_id_then_frame_index"
    )

    _write_jsonl(prediction, [_row(0, azimuth=0.0), _row(2, azimuth=0.0)])
    with pytest.raises(ValueError, match="identical recording_id/frame_index keys"):
        normalize_doa_jsonl(reference, prediction)


def test_doa_normalization_rejects_timestamp_mismatch(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    reference_row = _row(0, azimuth=0.0)
    prediction_row = _row(0, azimuth=0.0)
    prediction_row["timestamp"] = 0.1
    _write_jsonl(reference, [reference_row])
    _write_jsonl(prediction, [prediction_row])

    with pytest.raises(ValueError, match="timestamp mismatch"):
        normalize_doa_jsonl(reference, prediction)


def test_doa_normalization_rejects_invalid_coordinate_and_angle_unit() -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl.node import frame_from_payload

    invalid_coordinate = _row(0, azimuth=0.0)
    invalid_coordinate["coordinate_system"] = "world_spherical"
    with pytest.raises(ValueError, match="array_local_spherical"):
        frame_from_payload(invalid_coordinate, role="reference", dimension="2d")

    invalid_unit = _row(0, azimuth=0.0)
    invalid_unit["angle_unit"] = "radian"
    with pytest.raises(ValueError, match="angle_unit='degree'"):
        frame_from_payload(invalid_unit, role="reference", dimension="2d")


def test_doa_normalization_requires_elevation_in_3d() -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl.node import frame_from_payload

    with pytest.raises(ValueError, match="requires elevation for 3d"):
        frame_from_payload(_row(0, azimuth=0.0), role="reference", dimension="3d")


def test_doa_normalization_rejects_malformed_and_empty_jsonl(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    reference.write_text("{not-json}\n", encoding="utf-8")
    _write_jsonl(prediction, [_row(0, azimuth=0.0)])
    with pytest.raises(ValueError, match="invalid JSON"):
        normalize_doa_jsonl(reference, prediction)

    reference.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one frame"):
        normalize_doa_jsonl(reference, prediction)


@pytest.mark.parametrize("azimuth", [math.nan, math.inf, -math.inf])
def test_doa_normalization_rejects_non_finite_azimuth(azimuth: float) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl.node import frame_from_payload

    with pytest.raises(ValueError, match="finite number"):
        frame_from_payload(_row(0, azimuth=azimuth), role="reference", dimension="2d")


def test_doa_scoring_uses_circular_error_and_strict_gate(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [_row(0, azimuth=179.0)])
    _write_jsonl(prediction, [_row(0, azimuth=-179.0)])
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    result = score_doa_localization(bundle, metric="localization_recall", threshold_deg=2.0)

    assert result.details["score"] == pytest.approx(0.0)
    assert result.details["micro"]["raw_errors_deg"] == [pytest.approx(2.0)]


def test_doa_scoring_assigns_multiple_sources_by_minimum_total_cost(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    def multi_row(angles: list[float]) -> dict:
        row = _row(0, azimuth=angles[0])
        row["sources"] = [
            {"source_id": f"s{i}", "azimuth": angle, "activity": True}
            for i, angle in enumerate(angles)
        ]
        return row

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [multi_row([0.0, 100.0])])
    _write_jsonl(prediction, [multi_row([98.0, 3.0])])
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    result = score_doa_localization(bundle, metric="mae", threshold_deg=10.0)

    assert result.details["score"] == pytest.approx(2.5)
    assert result.details["per_recording"]["rec-1"]["assigned_pairs"] == 2


def test_doa_scoring_handles_more_references_than_predictions(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    def multi_row(angles: list[float]) -> dict:
        row = _row(0, azimuth=angles[0])
        row["sources"] = [
            {"source_id": f"s{i}", "azimuth": angle, "activity": True}
            for i, angle in enumerate(angles)
        ]
        return row

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [multi_row([0.0, 100.0, 200.0])])
    _write_jsonl(prediction, [multi_row([2.0, 198.0])])
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    result = score_doa_localization(bundle, metric="mae", threshold_deg=10.0)

    assert result.details["score"] == pytest.approx(2.0)
    assert result.details["micro"]["assigned_pairs"] == 2
    assert result.details["micro"]["missed_reference_sources"] == 1


def test_doa_scoring_handles_more_predictions_than_references(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    reference_row = _row(0, azimuth=0.0)
    prediction_row = _row(0, azimuth=1.0)
    prediction_row["sources"].append({"source_id": "extra", "azimuth": 90.0, "activity": True})
    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [reference_row])
    _write_jsonl(prediction, [prediction_row])
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    result = score_doa_localization(bundle, metric="localization_precision")

    assert result.details["score"] == pytest.approx(0.5)
    assert result.details["micro"]["assigned_pairs"] == 1
    assert result.details["micro"]["false_prediction_sources"] == 1


def test_doa_3d_scoring_uses_great_circle_distance(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [_row(0, azimuth=0.0, elevation=0.0)])
    _write_jsonl(prediction, [_row(0, azimuth=0.0, elevation=90.0)])
    bundle, _ = normalize_doa_jsonl(reference, prediction, dimension="3d")

    result = score_doa_localization(bundle, metric="mae")

    assert result.details["score"] == pytest.approx(90.0)


def test_doa_angular_distance_independent_boundary_values() -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import DOASource
    from sure_eval.evaluation.nodes.scoring.doa_localization.node import _angular_distance

    def source(azimuth: float, elevation: float | None = None) -> DOASource:
        return DOASource("s", azimuth, elevation, True)

    assert _angular_distance(source(0.0), source(360.0), "2d") == pytest.approx(0.0)
    assert _angular_distance(source(-180.0), source(180.0), "2d") == pytest.approx(0.0)
    assert _angular_distance(source(0.0), source(180.0), "2d") == pytest.approx(180.0)
    assert _angular_distance(source(179.999999), source(-179.999999), "2d") == pytest.approx(
        0.000002, abs=1e-9
    )
    assert _angular_distance(source(0.0, 0.0), source(0.0, 0.0), "3d") == pytest.approx(0.0)
    assert _angular_distance(source(0.0, 0.0), source(180.0, 0.0), "3d") == pytest.approx(180.0)
    assert _angular_distance(source(0.0, 90.0), source(123.0, 90.0), "3d") == pytest.approx(0.0)


@pytest.mark.parametrize(
    "costs",
    [
        [[1.0]],
        [[1.0, 1.0]],
        [[4.0, 1.0], [2.0, 3.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[9.0, 2.0, 7.0], [6.0, 4.0, 3.0], [5.0, 8.0, 1.0]],
        [[0.1, 0.1000000001, 4.0], [0.1000000002, 0.1, 3.0]],
    ],
)
def test_doa_hungarian_matches_independent_exhaustive_reference(
    costs: list[list[float]],
) -> None:
    from sure_eval.evaluation.nodes.scoring.doa_localization.node import (
        _hungarian_rectangular,
    )

    assignment = _hungarian_rectangular(costs)
    actual = sum(costs[row][column] for row, column in enumerate(assignment))
    expected = min(
        sum(costs[row][column] for row, column in enumerate(columns))
        for columns in permutations(range(len(costs[0])), len(costs))
    )
    assert actual == pytest.approx(expected)
    assert len(set(assignment)) == len(assignment)


def test_doa_hungarian_matches_exhaustive_reference_on_random_small_matrices() -> None:
    from sure_eval.evaluation.nodes.scoring.doa_localization.node import (
        _hungarian_rectangular,
    )

    generator = random.Random(20260921)
    for rows in range(1, 5):
        for columns in range(rows, 6):
            for _ in range(5):
                costs = [
                    [generator.uniform(0.0, 180.0) for _ in range(columns)] for _ in range(rows)
                ]
                assignment = _hungarian_rectangular(costs)
                actual = sum(costs[row][column] for row, column in enumerate(assignment))
                expected = min(
                    sum(costs[row][column] for row, column in enumerate(candidate))
                    for candidate in permutations(range(columns), rows)
                )
                assert actual == pytest.approx(expected)


def test_doa_assignment_is_minimum_raw_cost_not_maximum_gated_tp(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    def multi_row(angles: list[float]) -> dict:
        row = _row(0, azimuth=angles[0])
        row["sources"] = [
            {"source_id": f"s{i}", "azimuth": angle, "activity": True}
            for i, angle in enumerate(angles)
        ]
        return row

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [multi_row([0.0, 11.0])])
    _write_jsonl(prediction, [multi_row([10.0, 21.0])])
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    result = score_doa_localization(bundle, metric="localization_recall", threshold_deg=10.0)

    assert result.details["micro"]["raw_errors_deg"] == pytest.approx([10.0, 10.0])
    assert result.details["score"] == pytest.approx(0.0)
    assert "does not maximize" in result.details["assignment_tp_policy"]
    f1_result = score_doa_localization(bundle, metric="localization_f1", threshold_deg=10.0)
    assert f1_result.details["score"] == pytest.approx(0.0)


def test_doa_source_ids_do_not_match_identity_and_activity_policy_is_explicit(
    tmp_path: Path,
) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    reference_row = _row(0, azimuth=5.0, source_id="reference-track")
    prediction_row = _row(
        0,
        azimuth=5.0,
        source_id="unrelated-prediction-label",
        activity=False,
    )
    prediction_row["sources"][0]["confidence"] = 0.01
    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [reference_row])
    _write_jsonl(prediction, [prediction_row])

    presence_bundle, _ = normalize_doa_jsonl(
        reference,
        prediction,
        prediction_activity_policy="presence",
    )
    explicit_bundle, _ = normalize_doa_jsonl(
        reference,
        prediction,
        prediction_activity_policy="explicit",
    )
    presence = score_doa_localization(presence_bundle, metric="localization_recall")
    explicit = score_doa_localization(explicit_bundle, metric="localization_recall")

    assert presence.details["score"] == pytest.approx(1.0)
    assert explicit.details["score"] == pytest.approx(0.0)
    assert explicit.details["micro"]["active_prediction_sources"] == 0


def test_doa_threshold_counters_and_fixed_acc_threshold_are_real(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [_row(index, azimuth=0.0) for index in range(4)])
    _write_jsonl(
        prediction,
        [_row(index, azimuth=error) for index, error in enumerate([9.0, 10.0, 14.0, 29.0])],
    )
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    result = score_doa_localization(bundle, metric="acc_15", threshold_deg=99.0)
    summary = result.details["micro"]

    assert result.details["threshold_deg"] == 15.0
    assert result.details["configured_threshold_deg"] == 99.0
    assert summary["true_positive_10"] == 1
    assert summary["true_positive_15"] == 3
    assert summary["true_positive_30"] == 4
    assert summary["true_positive"] == 3
    assert summary["missed_reference_sources"] == 1
    assert summary["false_prediction_sources"] == 1
    assert result.details["score"] == pytest.approx(0.75)


def test_doa_empty_sources_have_documented_undefined_scores(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    empty = _row(0, azimuth=0.0)
    empty["sources"] = []
    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [empty])
    _write_jsonl(prediction, [empty])
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    for metric in ("mae", "rmse", "acc_10", "localization_precision", "localization_f1"):
        result = score_doa_localization(bundle, metric=metric)
        assert result.details["score"] is None
        assert result.details["micro"]["true_positive_10"] == 0
    assert "None" in result.details["undefined_score_policy"]


def test_doa_aggregation_distinguishes_micro_and_macro_recording(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    def rows(recording_id: str, errors: list[float]) -> list[dict]:
        return [
            _row(index, azimuth=error, recording_id=recording_id)
            for index, error in enumerate(errors)
        ]

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    reference_rows = rows("short", [0.0]) + rows("long", [0.0, 0.0])
    prediction_rows = rows("short", [0.0]) + rows("long", [10.0, 10.0])
    _write_jsonl(reference, reference_rows)
    _write_jsonl(prediction, prediction_rows)
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    micro = score_doa_localization(bundle, metric="mae", aggregation="micro")
    macro = score_doa_localization(bundle, metric="mae", aggregation="macro_recording")

    assert micro.details["score"] == pytest.approx(20.0 / 3.0)
    assert macro.details["score"] == pytest.approx(5.0)
    assert macro.details["results"]["mae"] == pytest.approx(5.0)
    assert macro.details["micro_results"]["mae"] == pytest.approx(20.0 / 3.0)


def test_doa_rmse_micro_is_pooled_and_macro_is_mean_recording_rmse(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    reference_rows = [
        _row(0, azimuth=0.0, recording_id="a"),
        _row(0, azimuth=0.0, recording_id="b"),
        _row(1, azimuth=0.0, recording_id="b"),
    ]
    prediction_rows = [
        _row(0, azimuth=0.0, recording_id="a"),
        _row(0, azimuth=3.0, recording_id="b"),
        _row(1, azimuth=4.0, recording_id="b"),
    ]
    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, reference_rows)
    _write_jsonl(prediction, prediction_rows)
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    micro = score_doa_localization(bundle, metric="rmse", aggregation="micro")
    macro = score_doa_localization(bundle, metric="rmse", aggregation="macro_recording")

    assert micro.details["score"] == pytest.approx(math.sqrt(25.0 / 3.0))
    assert macro.details["score"] == pytest.approx(math.sqrt(12.5) / 2.0)


def test_doa_macro_excludes_recordings_where_metric_is_undefined(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    defined_reference = _row(0, azimuth=0.0, recording_id="defined")
    defined_prediction = _row(0, azimuth=4.0, recording_id="defined")
    empty_reference = _row(0, azimuth=0.0, recording_id="empty")
    empty_prediction = _row(0, azimuth=0.0, recording_id="empty")
    empty_reference["sources"] = []
    empty_prediction["sources"] = []
    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    _write_jsonl(reference, [defined_reference, empty_reference])
    _write_jsonl(prediction, [defined_prediction, empty_prediction])
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    result = score_doa_localization(bundle, metric="mae", aggregation="macro_recording")

    assert result.details["score"] == pytest.approx(4.0)
    assert result.details["aggregation_summary"]["mae"] == {
        "defined_recordings": 1,
        "excluded_undefined_recordings": 1,
    }


def test_doa_scoring_is_deterministic(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.doa_jsonl import normalize_doa_jsonl
    from sure_eval.evaluation.nodes.scoring.doa_localization import score_doa_localization

    reference = tmp_path / "reference.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    reference_row = _row(0, azimuth=0.0)
    prediction_row = _row(0, azimuth=1.0)
    reference_row["sources"].append({"source_id": "s1", "azimuth": 90.0, "activity": True})
    prediction_row["sources"].append({"source_id": "s1", "azimuth": 89.0, "activity": True})
    _write_jsonl(reference, [reference_row])
    _write_jsonl(prediction, [prediction_row])
    bundle, _ = normalize_doa_jsonl(reference, prediction)

    first = score_doa_localization(bundle, metric="localization_f1")
    second = score_doa_localization(bundle, metric="localization_f1")

    assert first == second
