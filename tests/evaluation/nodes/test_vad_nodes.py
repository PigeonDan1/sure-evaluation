from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ref_row(*, speech_segments: list[dict] | None = None, duration: float = 1.0) -> dict:
    return {
        "key": "utt1",
        "duration": duration,
        "speech_segments": speech_segments
        if speech_segments is not None
        else [{"start": 0.2, "end": 0.6}],
    }


def _pred_row(
    *,
    speech_segments: list[dict] | None = None,
    frame_scores: list[dict] | None = None,
    include_speech_segments: bool = True,
    include_frame_scores: bool = True,
) -> dict:
    row = {"key": "utt1", "audio_duration": 1.0}
    if include_speech_segments:
        row["speech_segments"] = (
            speech_segments if speech_segments is not None else [{"start": 0.2, "end": 0.6}]
        )
    if include_frame_scores:
        row["frame_scores"] = (
            frame_scores
            if frame_scores is not None
            else [
                {"start": 0.0, "end": 0.5, "score": 0.1},
                {"start": 0.5, "end": 1.0, "score": 0.9},
            ]
        )
    return row


def _normalize_from_files(reference_jsonl: Path, sample_output: Path):
    from sure_eval.evaluation.nodes.normalization.vad_timebase import normalize_vad_timebase
    from sure_eval.evaluation.nodes.validation.vad_contract import validate_vad_contract

    validated, _ = validate_vad_contract(reference_jsonl, sample_output)
    normalized, _ = normalize_vad_timebase(validated)
    return normalized


def test_vad_contract_accepts_minimal_reference_and_prediction(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.validation.vad_contract import validate_vad_contract

    reference_jsonl = tmp_path / "ref.jsonl"
    sample_output = tmp_path / "pred.jsonl"
    _write_jsonl(reference_jsonl, [_ref_row()])
    _write_jsonl(sample_output, [_pred_row()])

    bundle, trace = validate_vad_contract(reference_jsonl, sample_output)

    assert trace.node_id == "validation/vad_contract"
    assert bundle.input_summary["num_rows"] == 1
    row = bundle.rows[0]
    assert row.key == "utt1"
    assert row.available_metrics == {"f1", "p_fa", "p_miss", "dcf_nist", "auc_roc"}
    assert row.skipped_metrics == {}


def test_vad_contract_rejects_score_aliases(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.validation.vad_contract import validate_vad_contract

    reference_jsonl = tmp_path / "ref.jsonl"
    sample_output = tmp_path / "pred.jsonl"
    _write_jsonl(reference_jsonl, [_ref_row()])
    _write_jsonl(
        sample_output,
        [
            {
                "key": "utt1",
                "scores": [{"start": 0.0, "end": 0.01, "score": 0.1}],
            }
        ],
    )

    with pytest.raises(ValueError, match="unsupported score alias"):
        validate_vad_contract(reference_jsonl, sample_output)


def test_vad_detection_exact_match_f1_1(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.vad_detection_duration import (
        score_vad_detection_duration,
    )

    reference_jsonl = tmp_path / "ref.jsonl"
    sample_output = tmp_path / "pred.jsonl"
    _write_jsonl(reference_jsonl, [_ref_row()])
    _write_jsonl(sample_output, [_pred_row()])

    result = score_vad_detection_duration(_normalize_from_files(reference_jsonl, sample_output))

    assert result.details["primary_scores"]["f1"] == 1.0
    assert result.details["primary_scores"]["p_fa"] == 0.0
    assert result.details["primary_scores"]["p_miss"] == 0.0
    assert result.details["primary_scores"]["dcf_nist"] == 0.0


def test_vad_detection_false_alarm_and_miss_duration(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.vad_detection_duration import (
        score_vad_detection_duration,
    )

    reference_jsonl = tmp_path / "ref.jsonl"
    sample_output = tmp_path / "pred.jsonl"
    _write_jsonl(reference_jsonl, [_ref_row(speech_segments=[{"start": 0.2, "end": 0.6}])])
    _write_jsonl(sample_output, [_pred_row(speech_segments=[{"start": 0.4, "end": 0.8}])])

    result = score_vad_detection_duration(_normalize_from_files(reference_jsonl, sample_output))

    assert result.details["auxiliary"]["false_alarm_sec"] == pytest.approx(0.2)
    assert result.details["auxiliary"]["miss_sec"] == pytest.approx(0.2)
    assert result.details["primary_scores"]["f1"] == pytest.approx(0.5)
    assert result.details["primary_scores"]["p_fa"] == pytest.approx(1.0 / 3.0)
    assert result.details["primary_scores"]["p_miss"] == pytest.approx(0.5)


def test_vad_dcf_uses_025_fa_075_miss(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.vad_detection_duration import (
        score_vad_detection_duration,
    )

    reference_jsonl = tmp_path / "ref.jsonl"
    sample_output = tmp_path / "pred.jsonl"
    _write_jsonl(reference_jsonl, [_ref_row(speech_segments=[{"start": 0.2, "end": 0.6}])])
    _write_jsonl(sample_output, [_pred_row(speech_segments=[{"start": 0.4, "end": 0.8}])])

    result = score_vad_detection_duration(_normalize_from_files(reference_jsonl, sample_output))

    expected = 0.25 * (1.0 / 3.0) + 0.75 * 0.5
    assert result.details["primary_scores"]["dcf_nist"] == pytest.approx(expected)


def test_vad_auc_requires_frame_scores(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.vad.pipeline import evaluate_vad_files

    reference_jsonl = tmp_path / "ref.jsonl"
    sample_output = tmp_path / "pred.jsonl"
    _write_jsonl(reference_jsonl, [_ref_row()])
    _write_jsonl(sample_output, [_pred_row(include_frame_scores=False)])

    report = evaluate_vad_files(
        reference_jsonl=reference_jsonl,
        sample_output=sample_output,
        metric="auc_roc",
    )

    assert report.score is None
    assert report.details["primary_scores"]["auc_roc"] is None
    assert report.details["auxiliary"]["num_auc_samples"] == 0
    assert report.details["skipped_metrics"][0]["reason"] == "missing prediction field: frame_scores"


def test_vad_auc_returns_none_for_single_class_labels(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.vad.pipeline import evaluate_vad_files

    reference_jsonl = tmp_path / "ref.jsonl"
    sample_output = tmp_path / "pred.jsonl"
    _write_jsonl(reference_jsonl, [_ref_row(speech_segments=[{"start": 0.0, "end": 0.03}], duration=0.03)])
    _write_jsonl(
        sample_output,
        [
            _pred_row(
                speech_segments=[],
                frame_scores=[
                    {"start": 0.0, "end": 0.01, "score": 0.1},
                    {"start": 0.01, "end": 0.02, "score": 0.2},
                    {"start": 0.02, "end": 0.03, "score": 0.3},
                ],
            )
        ],
    )

    report = evaluate_vad_files(
        reference_jsonl=reference_jsonl,
        sample_output=sample_output,
        metric="auc_roc",
    )

    assert report.score is None
    assert report.details["primary_scores"]["auc_roc"] is None
    assert report.details["auxiliary"]["num_auc_samples"] == 3
    assert report.details["skipped_metrics"][-1]["reason"] == "single_class_labels"
