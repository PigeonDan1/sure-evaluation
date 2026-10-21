from __future__ import annotations
import json
import pytest

from sure_eval.evaluation.tasks.kws.loaders import (
    load_samples_from_jsonl_and_outputs,
    load_samples_from_wekws_frame_score_file,
    load_samples_from_wekws_score_file,
)
from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_files, evaluate_kws_samples


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def fixture_files(tmp_path):
    ref, out = tmp_path / "ref.jsonl", tmp_path / "out.json"
    write_rows(
        ref,
        [
            {"key": "p", "expected": True, "expected_keyword": "A", "keyword_end_sec": 1},
            {"key": "n", "expected": False, "duration": 36000},
        ],
    )
    out.write_text(
        json.dumps(
            [
                {"key": "p", "result": {"keyword": "A", "score": 0.7, "trigger_time_sec": 1.2}},
                {"key": "n", "result": {"keyword": "A", "score": 0.8}},
            ]
        )
    )
    return ref, out


def test_file_route_has_v2_identity_and_new_metrics(tmp_path):
    ref, out = fixture_files(tmp_path)
    report = evaluate_kws_files(reference_jsonl=ref, sample_output=out)
    assert report.metric == "macro_recall"
    assert (
        report.pipeline_id
        == "kws.any.macro_recall.conversion_kws_sure_json_to_samples_v2.wekws_det_v2"
    )
    assert report.score == 1
    assert report.pipeline_trace[0].version == "v2"
    assert report.details["summary"]["false_alarm_per_hour"] == 0.1
    assert report.details["results"]["trigger_latency"]["score"] == pytest.approx(200)
    assert set(report.details["results"]) == {
        "macro_recall",
        "per_keyword_recall",
        "trigger_latency",
    }
    with pytest.raises(ValueError, match="Unsupported"):
        evaluate_kws_files(reference_jsonl=ref, sample_output=out, metric="accuracy")


@pytest.mark.parametrize("problem", ["missing", "extra", "duplicate_output", "duplicate_reference"])
def test_loader_requires_exact_one_to_one_key_alignment(tmp_path, problem):
    ref, out = fixture_files(tmp_path)
    outputs = json.loads(out.read_text())
    if problem == "missing":
        outputs.pop()
    elif problem == "extra":
        outputs.append({"key": "extra"})
    elif problem == "duplicate_output":
        outputs.append(outputs[0])
    else:
        ref.write_text(ref.read_text() + ref.read_text().splitlines()[0] + "\n")
    out.write_text(json.dumps(outputs))
    with pytest.raises(ValueError, match="mismatch|Duplicate"):
        load_samples_from_jsonl_and_outputs(ref, out)


def test_jsonl_event_lists_support_multiple_triggers(tmp_path):
    ref, out = tmp_path / "ref.jsonl", tmp_path / "pred.jsonl"
    write_rows(
        ref,
        [
            {"key": "p", "events": [{"keyword": "A", "start_sec": 0.5, "end_sec": 1}]},
            {"key": "n", "events": [], "duration": 72000},
        ],
    )
    write_rows(
        out,
        [
            {"key": "p", "detections": [{"keyword": "A", "score": 0.9, "trigger_time_sec": 1.1}]},
            {
                "key": "n",
                "detections": [{"keyword": "A", "score": 0.8}, {"keyword": "B", "score": 0.8}],
            },
        ],
    )
    report = evaluate_kws_files(reference_jsonl=ref, sample_output=out, thresholds=[0.5])
    assert report.score == 1
    assert report.details["summary"]["false_alarms"] == 2
    assert report.details["summary"]["false_alarm_per_hour"] == 0.1


def test_wekws_frame_input_preserves_below_default_threshold_candidates(tmp_path):
    ref, scores = tmp_path / "ref.jsonl", tmp_path / "scores.txt"
    write_rows(
        ref,
        [
            {"key": "p", "txt": "A", "keyword_end_sec": 0.15, "frame_shift_sec": 0.1},
            {"key": "n", "txt": "noise", "duration": 3600},
        ],
    )
    scores.write_text("p A 0.1 0.4 0.45\nn A 0.1 0.2 0.3\n")
    samples = load_samples_from_wekws_frame_score_file(ref, scores, keyword="A")
    report = evaluate_kws_samples(samples, thresholds=[0.4])
    assert report.score == 1
    assert report.details["rows"][0]["trigger_time_sec"] == pytest.approx(0.2)
    assert report.details["results"]["trigger_latency"]["score"] == pytest.approx(50)


def test_wekws_ctc_requires_scores_even_on_rejected_rows(tmp_path):
    ref, scores = tmp_path / "ref.jsonl", tmp_path / "scores.txt"
    write_rows(ref, [{"key": "p", "txt": "A"}, {"key": "n", "txt": "noise", "duration": 3600}])
    scores.write_text("p detected A .4\nn rejected\n")
    with pytest.raises(ValueError, match="uncensored score"):
        load_samples_from_wekws_score_file(ref, scores, keyword="A")
    scores.write_text("p rejected A .4\nn rejected A .2\n")
    assert evaluate_kws_files(wekws_label_file=ref, wekws_score_file=scores, keyword="A").score == 1


def test_runner_uses_pipeline_json(tmp_path, capsys):
    from scripts.run_kws_metric_pipeline import main
    from sure_eval.evaluation.cli_adapters import build_pipeline_spec, write_json

    ref, out = fixture_files(tmp_path)
    pipeline = tmp_path / "pipeline.json"
    write_json(pipeline, build_pipeline_spec("kws"))
    assert (
        main(
            [
                "--pipeline",
                str(pipeline),
                "--reference-jsonl",
                str(ref),
                "--sample-output",
                str(out),
                "--output-dir",
                str(tmp_path / "result"),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["metric"] == "macro_recall"
    assert payload["score"] == 1
