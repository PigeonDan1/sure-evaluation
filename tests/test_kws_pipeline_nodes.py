from __future__ import annotations

import json
from pathlib import Path


def _samples():
    from sure_eval.evaluation.tasks.kws import KWSSample

    return [
        KWSSample(
            key="pos",
            expected_detected=True,
            expected_keyword="嗨小问",
            duration=1.2,
            detected=True,
            predicted_keyword="嗨小问",
            score=0.9,
        ),
        KWSSample(
            key="neg",
            expected_detected=False,
            duration=2.4,
            detected=False,
            score=None,
        ),
    ]


def test_kws_task_route_uses_wekws_det_scoring_node() -> None:
    from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_samples

    report = evaluate_kws_samples(_samples(), threshold=0.5, thresholds=[0.0, 0.5, 0.95])

    assert report.task == "KWS"
    assert report.metric == "accuracy"
    assert report.score == 1.0
    assert report.pipeline_id == "kws.any.accuracy.wekws_det_v1"
    assert report.pipeline_trace[0].node_id == "scoring/wekws_det"
    assert report.pipeline_trace[0].internal_stages == (
        "keyword_normalization",
        "threshold_decision",
        "det_curve",
        "operating_point_summary",
    )
    assert report.details["results"]["accuracy"]["score"] == 1.0
    assert report.details["summary"]["best_threshold"] == 0.5
    assert report.input_contract is not None
    assert report.input_contract.metric_id == "scoring/wekws_det"
    assert report.input_contract.required_roles == ("samples",)


def test_kws_task_route_supports_macro_recall_primary_metric() -> None:
    from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_samples

    report = evaluate_kws_samples(
        _samples(),
        metric="macro-recall",
        threshold=0.5,
        thresholds=[0.0, 0.5, 0.95],
        macro_recall_false_alarms=0,
    )

    assert report.metric == "macro_recall"
    assert report.score == report.details["results"]["macro-recall"]["score"]
    assert report.pipeline_id == "kws.any.macro_recall.wekws_det_v1"
    assert report.details["summary"]["macro_recall_false_alarm_budget"] == 0


def test_kws_routes_cover_accuracy_and_macro_recall_for_each_input_mode() -> None:
    import yaml

    routes = yaml.safe_load(
        Path("src/sure_eval/evaluation/tasks/kws/routes.yaml").read_text(encoding="utf-8")
    )
    route_keys = {(route["input_mode"], route["metric"]) for route in routes["routes"]}

    assert route_keys == {
        ("sure_json", "accuracy"),
        ("sure_json", "macro_recall"),
        ("wekws_score_ctc", "accuracy"),
        ("wekws_score_ctc", "macro_recall"),
        ("wekws_frame_score", "accuracy"),
        ("wekws_frame_score", "macro_recall"),
    }


def test_legacy_kws_pipeline_matches_task_route() -> None:
    from sure_eval.evaluation.tasks.kws import KWSMetricPipeline
    from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_samples

    samples = _samples()
    legacy = KWSMetricPipeline(threshold=0.5, thresholds=[0.0, 0.5, 0.95]).evaluate(samples)
    routed = evaluate_kws_samples(samples, threshold=0.5, thresholds=[0.0, 0.5, 0.95])

    assert legacy.results["accuracy"].score == routed.details["results"]["accuracy"]["score"]
    assert legacy.results["false_alarm_per_hour"].score == (
        routed.details["results"]["false_alarm_per_hour"]["score"]
    )
    assert (
        legacy.results["macro-recall"].score == routed.details["results"]["macro-recall"]["score"]
    )
    assert legacy.rows == routed.details["rows"]
    assert legacy.summary == routed.details["summary"]


def test_kws_file_route_declares_sure_json_input_contract(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_files

    gt = tmp_path / "gt.jsonl"
    gt.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "key": "pos",
                        "expected": "detect",
                        "text": "嗨小问",
                        "duration": 1.0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "key": "neg",
                        "expected": "reject",
                        "duration": 2.0,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs = tmp_path / "sample_output.json"
    outputs.write_text(
        json.dumps(
            [
                {"key": "pos", "result": {"detected": True, "keyword": "嗨小问", "score": 0.91}},
                {"key": "neg", "result": {"detected": False, "keyword": None, "score": None}},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_kws_files(
        reference_jsonl=gt,
        sample_output=outputs,
        threshold=0.5,
        metric="macro-recall",
        macro_recall_false_alarms=0,
    )

    assert report.metric == "macro_recall"
    assert report.score == 1.0
    assert report.pipeline_id == (
        "kws.any.macro_recall.conversion_kws_sure_json_to_samples_v1.wekws_det_v1"
    )
    assert report.details["input_mode"] == "sure_json"
    assert report.details["input_contract"]["required_roles"] == [
        "reference_jsonl",
        "sample_output",
    ]
    assert report.details["input_files"] == {
        "reference_jsonl": str(gt),
        "sample_output": str(outputs),
    }


def test_kws_file_route_declares_wekws_score_input_contract(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_files

    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            [
                json.dumps({"key": "pos", "txt": "嗨小问", "duration": 1.0}, ensure_ascii=False),
                json.dumps({"key": "neg", "txt": "其他文本", "duration": 2.0}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scores = tmp_path / "score.txt"
    scores.write_text("pos detected 嗨小问 0.930\nneg rejected\n", encoding="utf-8")

    report = evaluate_kws_files(wekws_label_file=labels, wekws_score_file=scores, keyword="嗨小问")

    assert report.score == 1.0
    assert report.details["input_mode"] == "wekws_score_ctc"
    assert report.details["input_contract"]["required_roles"] == [
        "wekws_label_file",
        "wekws_score_file",
        "keyword",
    ]


def test_kws_runner_outputs_pipeline_metadata(tmp_path: Path, capsys) -> None:
    from scripts.run_kws_metric_pipeline import main

    gt = tmp_path / "gt.jsonl"
    gt.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "key": "pos",
                        "expected": "detect",
                        "text": "嗨小问",
                        "duration": 1.0,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "key": "neg",
                        "expected": "reject",
                        "duration": 2.0,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sample_output = tmp_path / "sample_output.json"
    sample_output.write_text(
        json.dumps(
            [
                {"key": "pos", "result": {"detected": True, "keyword": "嗨小问", "score": 0.91}},
                {"key": "neg", "result": {"detected": False, "keyword": None, "score": None}},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--reference-jsonl",
            str(gt),
            "--sample-output",
            str(sample_output),
            "--threshold",
            "0.5",
            "--metric",
            "macro-recall",
            "--macro-recall-false-alarms",
            "0",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["metric"] == "macro_recall"
    assert payload["pipeline_id"] == (
        "kws.any.macro_recall.conversion_kws_sure_json_to_samples_v1.wekws_det_v1"
    )
    assert "macro-recall" in payload["metrics"]
    assert payload["input_contract"]["metric_id"] == "scoring/wekws_det"
    assert payload["pipeline_trace"][0]["node_id"] == "scoring/wekws_det"
