from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_kws_pipeline_scores_detection_samples() -> None:
    from sure_eval.evaluation.tasks.kws import KWSSample, KWSMetricPipeline

    samples = [
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

    report = KWSMetricPipeline(thresholds=[0.0, 0.5, 0.95]).evaluate(samples)

    assert report.results["accuracy"].score == 1.0
    assert report.results["recall"].score == 1.0
    assert report.results["false_reject_rate"].score == 0.0
    assert report.results["false_alarm_rate"].score == 0.0
    assert report.results["false_alarm_per_hour"].score == 0.0
    assert report.results["det_curve"].details["points"][1] == {
        "threshold": 0.5,
        "false_alarm_per_hour": 0.0,
        "false_reject_rate": 0.0,
        "false_alarm_rate": 0.0,
        "true_detect_rate": 1.0,
        "false_alarms": 0,
        "false_rejects": 0,
    }
    assert report.summary["best_threshold"] == 0.5
    assert report.results["macro-recall"].score == 1.0


def test_kws_macro_recall_at_false_alarm_budget() -> None:
    from sure_eval.evaluation.tasks.kws import KWSSample, KWSMetricPipeline

    samples = [
        KWSSample(
            key="pos_high",
            expected_detected=True,
            expected_keyword="嗨小问",
            detected=True,
            predicted_keyword="嗨小问",
            score=0.9,
        ),
        KWSSample(
            key="pos_low",
            expected_detected=True,
            expected_keyword="嗨小问",
            detected=True,
            predicted_keyword="嗨小问",
            score=0.7,
        ),
        KWSSample(
            key="neg_high", expected_detected=False, detected=True, score=0.8, duration=3600.0
        ),
        KWSSample(
            key="neg_low", expected_detected=False, detected=True, score=0.2, duration=3600.0
        ),
    ]

    zero_fa = KWSMetricPipeline(
        thresholds=[0.5, 0.75, 0.85],
        macro_recall_false_alarms=0,
    ).evaluate(samples)
    one_fa = KWSMetricPipeline(
        thresholds=[0.5, 0.75, 0.85],
        macro_recall_false_alarms=1,
    ).evaluate(samples)

    assert zero_fa.results["macro-recall"].score == 0.5
    assert zero_fa.results["macro-recall"].details["threshold"] == 0.85
    assert zero_fa.results["macro-recall"].details["achieved_false_alarms"] == 0
    assert one_fa.results["macro-recall"].score == 1.0
    assert one_fa.results["macro-recall"].details["threshold"] == 0.5
    assert one_fa.results["macro-recall"].details["achieved_false_alarms"] == 1


def test_kws_macro_recall_rejects_negative_false_alarm_budget() -> None:
    from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import (
        compute_macro_recall_at_false_alarms,
    )

    with pytest.raises(ValueError, match="non-negative"):
        compute_macro_recall_at_false_alarms([], false_alarm_budget=-1)


def test_kws_macro_recall_penalizes_wrong_keyword() -> None:
    from sure_eval.evaluation.tasks.kws import KWSSample, KWSMetricPipeline

    report = KWSMetricPipeline(thresholds=[0.5], macro_recall_false_alarms=0).evaluate(
        [
            KWSSample(
                key="pos",
                expected_detected=True,
                expected_keyword="嗨小问",
                detected=True,
                predicted_keyword="你好问问",
                score=0.99,
            )
        ]
    )

    assert report.results["macro-recall"].score == 0.0


def test_kws_macro_recall_without_positive_samples_is_zero() -> None:
    from sure_eval.evaluation.tasks.kws import KWSSample, KWSMetricPipeline

    report = KWSMetricPipeline(thresholds=[0.5], macro_recall_false_alarms=0).evaluate(
        [KWSSample(key="neg", expected_detected=False, detected=False, score=0.1)]
    )

    assert report.results["macro-recall"].score == 0.0


def test_kws_macro_recall_registry_metric_uses_macro_recall_score() -> None:
    from sure_eval.evaluation.registry import MetricRegistry
    from sure_eval.evaluation.tasks.kws import KWSSample

    samples = [
        KWSSample(
            key="pos_high",
            expected_detected=True,
            expected_keyword="嗨小问",
            detected=True,
            predicted_keyword="嗨小问",
            score=0.9,
        ),
        KWSSample(
            key="pos_low",
            expected_detected=True,
            expected_keyword="嗨小问",
            detected=True,
            predicted_keyword="嗨小问",
            score=0.7,
        ),
        KWSSample(key="neg_high", expected_detected=False, detected=True, score=0.8),
        KWSSample(key="neg_low", expected_detected=False, detected=True, score=0.2),
    ]

    accuracy = MetricRegistry.get_metric("kws_accuracy").calculate_samples(
        samples,
        thresholds=[0.5, 0.75, 0.85],
        macro_recall_false_alarms=0,
    )
    macro_recall = MetricRegistry.get_metric("macro-recall").calculate_samples(
        samples,
        thresholds=[0.5, 0.75, 0.85],
        macro_recall_false_alarms=0,
    )

    assert accuracy.metric_name == "kws_accuracy"
    assert accuracy.score == 0.75
    assert macro_recall.metric_name == "macro-recall"
    assert macro_recall.score == 0.5
    assert macro_recall.details["macro_recall"] == 0.5


def test_kws_pipeline_penalizes_wrong_keyword() -> None:
    from sure_eval.evaluation.tasks.kws import KWSSample, KWSMetricPipeline

    sample = KWSSample(
        key="pos",
        expected_detected=True,
        expected_keyword="嗨小问",
        duration=1.0,
        detected=True,
        predicted_keyword="你好问问",
        score=0.99,
    )

    report = KWSMetricPipeline(thresholds=[0.5]).evaluate([sample])

    assert report.results["accuracy"].score == 0.0
    assert report.results["recall"].score == 0.0
    assert report.results["false_reject_rate"].score == 1.0
    assert report.rows[0]["correct"] is False
    assert report.rows[0]["error_type"] == "wrong_keyword"


def test_kws_loads_sure_json_outputs(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.kws.loaders import load_samples_from_jsonl_and_outputs

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
                {
                    "key": "pos",
                    "result": {
                        "detected": True,
                        "keyword": "嗨小问",
                        "score": 0.91,
                    },
                },
                {
                    "key": "neg",
                    "result": {
                        "detected": False,
                        "keyword": None,
                        "score": None,
                    },
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    samples = load_samples_from_jsonl_and_outputs(gt, outputs)

    assert [sample.key for sample in samples] == ["pos", "neg"]
    assert samples[0].expected_detected is True
    assert samples[0].expected_keyword == "嗨小问"
    assert samples[0].detected is True
    assert samples[0].score == 0.91
    assert samples[1].expected_detected is False


def test_kws_loads_wekws_score_format(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.kws.loaders import load_samples_from_wekws_score_file

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
    scores.write_text(
        "\n".join(
            [
                "pos detected 嗨小问 0.930",
                "neg rejected",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_samples_from_wekws_score_file(labels, scores, keyword="嗨小问")

    assert len(samples) == 2
    assert samples[0].expected_detected is True
    assert samples[0].detected is True
    assert samples[0].predicted_keyword == "嗨小问"
    assert samples[0].score == 0.93
    assert samples[1].expected_detected is False
    assert samples[1].detected is False


def test_kws_loads_wekws_frame_score_format(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.kws.loaders import load_samples_from_wekws_frame_score_file

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
    scores.write_text(
        "\n".join(
            [
                "pos 嗨小问 0.100000 0.930000 0.700000",
                "neg 嗨小问 0.100000 0.200000 0.300000",
                "pos 其他 0.990000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_samples_from_wekws_frame_score_file(labels, scores, keyword="嗨小问")

    assert len(samples) == 2
    assert samples[0].expected_detected is True
    assert samples[0].scores == [0.1, 0.93, 0.7]
    assert samples[0].score == 0.93
    assert samples[1].expected_detected is False
    assert samples[1].score == 0.3
