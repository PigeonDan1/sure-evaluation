from __future__ import annotations

from dataclasses import replace
import math
import pytest

from sure_eval.evaluation.tasks.kws import KWSEvent, KWSMetricPipeline, KWSSample, KWSTrigger


def positive(key="pos", keyword="A", score=0.9, **kwargs):
    return KWSSample(
        key=key,
        expected_detected=True,
        expected_keyword=keyword,
        predicted_keyword=keyword,
        score=score,
        **kwargs,
    )


def negative(duration=3600.0, score=0.8, **kwargs):
    return KWSSample(key="neg", expected_detected=False, duration=duration, score=score, **kwargs)


def test_default_uses_rate_budget_and_duration_changes_operating_point():
    positives = [positive("high", score=0.9), positive("low", score=0.7)]
    short = KWSMetricPipeline().evaluate([*positives, negative(3600)])
    long = KWSMetricPipeline().evaluate([*positives, negative(36000)])
    assert short.summary["target_fa_per_hour"] == 0.1
    assert short.results["macro_recall"].score == 0.5
    assert short.summary["false_alarms"] == 0
    assert long.results["macro_recall"].score == 1
    assert long.summary["false_alarm_per_hour"] == 0.1
    assert long.summary["false_alarms"] == 1


def test_macro_is_equal_keyword_average_at_one_shared_threshold():
    samples = [positive(f"a{i}") for i in range(9)]
    samples += [replace(positive("b", "B"), predicted_keyword="WRONG"), negative(score=0.1)]
    report = KWSMetricPipeline().evaluate(samples)
    assert report.results["macro_recall"].score == 0.5
    assert report.results["per_keyword_recall"].details == {
        "A": {"recall": 1, "true_positives": 9, "positive_events": 9},
        "B": {"recall": 0, "true_positives": 0, "positive_events": 1},
    }
    assert report.summary["threshold_scope"] == "global"
    assert set(report.results) == {"macro_recall", "per_keyword_recall", "trigger_latency"}


def test_keyword_normalization_and_detected_flag_do_not_censor_raw_scores():
    report = KWSMetricPipeline().evaluate(
        [
            replace(
                positive(keyword="Hello World", score=0.1234),
                predicted_keyword=" helloWORLD ",
                detected=False,
            ),
            negative(score=0.12),
        ]
    )
    assert report.results["macro_recall"].score == 1
    assert report.summary["threshold"] == 0.1234


def test_tied_scores_are_atomic_and_all_rejected_point_exists():
    report = KWSMetricPipeline(target_fa_per_hour=0, thresholds=[0.5]).evaluate(
        [positive(score=1), negative(score=1)]
    )
    assert report.results["macro_recall"].score == 0
    assert report.summary["threshold"] > 1
    assert math.isfinite(report.summary["threshold"])
    assert report.summary["false_alarms"] == 0


def test_multiple_false_alarm_events_share_system_budget():
    samples = [
        positive(score=0.7),
        KWSSample(
            key="neg",
            expected_detected=False,
            duration=36000,
            detections=[KWSTrigger("A", 0.8, 10), KWSTrigger("B", 0.75, 20)],
        ),
    ]
    strict = KWSMetricPipeline().evaluate(samples)
    loose = KWSMetricPipeline(target_fa_per_hour=0.2).evaluate(samples)
    assert strict.results["macro_recall"].score == 0
    assert loose.results["macro_recall"].score == 1
    assert loose.summary["false_alarms"] == 2
    assert loose.summary["false_alarm_per_hour"] == 0.2


def test_latency_uses_first_correct_trigger_and_preserves_early_detection():
    samples = [
        KWSSample(
            key="stream",
            expected_detected=True,
            duration=6,
            events=[KWSEvent("A", 1, 0.5), KWSEvent("A", 4, 3.5)],
            detections=[
                KWSTrigger("A", 0.9, 1.2),
                KWSTrigger("A", 0.9, 1.4),
                KWSTrigger("A", 0.9, 3.9),
            ],
        ),
        negative(score=0.1),
    ]
    report = KWSMetricPipeline().evaluate(samples)
    latency = report.results["trigger_latency"]
    assert [row["latency_ms"] for row in report.rows] == pytest.approx([200, -100])
    assert latency.score == pytest.approx(50)
    assert latency.details["overall"]["p50_ms"] == pytest.approx(50)
    assert latency.details["overall"]["p95_ms"] == pytest.approx(185)
    assert latency.details["overall"]["count"] == 2
    assert report.summary["per_keyword"]["A"]["true_positives"] == 2
    assert report.summary["unmatched_triggers"] == 1


def test_unmatched_positive_trigger_is_diagnostic_only_for_background_fa_rate():
    from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import KWSMetric

    result = KWSMetric().calculate_samples(
        [
            KWSSample(
                key="positive-stream",
                expected_detected=True,
                duration=5,
                events=[KWSEvent("A", 1, 0.5)],
                detections=[KWSTrigger("WRONG", 0.9, 3)],
            ),
            negative(duration=3600, score=0.1),
        ]
    )

    point = next(item for item in result.details["operating_curve"] if item["threshold"] == pytest.approx(0.9))
    assert point["false_alarms"] == 0
    assert point["false_alarm_per_hour"] == 0.0
    assert point["unmatched_triggers"] == 1


def test_one_trigger_cannot_match_two_reference_events():
    report = KWSMetricPipeline().evaluate(
        [
            KWSSample(
                key="p",
                expected_detected=True,
                events=[KWSEvent("A", 1), KWSEvent("A", 1.5)],
                detections=[KWSTrigger("A", 0.9, 1.2)],
            ),
            negative(score=0.1),
        ]
    )
    assert report.results["macro_recall"].score == 0.5
    assert [row["matched"] for row in report.rows] == [True, False]


def test_out_of_window_trigger_is_a_miss():
    report = KWSMetricPipeline(match_tolerance_sec=0.2).evaluate(
        [positive(keyword_end_sec=1, trigger_time_sec=1.5), negative(score=0.1)]
    )
    assert report.results["macro_recall"].score == 0
    assert report.results["trigger_latency"].score is None


def test_missing_timestamps_report_latency_coverage_instead_of_zero():
    report = KWSMetricPipeline().evaluate(
        [
            positive("timed", keyword_end_sec=1, trigger_time_sec=1.2),
            positive("untimed"),
            negative(score=0.1),
        ]
    )
    overall = report.results["trigger_latency"].details["overall"]
    assert overall["matched_events"] == 2
    assert overall["count"] == 1
    assert overall["coverage"] == 0.5
    assert overall["mean_ms"] == pytest.approx(200)


def test_frame_latency_uses_threshold_crossing_not_peak_and_no_prior_gate():
    report = KWSMetricPipeline().evaluate(
        [
            KWSSample(
                key="p",
                expected_detected=True,
                expected_keyword="A",
                predicted_keyword="A",
                detected=False,
                scores=[0.1, 0.6, 0.9],
                frame_shift_sec=0.1,
                keyword_end_sec=0.15,
            ),
            negative(score=0.1),
        ]
    )
    assert report.results["macro_recall"].score == 1
    # Tie policy selects the higher threshold (.9); use a fixed search to exercise .6.
    fixed = KWSMetricPipeline(thresholds=[0.6]).evaluate(
        [
            KWSSample(
                key="p",
                expected_detected=True,
                expected_keyword="A",
                predicted_keyword="A",
                scores=[0.1, 0.6, 0.9],
                frame_shift_sec=0.1,
                keyword_end_sec=0.15,
            ),
            negative(score=0.1),
        ]
    )
    assert fixed.rows[0]["trigger_time_sec"] == pytest.approx(0.2)
    assert fixed.rows[0]["latency_ms"] == pytest.approx(50)


@pytest.mark.parametrize(
    "scores,expected_auc", [([0.9, 0.1], 1), ([0.1, 0.9], 0), ([0.5, 0.5], 0.5)]
)
def test_optional_roc_has_standard_fpr_axis_and_auc(scores, expected_auc):
    report = KWSMetricPipeline(include_roc=True).evaluate(
        [positive(score=scores[0]), negative(score=scores[1])]
    )
    assert report.results["auc"].score == expected_auc
    roc = report.results["roc_curve"].details
    assert roc["x_axis"] == "false_positive_rate"
    points = roc["per_keyword"]["A"]["points"]
    assert (points[0]["fpr"], points[0]["tpr"]) == (0, 0)
    assert (points[-1]["fpr"], points[-1]["tpr"]) == (1, 1)


def test_multiclass_roc_requires_complete_per_keyword_scores():
    samples = [positive("a"), positive("b", "B"), negative(score=0.1)]
    missing = KWSMetricPipeline(include_roc=True).evaluate(samples)
    assert missing.results["auc"].score is None
    assert not missing.results["roc_curve"].details["available"]
    samples = [
        replace(samples[0], keyword_scores={"A": 0.9, "B": 0.1}),
        replace(samples[1], keyword_scores={"A": 0.1, "B": 0.9}),
        replace(samples[2], keyword_scores={"A": 0.1, "B": 0.1}),
    ]
    assert KWSMetricPipeline(include_roc=True).evaluate(samples).results["auc"].score == 1


def test_event_stream_roc_is_explicitly_unavailable():
    report = KWSMetricPipeline(include_roc=True).evaluate(
        [positive(), KWSSample(key="n", expected_detected=False, duration=3600, detections=[])]
    )
    assert report.results["auc"].score is None
    assert "event lists" in report.results["roc_curve"].details["reason"]


@pytest.mark.parametrize("duration", [None, 0, -1, float("nan"), float("inf")])
def test_invalid_negative_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="duration"):
        KWSMetricPipeline().evaluate([positive(), negative(duration)])


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_invalid_rate_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="target_fa_per_hour"):
        KWSMetricPipeline(target_fa_per_hour=budget).evaluate([positive(), negative()])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_are_rejected(score):
    with pytest.raises(ValueError, match="score"):
        KWSMetricPipeline().evaluate([positive(score=score), negative()])


def test_missing_positive_negative_or_raw_scores_are_rejected():
    for samples, message in [
        ([], "requires samples"),
        ([positive()], "negative audio"),
        ([negative()], "positive keyword"),
        ([positive(), negative(score=None)], "Raw score"),
    ]:
        with pytest.raises(ValueError, match=message):
            KWSMetricPipeline().evaluate(samples)


def test_empty_candidate_list_is_valid_and_duplicate_keys_are_rejected():
    samples = [
        positive(),
        KWSSample(key="n", expected_detected=False, duration=3600, detections=[]),
    ]
    assert KWSMetricPipeline().evaluate(samples).results["macro_recall"].score == 1
    with pytest.raises(ValueError, match="Duplicate"):
        KWSMetricPipeline().evaluate([*samples, samples[0]])


def test_registry_removes_kws_accuracy_and_exposes_new_macro():
    from sure_eval.evaluation.registry import MetricRegistry

    assert "kws_accuracy" not in MetricRegistry.list_metrics()
    result = MetricRegistry.get_metric("kws_macro_recall").calculate_samples(
        [positive(), negative()]
    )
    assert result.metric_name == "macro_recall"
    assert result.score == 1


def test_optional_roc_does_not_relabel_another_keywords_score():
    report = KWSMetricPipeline(include_roc=True).evaluate(
        [replace(positive(), predicted_keyword="B"), negative(score=0.1)]
    )
    assert report.results["macro_recall"].score == 0
    assert report.results["auc"].score is None
    assert "another predicted keyword" in report.results["roc_curve"].details["reason"]


def test_timestamp_and_multievent_validation():
    invalid = [
        positive(keyword_end_sec=-1),
        positive(keyword_end_sec=1, keyword_start_sec=2),
        positive(trigger_time_sec=float("nan")),
        positive(duration=1, keyword_end_sec=2),
        KWSSample(
            key="p",
            expected_detected=True,
            events=[KWSEvent("A", 1), KWSEvent("A", 2)],
            detections=[KWSTrigger("A", 0.9)],
        ),
        replace(positive(), detections=[]),
    ]
    for sample in invalid:
        with pytest.raises(ValueError):
            KWSMetricPipeline().evaluate([sample, negative()])


def test_frame_first_crossing_can_leave_matching_window_as_threshold_drops():
    from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import KWSMetric

    samples = [
        KWSSample(
            key="p",
            expected_detected=True,
            expected_keyword="A",
            predicted_keyword="A",
            scores=[0.6, 0.9],
            frame_shift_sec=1,
            keyword_end_sec=2,
        ),
        negative(score=0.1),
    ]
    result = KWSMetric(match_tolerance_sec=0.1).calculate_samples(samples)
    points = {point["threshold"]: point for point in result.details["operating_curve"]}
    assert points[0.6]["macro_recall"] == 0
    assert points[0.9]["macro_recall"] == 1
    assert result.score == 1
