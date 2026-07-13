from __future__ import annotations

from pathlib import Path


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def _assert_matches_legacy(actual: dict, legacy: dict) -> None:
    """Every legacy field must match; scoring results may add coverage fields."""

    for key, value in legacy.items():
        if isinstance(value, dict):
            _assert_matches_legacy(actual[key], value)
        else:
            assert actual[key] == value, f"legacy field {key!r} diverged"


def _fake_wetext_normalizer(files, *, profile: str):
    from sure_eval.evaluation.core.types import PipelineNodeResult

    return (
        files,
        PipelineNodeResult(
            stage="normalization",
            node_id="normalization/wetext_norm",
            version="v1",
            details={"profile": profile, "language": profile.split("_", 1)[0], "direction": profile.split("_", 1)[1]},
            internal_stages=("fake_wetext",),
        ),
    )


def _fake_sctk_sclite_wer(files):
    from sure_eval.evaluation.core.types import PipelineNodeResult

    result = {
        "metric_name": "wer",
        "score": 0.0,
        "wer": 0.0,
        "wer_percent": 0.0,
        "all": 2,
        "cor": 2,
        "sub": 0,
        "del": 0,
        "ins": 0,
    }
    return (
        files,
        PipelineNodeResult(
            stage="scoring",
            node_id="scoring/sctk_sclite",
            version="v1",
            details={"metric": "wer", "result": result},
            internal_stages=("fake_sclite",),
        ),
    )


def _fake_sctk_sclite_cer(files):
    from sure_eval.evaluation.core.types import PipelineNodeResult

    result = {
        "metric_name": "cer",
        "score": 0.0,
        "cer": 0.0,
        "cer_percent": 0.0,
        "all": 4,
        "cor": 4,
        "sub": 0,
        "del": 0,
        "ins": 0,
    }
    return (
        files,
        PipelineNodeResult(
            stage="scoring",
            node_id="scoring/sctk_sclite",
            version="v1",
            details={"metric": "cer", "result": result},
            internal_stages=("fake_sclite",),
        ),
    )


def test_asr_zh_cer_pipeline_matches_sure_evaluator(tmp_path: Path) -> None:
    from sure_eval.evaluation.sure_evaluator import SUREEvaluator
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "我有2个苹果。"), ("utt2", "你好世界")])
    _write_key_text(hyp_file, [("utt1", "我有两个苹果"), ("utt2", "你好世")])

    legacy = SUREEvaluator(language="zh").evaluate("ASR", str(ref_file), str(hyp_file), tochar=True)
    report = evaluate_asr_files(str(ref_file), str(hyp_file), language="zh", metric="cer")

    assert report.task == "ASR"
    assert report.language == "zh"
    assert report.metric == "cer"
    assert report.score == legacy["score"]
    _assert_matches_legacy(report.details["scoring_result"], legacy)
    assert report.details["input_contract"]["required_roles"] == ["hyp", "ref"]
    assert report.details["input_contract"]["aggregation"] == "corpus_edit_distance"
    assert report.details["input_files"] == {"ref": str(ref_file), "hyp": str(hyp_file)}
    assert report.pipeline_trace[0].stage == "normalization"
    assert report.pipeline_trace[0].node_id == "normalization/aispeech_norm"
    assert report.pipeline_trace[0].details["profile"] == "zh"
    assert report.pipeline_trace[1].stage == "scoring"
    assert report.pipeline_trace[1].node_id == "scoring/wenet_cer"
    assert report.pipeline_trace[1].internal_stages == ("tokenization", "case_normalization", "edit_distance")


def test_asr_en_wer_pipeline_uses_whisper_normalization_by_default(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "I have two apples.")])
    _write_key_text(hyp_file, [("utt1", "i have 2 apples")])

    report = evaluate_asr_files(str(ref_file), str(hyp_file), language="en", metric="wer")

    assert report.task == "ASR"
    assert report.language == "en"
    assert report.metric == "wer"
    assert report.score == 0.0
    assert report.pipeline_id == "asr.en.wer.whisper_norm.wenet_wer"
    assert report.details["input_contract"]["required_roles"] == ["hyp", "ref"]
    assert report.details["input_contract"]["metric_id"] == "scoring/wenet_wer"
    assert report.pipeline_trace[0].node_id == "normalization/whisper_norm"
    assert report.pipeline_trace[0].details["profile"] == "english"
    assert report.pipeline_trace[0].details["normalization"]["normalizer_class"] == "EnglishTextNormalizer"
    assert report.pipeline_trace[1].node_id == "scoring/wenet_wer"


def test_asr_en_wer_pipeline_can_use_legacy_aispeech_normalization(tmp_path: Path) -> None:
    from sure_eval.evaluation.sure_evaluator import SUREEvaluator
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "hello world"), ("utt2", "I have 2 apples.")])
    _write_key_text(hyp_file, [("utt1", "hello brave world"), ("utt2", "I have two apples")])

    legacy = SUREEvaluator(language="en").evaluate("ASR", str(ref_file), str(hyp_file), tochar=False)
    report = evaluate_asr_files(
        str(ref_file),
        str(hyp_file),
        language="en",
        metric="wer",
        normalizer="aispeech",
    )

    assert report.score == legacy["score"]
    _assert_matches_legacy(report.details["scoring_result"], legacy)
    assert report.pipeline_id == "asr.en.wer.aispeech_norm.wenet_wer"
    assert report.pipeline_trace[0].node_id == "normalization/aispeech_norm"
    assert report.pipeline_trace[0].details["profile"] == "en"


def test_asr_can_explicitly_use_sctk_sclite_scorer(tmp_path: Path, monkeypatch) -> None:
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline

    monkeypatch.setattr(asr_pipeline, "score_sctk_sclite_wer", _fake_sctk_sclite_wer)
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "hello world")])
    _write_key_text(hyp_file, [("utt1", "hello world")])

    report = asr_pipeline.evaluate_asr_files(
        str(ref_file),
        str(hyp_file),
        language="en",
        metric="wer",
        scorer="sctk_sclite",
    )

    assert report.pipeline_id == "asr.en.wer.whisper_norm.sctk_sclite_wer"
    assert report.score == 0.0
    assert [node.node_id for node in report.pipeline_trace] == [
        "normalization/whisper_norm",
        "scoring/sctk_sclite",
    ]


def test_asr_can_combine_wetext_normalizer_and_sctk_sclite_scorer(tmp_path: Path, monkeypatch) -> None:
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline

    monkeypatch.setattr(asr_pipeline, "normalize_wetext_key_text_files", _fake_wetext_normalizer)
    monkeypatch.setattr(asr_pipeline, "score_sctk_sclite_cer", _fake_sctk_sclite_cer)
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好世界")])
    _write_key_text(hyp_file, [("utt1", "你好世界")])

    report = asr_pipeline.evaluate_asr_files(
        str(ref_file),
        str(hyp_file),
        language="zh",
        metric="cer",
        normalizer="wetext:zh_tn",
        scorer="sctk_sclite",
    )

    assert report.pipeline_id == "asr.zh.cer.wetext_zh_tn.sctk_sclite_cer"
    assert [node.node_id for node in report.pipeline_trace] == [
        "normalization/wetext_norm",
        "scoring/sctk_sclite",
    ]


def test_asr_codeswitch_rejects_explicit_sctk_sclite_scorer(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "hello 世界")])
    _write_key_text(hyp_file, [("utt1", "hello 世界")])

    try:
        evaluate_asr_files(
            str(ref_file),
            str(hyp_file),
            language="cs",
            metric="mer",
            scorer="sctk_sclite",
        )
    except ValueError as exc:
        assert "does not support explicit scorer selection" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_asr_can_explicitly_use_wetext_normalizer_without_changing_defaults(tmp_path: Path, monkeypatch) -> None:
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline

    monkeypatch.setattr(asr_pipeline, "normalize_wetext_key_text_files", _fake_wetext_normalizer)
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好世界")])
    _write_key_text(hyp_file, [("utt1", "你好世界")])

    report = asr_pipeline.evaluate_asr_files(
        str(ref_file),
        str(hyp_file),
        language="zh",
        metric="cer",
        normalizer="wetext:zh_tn",
    )

    assert report.pipeline_id == "asr.zh.cer.wetext_zh_tn.wenet_cer"
    assert report.score == 0.0
    assert [node.node_id for node in report.pipeline_trace] == [
        "normalization/wetext_norm",
        "scoring/wenet_cer",
    ]
    assert report.pipeline_trace[0].details["profile"] == "zh_tn"


def test_asr_can_explicitly_use_punctuation_strip_normalizer(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好，世界！")])
    _write_key_text(hyp_file, [("utt1", "你好世界")])

    report = evaluate_asr_files(
        str(ref_file),
        str(hyp_file),
        language="zh",
        metric="cer",
        normalizer="punctuation_strip",
    )

    assert report.pipeline_id == "asr.zh.cer.punctuation_strip_norm.wenet_cer"
    assert report.score == 0.0
    assert [node.node_id for node in report.pipeline_trace] == [
        "normalization/punctuation_strip_norm",
        "scoring/wenet_cer",
    ]
    assert (
        report.pipeline_trace[0].details["normalization"]["operation"]
        == "punctuation_stripping_only"
    )


def test_asr_rejects_wetext_profile_that_does_not_match_language(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "hello")])
    _write_key_text(hyp_file, [("utt1", "hello")])

    try:
        evaluate_asr_files(
            str(ref_file),
            str(hyp_file),
            language="zh",
            metric="cer",
            normalizer="wetext:en_tn",
        )
    except ValueError as exc:
        assert "does not match ASR language" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_asr_rejects_unknown_wetext_profile_before_node_execution(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好")])
    _write_key_text(hyp_file, [("utt1", "你好")])

    try:
        evaluate_asr_files(
            str(ref_file),
            str(hyp_file),
            language="zh",
            metric="cer",
            normalizer="wetext:zh_unknown",
        )
    except ValueError as exc:
        assert "Unsupported wetext_norm profile" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_asr_metric_class_forwards_explicit_normalizer(tmp_path: Path, monkeypatch) -> None:
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline
    from sure_eval.evaluation.tasks.asr.metrics import CERMetric

    monkeypatch.setattr(asr_pipeline, "normalize_wetext_key_text_files", _fake_wetext_normalizer)

    result = CERMetric().calculate("你好世界", "你好世界", language="zh", normalizer="wetext:zh_tn")

    assert result.score == 0.0
    assert result.details["pipeline_id"] == "asr.zh.cer.wetext_zh_tn.wenet_cer"
    assert result.details["pipeline_trace"][0]["node_id"] == "normalization/wetext_norm"
    assert result.details["pipeline_trace"][0]["profile"] == "zh_tn"


def test_asr_codeswitch_mer_pipeline_matches_sure_evaluator(tmp_path: Path) -> None:
    from sure_eval.evaluation.sure_evaluator import SUREEvaluator
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "hello 世界"), ("utt2", "我有 2 apples")])
    _write_key_text(hyp_file, [("utt1", "hello 世"), ("utt2", "我有 two apples")])

    legacy = SUREEvaluator(language="cs").evaluate("ASR", str(ref_file), str(hyp_file))
    report = evaluate_asr_files(str(ref_file), str(hyp_file), language="cs", metric="mer")

    assert report.task == "ASR"
    assert report.language == "cs"
    assert report.metric == "mer"
    assert report.score == legacy["score"]
    _assert_matches_legacy(report.details["scoring_result"], legacy)
    assert report.details["input_contract"]["required_roles"] == ["hyp", "ref"]
    assert report.details["input_contract"]["metric_id"] == "scoring/wenet_mer"
    assert report.pipeline_trace[0].node_id == "normalization/aispeech_norm"
    assert report.pipeline_trace[0].details["profile"] == "cs"
    assert report.pipeline_trace[1].node_id == "scoring/wenet_mer"
