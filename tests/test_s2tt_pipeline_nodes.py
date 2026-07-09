from __future__ import annotations

from pathlib import Path


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def test_s2tt_zh_pipeline_matches_sure_evaluator(tmp_path: Path) -> None:
    from sure_eval.evaluation.sure_evaluator import SUREEvaluator
    from sure_eval.evaluation.tasks.s2tt.pipeline import evaluate_s2tt_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好世界。"), ("utt2", "今天天气很好。")])
    _write_key_text(hyp_file, [("utt1", "你好世界。"), ("utt2", "今天的天气很好。")])

    legacy = SUREEvaluator(language="zh").evaluate("S2TT", str(ref_file), str(hyp_file))
    report = evaluate_s2tt_files(str(ref_file), str(hyp_file), language="zh", metric="bleu")

    assert report.task == "S2TT"
    assert report.language == "zh"
    assert report.metric == "bleu"
    assert report.score == legacy["score"]
    assert report.details["scoring_result"] == legacy
    assert report.details["input_contract"]["required_roles"] == ["hyp", "ref"]
    assert report.details["input_contract"]["aggregation"] == "corpus_metric"
    assert report.details["input_files"] == {"ref": str(ref_file), "hyp": str(hyp_file)}
    assert len(report.pipeline_trace) == 1
    assert report.pipeline_trace[0].stage == "scoring"
    assert report.pipeline_trace[0].node_id == "scoring/sacrebleu"
    assert report.pipeline_trace[0].details["tokenizer_profile"] == "zh"
    assert report.pipeline_trace[0].details["tokenizer"] == "zh"
    assert report.pipeline_trace[0].internal_stages == (
        "tokenizer_selection",
        "corpus_bleu",
        "corpus_chrf2",
    )


def test_s2tt_model_artifact_pipeline_matches_sure_evaluator() -> None:
    from sure_eval.evaluation.sure_evaluator import SUREEvaluator
    from sure_eval.evaluation.tasks.s2tt.pipeline import evaluate_s2tt_files

    ref_file = "src/sure_eval/models/moonshotai__Kimi-Audio-7B-Instruct/artifacts/ref_s2tt.txt"
    hyp_file = "src/sure_eval/models/moonshotai__Kimi-Audio-7B-Instruct/artifacts/hyp_s2tt.txt"

    legacy = SUREEvaluator(language="zh").evaluate("S2TT", ref_file, hyp_file)
    report = evaluate_s2tt_files(ref_file, hyp_file, language="zh", metric="bleu")

    assert report.details["scoring_result"] == legacy
    assert report.score == legacy["score"]
    assert report.pipeline_id == "s2tt.zh.bleu.sacrebleu"
    assert report.details["input_contract"]["metric_id"] == "scoring/sacrebleu"


def test_s2tt_xcomet_pipeline_uses_src_hyp_ref_and_segment_mean(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.xcomet_xl.node import SegmentScore
    from sure_eval.evaluation.tasks.s2tt.pipeline import evaluate_s2tt_files

    src_file = tmp_path / "src.txt"
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(src_file, [("utt1", "hello world"), ("utt2", "good morning")])
    _write_key_text(ref_file, [("utt1", "你好世界。"), ("utt2", "早上好。")])
    _write_key_text(hyp_file, [("utt1", "你好世界。"), ("utt2", "早安。")])

    seen_rows: list[dict[str, str]] = []

    def runner(rows: list[dict[str, str]]) -> list[SegmentScore]:
        seen_rows.extend(rows)
        return [
            SegmentScore(key=rows[0]["key"], score=0.9),
            SegmentScore(key=rows[1]["key"], score=0.7),
        ]

    report = evaluate_s2tt_files(
        str(ref_file),
        str(hyp_file),
        language="zh",
        metric="xcomet_xl",
        src_file=str(src_file),
        xcomet_runner=runner,
    )

    assert seen_rows == [
        {"key": "utt1", "src": "hello world", "hyp": "你好世界。", "ref": "你好世界。"},
        {"key": "utt2", "src": "good morning", "hyp": "早安。", "ref": "早上好。"},
    ]
    assert report.score == 0.8
    assert report.pipeline_id == "s2tt.zh.xcomet_xl.xcomet_xl"
    assert report.details["input_contract"]["required_roles"] == ["src", "hyp", "ref"]
    assert report.details["input_contract"]["aggregation"] == "segment_mean"
    assert report.details["input_files"] == {
        "src": str(src_file),
        "ref": str(ref_file),
        "hyp": str(hyp_file),
    }
    assert report.details["scoring_result"]["segment_scores"] == [0.9, 0.7]
    assert report.pipeline_trace[0].node_id == "scoring/xcomet_xl"


def test_s2tt_xcomet_requires_source_file(tmp_path: Path) -> None:
    import pytest

    from sure_eval.evaluation.tasks.s2tt.pipeline import evaluate_s2tt_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好世界。")])
    _write_key_text(hyp_file, [("utt1", "你好世界。")])

    with pytest.raises(ValueError, match="src_file is required"):
        evaluate_s2tt_files(
            str(ref_file),
            str(hyp_file),
            language="zh",
            metric="xcomet_xl",
        )


def test_s2tt_bleurt_pipeline_uses_hyp_ref_and_segment_mean(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.bleurt_20.node import SegmentScore
    from sure_eval.evaluation.tasks.s2tt.pipeline import evaluate_s2tt_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好世界。"), ("utt2", "早上好。")])
    _write_key_text(hyp_file, [("utt1", "你好世界。"), ("utt2", "早安。")])

    seen_rows: list[dict[str, str]] = []

    def runner(rows: list[dict[str, str]]) -> list[SegmentScore]:
        seen_rows.extend(rows)
        return [
            SegmentScore(key=rows[0]["key"], score=0.8),
            SegmentScore(key=rows[1]["key"], score=0.6),
        ]

    report = evaluate_s2tt_files(
        str(ref_file),
        str(hyp_file),
        language="zh",
        metric="bleurt_20",
        bleurt_runner=runner,
    )

    assert seen_rows == [
        {"key": "utt1", "hyp": "你好世界。", "ref": "你好世界。"},
        {"key": "utt2", "hyp": "早安。", "ref": "早上好。"},
    ]
    assert report.score == 0.7
    assert report.pipeline_id == "s2tt.zh.bleurt_20.bleurt_20"
    assert report.details["input_contract"]["required_roles"] == ["hyp", "ref"]
    assert report.details["input_contract"]["aggregation"] == "segment_mean"
    assert report.details["scoring_result"]["segment_scores"] == [0.8, 0.6]
    assert report.pipeline_trace[0].node_id == "scoring/bleurt_20"
