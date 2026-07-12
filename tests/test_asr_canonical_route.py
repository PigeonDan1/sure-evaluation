from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cn2an")
pytest.importorskip("rapidfuzz")


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def _score_pair(tmp_path: Path, ref: str, hyp: str) -> float:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", ref)])
    _write_key_text(hyp_file, [("utt1", hyp)])
    report = evaluate_asr_files(str(ref_file), str(hyp_file), language="zh", metric="cer_canonical")
    return report.score


# Judge pairs: written-form variants of the same speech must score 0.
EQUIVALENT_PAIRS = [
    ("二零二四年七月", "2024年7月"),          # digit-by-digit year reading
    ("两千零二十四", "2024"),                  # quantity reading
    ("增长百分之五十", "增长50%"),             # percent word form
    ("增长百分之一百五十", "增长150%"),
    ("百分之百完成", "100%完成"),
    ("预算四千五百亿元", "预算4500亿元"),      # mixed written number
    ("规模三万亿", "规模3万亿"),
    ("一五一十地说清楚", "一五一十地说清楚"),  # numeral-bearing idiom protected
    ("你好，世界！", "你好 世界"),             # punctuation / spacing
    ("ＨＥＬＬＯ ｗｏｒｌｄ", "hello world"),   # NFKC + lowercase
]

# Real differences must stay errors.
ERROR_PAIRS = [
    ("预算四千五百亿元", "预算四千五百万元"),   # magnitude error
    ("增长百分之五十", "增长百分之十五"),       # value error
    ("零下三度", "三度"),                       # dropped negative/qualifier
    ("小数是四一点三", "小数是四一三"),         # decimal point placement
    ("今天天气很好", "今天天气很号"),           # plain substitution
]


@pytest.mark.parametrize("ref,hyp", EQUIVALENT_PAIRS)
def test_canonical_cer_treats_written_form_variants_as_equal(tmp_path: Path, ref: str, hyp: str) -> None:
    assert _score_pair(tmp_path, ref, hyp) == 0.0


@pytest.mark.parametrize("ref,hyp", ERROR_PAIRS)
def test_canonical_cer_keeps_real_errors(tmp_path: Path, ref: str, hyp: str) -> None:
    assert _score_pair(tmp_path, ref, hyp) > 0.0


def test_canonical_route_describe_and_run(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts import describe_pipeline, run_task

    description = describe_pipeline("asr", language="zh", metric="cer_canonical")
    assert description.pipeline_id == "asr.zh.cer_canonical.canonical_itn.token_cer"
    assert description.node_ids == ("normalization/canonical_itn", "scoring/token_cer")

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "增长百分之五十"), ("utt2", "预算四千五百亿元")])
    _write_key_text(hyp_file, [("utt1", "增长50%"), ("utt2", "预算4500亿元")])
    report = run_task(
        "asr",
        ref_file=str(ref_file),
        hyp_file=str(hyp_file),
        language="zh",
        metric="cer_canonical",
        output_dir=str(tmp_path / "eval"),
    )
    assert report.pipeline_id == "asr.zh.cer_canonical.canonical_itn.token_cer"
    assert report.score == 0.0
    assert (tmp_path / "eval" / "report.json").exists()
    assert (tmp_path / "eval" / "pipeline_description.json").exists()
    norm_details = report.pipeline_trace[0].details
    assert norm_details["engine"]["engine"] == "cn2an"
    assert "engine_version" in norm_details["engine"]


def test_canonical_cer_scores_empty_and_missing_hypotheses_as_deletions(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "今天天气很好"), ("utt2", "我们一起去公园"), ("utt3", "你好")])
    _write_key_text(hyp_file, [("utt1", "今天天气很好"), ("utt2", "")])  # utt3 missing entirely

    report = evaluate_asr_files(str(ref_file), str(hyp_file), language="zh", metric="cer_canonical")
    result = report.details["scoring_result"]
    assert result["all"] == 15
    assert result["del"] == 9
    assert result["num_hyp_missing_utts"] == 1
    assert result["hyp_missing_policy"] == "scored_as_empty_hypothesis"
    assert report.score == pytest.approx(9 / 15)


def test_canonical_cer_rejects_zero_reference_tokens(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1 今天天气很好\n", encoding="utf-8")  # space-separated: malformed
    hyp_file.write_text("utt1 今天天气很好\n", encoding="utf-8")

    with pytest.raises(ValueError):
        evaluate_asr_files(str(ref_file), str(hyp_file), language="zh", metric="cer_canonical")


def test_canonical_metric_rejects_foreign_normalizer_and_scorer(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好")])
    _write_key_text(hyp_file, [("utt1", "你好")])

    with pytest.raises(ValueError, match="canonical_itn"):
        evaluate_asr_files(
            str(ref_file), str(hyp_file), language="zh", metric="cer_canonical", normalizer="aispeech"
        )
    with pytest.raises(ValueError, match="token_cer"):
        evaluate_asr_files(
            str(ref_file), str(hyp_file), language="zh", metric="cer_canonical", scorer="sctk_sclite"
        )
    with pytest.raises(ValueError, match="cer_canonical"):
        evaluate_asr_files(
            str(ref_file), str(hyp_file), language="zh", metric="cer", scorer="token_cer"
        )


def test_canonical_cer_insertions_can_exceed_hundred_percent(tmp_path: Path) -> None:
    # Hallucination-style scoring: short reference, long fabricated hypothesis.
    score = _score_pair(tmp_path, "嗯", "嗯今天天气很好我们一起去公园")
    assert score > 1.0
