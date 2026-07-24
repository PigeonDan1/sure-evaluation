from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("cn2an")
pytest.importorskip("rapidfuzz")


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def _evaluate(tmp_path: Path, ref: str, hyp: str, *, language: str, metric: str, **kwargs):
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", ref)])
    _write_key_text(hyp_file, [("utt1", hyp)])
    return evaluate_asr_files(str(ref_file), str(hyp_file), language=language, metric=metric, **kwargs)


# --------------------------------------------------------------------------- #
# Degeneration guarantees (by construction, locked here)
# --------------------------------------------------------------------------- #
PURE_ZH_SENTENCES = [
    "今天天气很好",
    "我们一起去公园散步",
    "今年增长百分之五十",
    "预算四千五百亿元",
    "二零二四年七月六日开会",
    "会议室在三楼",
    "他说一五一十地讲清楚",
    "规模三万亿的市场",
    "涨了百分之一百五十",
    "零下三度的天气",
    "第十三次会议纪要",
    "两千零二十四个样本",
    "占比百分之百",
    "错误率下降了",
    "小数点后两位",
]


def test_pure_chinese_mixed_chain_equals_plain_chain() -> None:
    from sure_eval.evaluation.nodes.normalization.canonical_itn.chain import (
        normalize_text,
        normalize_text_mixed,
    )

    for sentence in PURE_ZH_SENTENCES:
        assert normalize_text_mixed(sentence) == normalize_text(sentence), sentence


def test_pure_chinese_mer_score_equals_cer_canonical(tmp_path: Path) -> None:
    pairs = list(zip(PURE_ZH_SENTENCES, PURE_ZH_SENTENCES[1:] + PURE_ZH_SENTENCES[:1]))
    for index, (ref, hyp) in enumerate(pairs[:5]):
        sub = tmp_path / f"case{index}"
        sub.mkdir()
        mer = _evaluate(sub, ref, hyp, language="cs", metric="mer_canonical")
        cer = _evaluate(sub, ref, hyp, language="zh", metric="cer_canonical")
        assert mer.score == cer.score
        mer_result = mer.details["scoring_result"]
        cer_result = cer.details["scoring_result"]
        for key in ("all", "cor", "sub", "del", "ins"):
            assert mer_result[key] == cer_result[key]


def test_pure_english_mer_equals_wer_canonical(tmp_path: Path) -> None:
    ref = "we need to check the final report"
    hyp = "we need to check the last report"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    mer = _evaluate(tmp_path / "a", ref, hyp, language="cs", metric="mer_canonical")
    wer = _evaluate(tmp_path / "b", ref, hyp, language="en", metric="wer_canonical")
    assert mer.score == wer.score
    assert mer.score == pytest.approx(1 / 7)


# --------------------------------------------------------------------------- #
# English judge pairs (Whisper normalization semantics)
# --------------------------------------------------------------------------- #
EQUIVALENT_PAIRS = [
    ("I haven't seen it", "i have not seen it"),     # contraction expansion
    ("don't do it", "dont do it"),                   # bare contraction restored
    ("I'm sure you're right", "im sure youre right"),
    ("it's fine", "its fine"),                       # 's collapsed, not expanded
    ("john's report is ready", "johns report is ready"),
    ("we need to produce it", "we needto produce it"),   # spacing repair (hyp glue)
    ("something great happened", "some thing great happened"),  # reverse split
    ("um okay that works", "okay that works"),       # spoken filler dropped
    ("uh hmm we agree", "we agree"),
    ("the colour is nice", "the color is nice"),     # British -> American fold
    ("fifty percent of them", "50% of them"),        # spoken number ITN
    ("one hundred and fifty dollars", "$150"),
]

ERROR_PAIRS = [
    ("we need the report", "we need the 报告"),      # CJK leakage stays an error
    ("fifty percent", "fifteen percent"),            # value error survives ITN
    ("check the report", "check report"),            # word deletion
    ("we needed to produce it", "we needto produce it"),  # letters differ: repair must NOT fire
    ("it is done", "it has done"),                   # real is/has difference
    # Forgone equivalence under the 's-collapse policy (inherent
    # non-transitivity: choosing it's==its and john's==johns gives up
    # it's==it is). Documented, not accidental.
    ("it's been okay", "it has been okay"),
]


@pytest.mark.parametrize("ref,hyp", EQUIVALENT_PAIRS)
def test_english_normalization_equivalents(tmp_path: Path, ref: str, hyp: str) -> None:
    report = _evaluate(tmp_path, ref, hyp, language="en", metric="wer_canonical")
    assert report.score == 0.0


@pytest.mark.parametrize("ref,hyp", ERROR_PAIRS)
def test_english_normalization_keeps_real_errors(tmp_path: Path, ref: str, hyp: str) -> None:
    report = _evaluate(tmp_path, ref, hyp, language="en", metric="wer_canonical")
    assert report.score > 0.0


def test_mixed_text_scores_both_scripts(tmp_path: Path) -> None:
    report = _evaluate(
        tmp_path,
        "我们讨论了 machine learning 的百分之五十",
        "我们讨论了 machine learning 的50%",
        language="cs",
        metric="mer_canonical",
    )
    assert report.score == 0.0
    assert report.pipeline_id == "asr.cs.mer.canonical_itn_cs_v1.token_mer_v1"


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #
def test_mer_canonical_pipeline_id_describe_and_run(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts import describe_pipeline, run_task

    pipeline_id = "asr.cs.mer.canonical_itn_cs_v1.token_mer_v1"
    description = describe_pipeline("asr", pipeline_id=pipeline_id)
    assert description.pipeline_id == "asr.cs.mer.canonical_itn_cs_v1.token_mer_v1"
    assert description.metric == "mer"
    assert description.execution_metrics == ("mer",)
    assert description.node_ids == ("normalization/canonical_itn", "scoring/token_mer")

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "开会讨论 it's been okay")])
    _write_key_text(hyp_file, [("utt1", "开会讨论 its been okay")])
    report = run_task(
        "asr",
        ref_file=str(ref_file),
        hyp_file=str(hyp_file),
        pipeline_id=pipeline_id,
        output_dir=str(tmp_path / "eval"),
    )
    assert report.score == 0.0
    norm_details = report.pipeline_trace[0].details
    assert norm_details["engine"]["en_span_normalizer"] == "whisper_english"
    assert (tmp_path / "eval" / "report.json").exists()


def test_mer_canonical_selector_is_not_public_script_compatibility() -> None:
    from sure_eval.evaluation.scripts import describe_pipeline

    with pytest.raises(ValueError, match="No configured route"):
        describe_pipeline("asr", language="cs", metric="mer_canonical")


def test_mer_canonical_rejects_foreign_scorer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="token_mer"):
        _evaluate(
            tmp_path,
            "今天天气很好",
            "今天天气很好",
            language="cs",
            metric="mer_canonical",
            scorer="wenet",
        )


def test_mer_canonical_zero_reference_tokens_raise(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1 no tab separator here\n", encoding="utf-8")
    hyp_file.write_text("utt1 no tab separator here\n", encoding="utf-8")
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    with pytest.raises(ValueError):
        evaluate_asr_files(str(ref_file), str(hyp_file), language="cs", metric="mer_canonical")
