from __future__ import annotations

from pathlib import Path

import pytest


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def test_asr_zh_cer_scores_empty_hypothesis_text_as_deletions(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "今天天气很好"), ("utt2", "我们一起去公园")])
    _write_key_text(hyp_file, [("utt1", "今天天气很好"), ("utt2", "")])

    report = evaluate_asr_files(str(ref_file), str(hyp_file), language="zh", metric="cer")
    result = report.details["scoring_result"]

    assert result["all"] == 13
    assert result["del"] == 7
    assert report.score == pytest.approx(7 / 13)


def test_asr_zh_cer_scores_missing_hypothesis_utterance_as_deletions(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "今天天气很好"), ("utt2", "我们一起去公园")])
    _write_key_text(hyp_file, [("utt1", "今天天气很好")])

    report = evaluate_asr_files(str(ref_file), str(hyp_file), language="zh", metric="cer")
    result = report.details["scoring_result"]

    assert result["num_ref_utts"] == 2
    assert result["num_hyp_utts"] == 1
    assert result["num_hyp_missing_utts"] == 1
    assert result["hyp_missing_keys_sample"] == ["utt2"]
    assert result["hyp_missing_policy"] == "scored_as_empty_hypothesis"
    assert result["all"] == 13
    assert result["del"] == 7
    assert report.score == pytest.approx(7 / 13)


def test_asr_scoring_result_reports_utterance_coverage(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "今天天气很好")])
    _write_key_text(hyp_file, [("utt1", "今天天气很好"), ("utt9", "多余的行")])

    report = evaluate_asr_files(str(ref_file), str(hyp_file), language="zh", metric="cer")
    result = report.details["scoring_result"]

    assert result["num_ref_utts"] == 1
    assert result["num_hyp_utts"] == 2
    assert result["num_hyp_missing_utts"] == 0
    assert result["num_hyp_extra_utts"] == 1
    assert result["hyp_extra_keys_sample"] == ["utt9"]
    assert report.score == 0.0


def test_asr_zh_cer_rejects_inputs_without_tab_separator(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1 今天天气很好\nutt2 我们一起去公园\n", encoding="utf-8")
    hyp_file.write_text("utt1 今天天气很好\nutt2 我们一起去公园\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tab-separated"):
        evaluate_asr_files(str(ref_file), str(hyp_file), language="zh", metric="cer")


def test_asr_en_wer_rejects_inputs_that_score_zero_reference_tokens(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1 hello world\n", encoding="utf-8")
    hyp_file.write_text("utt1 hello world\n", encoding="utf-8")

    with pytest.raises(ValueError, match="zero reference tokens"):
        evaluate_asr_files(str(ref_file), str(hyp_file), language="en", metric="wer")


def test_asr_normalization_keeps_empty_hypothesis_rows(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.aispeech_norm import normalize_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "今天天气很好"), ("utt2", "我们一起去公园")])
    _write_key_text(hyp_file, [("utt1", "今天天气很好"), ("utt2", "")])

    normalized, result = normalize_asr_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)), language="zh"
    )
    try:
        hyp_lines = Path(normalized.hyp_file).read_text(encoding="utf-8").splitlines()
        assert hyp_lines == ["utt1\t今天天气很好", "utt2\t"]
        assert result.details["row_stats"]["hyp"] == {
            "num_rows": 2,
            "num_empty_text_rows": 1,
            "num_dropped_malformed_rows": 0,
        }
    finally:
        Path(normalized.ref_file).unlink(missing_ok=True)
        Path(normalized.hyp_file).unlink(missing_ok=True)


def test_asr_codeswitch_mer_scores_empty_hypothesis_text_as_deletions(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好 world")])
    _write_key_text(hyp_file, [("utt1", "")])

    report = evaluate_asr_files(str(ref_file), str(hyp_file), language="cs", metric="mer")
    result = report.details["scoring_result"]

    assert report.score == pytest.approx(1.0)
    assert result["mer_details"]["all"] == 3
    assert result["mer_details"]["del"] == 3


def test_wenet_compute_wer_totals_do_not_depend_on_utterance_order(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.wenet_wer.wenet_compute_cer import compute_wer

    long_row = ("ulong", " ".join(["tok"] * 300))
    short_ref = [(f"u{i:03d}", "aa bb cc dd") for i in range(50)]
    short_hyp = [(f"u{i:03d}", "aa bb cc xx") for i in range(50)]  # one sub each

    totals = {}
    for order in ("long_first", "long_last"):
        ref_rows = [long_row] + short_ref if order == "long_first" else short_ref + [long_row]
        hyp_rows = [long_row] + short_hyp if order == "long_first" else short_hyp + [long_row]
        ref_file = tmp_path / f"ref_{order}.txt"
        hyp_file = tmp_path / f"hyp_{order}.txt"
        _write_key_text(ref_file, ref_rows)
        _write_key_text(hyp_file, hyp_rows)
        result = compute_wer(str(ref_file), str(hyp_file))
        totals[order] = {key: result[key] for key in ("all", "cor", "sub", "del", "ins")}

    assert totals["long_first"] == totals["long_last"]
    assert totals["long_first"]["all"] == 300 + 50 * 4
    assert totals["long_first"]["sub"] == 50


def test_asr_codeswitch_mer_allows_monolingual_side_scores(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好")])
    _write_key_text(hyp_file, [("utt1", "你好")])

    report = evaluate_asr_files(str(ref_file), str(hyp_file), language="cs", metric="mer")
    result = report.details["scoring_result"]

    assert report.score == 0.0
    assert result["wer"] == 0.0
    assert result["wer_details"]["all"] == 0
