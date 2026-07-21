from __future__ import annotations

from pathlib import Path

from sure_eval.evaluation.sure_evaluator import SUREEvaluator


def test_asr_char_metric_is_reported_as_cer(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1\tabc\n", encoding="utf-8")
    hyp_file.write_text("utt1\tadc\n", encoding="utf-8")

    result = SUREEvaluator(language="en").evaluate("ASR", str(ref_file), str(hyp_file), tochar=True)

    assert "cer" in result
    assert "wer" not in result
    assert result["score"] == result["cer"]
