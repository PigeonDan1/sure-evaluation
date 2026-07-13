from __future__ import annotations

from pathlib import Path


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def test_punctuation_strip_text_removes_punctuation_only() -> None:
    from sure_eval.evaluation.nodes.normalization.punctuation_strip_norm import (
        normalize_punctuation_strip_text,
    )

    assert normalize_punctuation_strip_text("我有2个苹果。") == "我有2个苹果"
    assert normalize_punctuation_strip_text("Hello, WORLD!") == "Hello WORLD"
    assert normalize_punctuation_strip_text("价格是$13.5") == "价格是135"


def test_punctuation_strip_key_text_files_preserve_keys_and_trace(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.punctuation_strip_norm import (
        normalize_punctuation_strip_key_text_files,
    )

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好，世界！"), ("utt2", "。？！")])
    _write_key_text(hyp_file, [("utt1", "你好世界"), ("utt2", "...")])

    normalized, trace = normalize_punctuation_strip_key_text_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)),
        language="zh",
    )

    try:
        assert Path(normalized.ref_file).read_text(encoding="utf-8") == "utt1\t你好世界\nutt2\t\n"
        assert Path(normalized.hyp_file).read_text(encoding="utf-8") == "utt1\t你好世界\nutt2\t\n"
        assert trace.node_id == "normalization/punctuation_strip_norm"
        assert trace.details["profile"] == "unicode_category_p_or_ascii"
        assert trace.details["language"] == "zh"
        assert trace.details["normalization"]["operation"] == "punctuation_stripping_only"
        assert trace.details["normalization"]["preserves_numbers"] is True
        assert trace.details["normalization"]["preserves_case"] is True
        assert trace.details["num_rows"] == {"ref": 2, "hyp": 2}
        assert trace.details["num_empty_after_normalization"] == {"ref": 1, "hyp": 1}
    finally:
        Path(normalized.ref_file).unlink(missing_ok=True)
        Path(normalized.hyp_file).unlink(missing_ok=True)
