from __future__ import annotations

from pathlib import Path


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def test_whisper_english_normalizer_matches_expected_asr_rules() -> None:
    from sure_eval.evaluation.nodes.normalization.whisper_norm import normalize_whisper_text

    assert normalize_whisper_text("I've got TWO apples.", profile="english") == "i have got 2 apples"
    assert normalize_whisper_text("colour centre", profile="english") == "color center"
    assert normalize_whisper_text("um, hello [noise]", profile="english") == "hello"


def test_whisper_basic_normalizer_is_available_without_english_number_rules() -> None:
    from sure_eval.evaluation.nodes.normalization.whisper_norm import normalize_whisper_text

    assert normalize_whisper_text("Hello, TWO apples.", profile="basic") == "hello two apples"


def test_whisper_normalization_preserves_keys_and_reports_empty_rows(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.whisper_norm import normalize_whisper_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "I have two apples."), ("utt2", "[noise]")])
    _write_key_text(hyp_file, [("utt1", "i have 2 apples"), ("utt2", "um")])

    normalized, trace = normalize_whisper_asr_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)),
        language="en",
        profile="english",
    )

    try:
        assert Path(normalized.ref_file).read_text(encoding="utf-8") == "utt1\ti have 2 apples\nutt2\t\n"
        assert Path(normalized.hyp_file).read_text(encoding="utf-8") == "utt1\ti have 2 apples\nutt2\t\n"
        assert trace.node_id == "normalization/whisper_norm"
        assert trace.details["profile"] == "english"
        assert trace.details["num_rows"] == {"ref": 2, "hyp": 2}
        assert trace.details["num_empty_after_normalization"] == {"ref": 1, "hyp": 1}
        assert trace.details["normalization"]["upstream_package"] == "openai-whisper"
        assert trace.details["normalization"]["upstream_version"] == "20250625"
    finally:
        Path(normalized.ref_file).unlink(missing_ok=True)
        Path(normalized.hyp_file).unlink(missing_ok=True)
