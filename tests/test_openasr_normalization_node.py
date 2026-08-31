from __future__ import annotations

from pathlib import Path

import pytest


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def _read_rows(path: str) -> list[tuple[str, str]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        key, text = line.split("\t", 1)
        rows.append((key, text))
    return rows


def test_openasr_text_normalizer_matches_leaderboard_behaviors() -> None:
    from sure_eval.evaluation.nodes.normalization.openasr_norm import normalize_openasr_text

    # filler words ignored (leaderboard-extended ignore list)
    assert normalize_openasr_text("um I uh think") == "i think"
    # acronym collapsing (not present in the stock Whisper normalizer)
    assert normalize_openasr_text("the B B C reported") == "the bbc reported"
    # currency expansion and contraction handling
    assert normalize_openasr_text("they paid $20 million, didn't they?") == (
        "they paid $20000000 did not they"
    )
    # British-American spelling and compound words
    assert normalize_openasr_text("we have wi fi in colour") == "we have wifi in color"
    # bracketed and parenthesized content removed
    assert normalize_openasr_text("hello [noise] (laughs) world") == "hello world"


def test_openasr_norm_files_drop_filtered_refs_from_both_sides(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.openasr_norm import (
        normalize_openasr_asr_files,
    )

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(
        ref_file,
        [
            ("keep", "Hello, World!"),
            ("marker", "ignore time segment in scoring"),
            ("empty", "uh um"),
        ],
    )
    _write_key_text(
        hyp_file,
        [("keep", "hello world"), ("marker", "something"), ("empty", "uh"), ("extra", "hyp only")],
    )

    normalized, trace = normalize_openasr_asr_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file))
    )

    try:
        assert _read_rows(normalized.ref_file) == [("keep", "hello world")]
        assert _read_rows(normalized.hyp_file) == [("keep", "hello world"), ("extra", "hyp only")]
        assert trace.node_id == "normalization/openasr_norm"
        assert trace.version == "v1"
        assert trace.details["profile"] == "english"
        assert trace.details["dropped_ref_keys"] == ["empty", "marker"]
        assert trace.details["drop_policy"] == "empty_or_ignore_marker_ref_dropped_from_both_sides"
        assert trace.details["num_rows"] == {"ref": 1, "hyp": 2}
        assert trace.details["normalization"]["backend"] == "open_asr_leaderboard"
        assert trace.details["normalization"]["vendored"] is True
        assert trace.details["normalization"]["upstream_commit"]
    finally:
        Path(normalized.ref_file).unlink(missing_ok=True)
        Path(normalized.hyp_file).unlink(missing_ok=True)


def test_openasr_norm_rejects_unknown_profile(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.openasr_norm import (
        normalize_openasr_asr_files,
    )

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "hello")])
    _write_key_text(hyp_file, [("utt1", "hello")])

    with pytest.raises(ValueError, match="Unsupported openasr_norm profile"):
        normalize_openasr_asr_files(
            KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)), profile="basic"
        )
