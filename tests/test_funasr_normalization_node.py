from __future__ import annotations

from pathlib import Path

import pytest


def _require_funasr_node_env() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    result = NodeEnvChecker().check_node("normalization/funasr_itn")
    if result.status != "ok":
        pytest.skip(f"funasr_itn node-local environment is not prepared: {result.message}")


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def test_funasr_unsupported_profile_fails() -> None:
    from sure_eval.evaluation.nodes.normalization.funasr_itn import normalize_funasr_text

    with pytest.raises(ValueError, match="Unsupported funasr_itn profile"):
        normalize_funasr_text("hello", profile="unsupported_lang")


def test_funasr_zh_profile_runtime_details() -> None:
    from sure_eval.evaluation.nodes.normalization.funasr_itn.node import funasr_runtime_details

    details = funasr_runtime_details("zh")
    assert details["node_id"] == "normalization/funasr_itn"
    assert details["profile"] == "zh"
    assert details["language"] == "zh"
    assert details["direction"] == "itn"
    assert "InverseNormalizer" in details["normalizer_class"]


@pytest.mark.parametrize(
    ("profile", "text", "expected"),
    [
        ("zh", "\u4e8c\u767e\u4e94\u5341", "250"),
    ],
)
def test_funasr_chinese_itn_smoke(profile: str, text: str, expected: str, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.funasr_itn import normalize_funasr_text

    _require_funasr_node_env()
    assert normalize_funasr_text(text, profile=profile) == expected


@pytest.mark.parametrize(
    ("profile", "text"),
    [
        ("en", "twenty dollars"),
        ("ja", "\u4e8c\u5343\u4e8c\u5341\u56db\u5e74"),
    ],
)
def test_funasr_other_profiles_smoke(profile: str, text: str, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.funasr_itn import normalize_funasr_text

    _require_funasr_node_env()
    normalized = normalize_funasr_text(text, profile=profile)
    assert isinstance(normalized, str)
    assert normalized


def test_funasr_node_env_checker_recognizes_node() -> None:
    """Verify NodeEnvChecker can locate and evaluate the funasr_itn node."""
    from sure_eval.evaluation.env_check import NodeEnvChecker

    result = NodeEnvChecker().check_node("normalization/funasr_itn")
    assert result.node_id == "normalization/funasr_itn"
    # Status is "ok" if node-local env is prepared, otherwise "failed"
    assert result.status in {"ok", "failed"}


def test_funasr_key_text_files_preserve_keys_and_trace_runtime(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.funasr_itn import normalize_funasr_files

    _require_funasr_node_env()
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "\u4e8c\u767e\u4e94\u5341"), ("utt2", "\u4e09\u5341\u516b")])
    _write_key_text(hyp_file, [("utt1", "\u4e8c\u767e\u4e94\u5341"), ("utt2", "\u4e09\u5341\u516b")])

    normalized, trace = normalize_funasr_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)),
        profile="zh",
    )

    try:
        assert trace.node_id == "normalization/funasr_itn"
        assert trace.details["profile"] == "zh"
        assert trace.details["language"] == "zh"
        assert trace.details["num_rows"] == {"ref": 2, "hyp": 2}
    finally:
        Path(normalized.ref_file).unlink(missing_ok=True)
        Path(normalized.hyp_file).unlink(missing_ok=True)
