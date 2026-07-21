from __future__ import annotations

from pathlib import Path

import pytest


def _require_wetext_node_env() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    result = NodeEnvChecker().check_node("normalization/wetext_norm")
    if result.status != "ok":
        pytest.skip(f"wetext_norm node-local environment is not prepared: {result.message}")


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


@pytest.mark.parametrize(
    ("profile", "text", "expected"),
    [
        ("zh_tn", "共465篇，约315万字", "共四百六十五篇,约三百一十五万字"),
        ("zh_tn", "2002/01/28", "二零零二年一月二十八日"),
        ("zh_tn", "价格是$13.5", "价格是十三点五美元"),
        ("zh_itn", "共四百六十五篇约三百一十五万字", "共465篇约315万字"),
        ("zh_itn", "同比增长百分之六点三", "同比增长6.3%"),
        ("zh_itn", "价格是十三点五美元", "价格是$13.5"),
    ],
)
def test_wetext_chinese_demo_cases(profile: str, text: str, expected: str, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.wetext_norm import normalize_wetext_text

    _require_wetext_node_env()
    assert normalize_wetext_text(text, profile=profile, cache_dir=str(tmp_path)) == expected


@pytest.mark.parametrize(
    ("profile", "text", "expected"),
    [
        ("en_tn", "I have 2 apples.", "I have two apples."),
        ("en_tn", "$20", "twenty dollars"),
        ("en_itn", "twenty dollars", "$20"),
        ("en_itn", "two point five", "2.5"),
    ],
)
def test_wetext_english_demo_cases(profile: str, text: str, expected: str, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.wetext_norm import normalize_wetext_text

    _require_wetext_node_env()
    assert normalize_wetext_text(text, profile=profile, cache_dir=str(tmp_path)) == expected


@pytest.mark.parametrize(
    ("profile", "text"),
    [
        ("ja_tn", "2024年7月6日"),
        ("ja_itn", "二千二十四年七月六日"),
    ],
)
def test_wetext_japanese_profiles_smoke(profile: str, text: str, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.wetext_norm import normalize_wetext_text

    _require_wetext_node_env()
    normalized = normalize_wetext_text(text, profile=profile, cache_dir=str(tmp_path))
    assert isinstance(normalized, str)
    assert normalized
    assert normalized != text


def test_wetext_key_text_files_preserve_keys_and_trace_runtime(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.wetext_norm import normalize_wetext_key_text_files

    _require_wetext_node_env()
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "2002/01/28"), ("utt2", "价格是$13.5")])
    _write_key_text(hyp_file, [("utt1", "二零零二年一月二十八日"), ("utt2", "价格是十三点五美元")])

    normalized, trace = normalize_wetext_key_text_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)),
        profile="zh_tn",
        cache_dir=str(tmp_path / "wetext_cache"),
    )

    try:
        assert Path(normalized.ref_file).read_text(encoding="utf-8") == (
            "utt1\t二零零二年一月二十八日\n"
            "utt2\t价格是十三点五美元\n"
        )
        assert Path(normalized.hyp_file).read_text(encoding="utf-8") == (
            "utt1\t二零零二年一月二十八日\n"
            "utt2\t价格是十三点五美元\n"
        )
        assert trace.node_id == "normalization/wetext_norm"
        assert trace.details["profile"] == "zh_tn"
        assert trace.details["language"] == "zh"
        assert trace.details["direction"] == "tn"
        assert trace.details["package"] == "WeTextProcessing"
        assert trace.details["package_version"] == "1.2.0"
        assert trace.details["pinned_package_version"] == "1.2.0"
        assert trace.details["normalizer_class"] == "tn.chinese.normalizer.Normalizer"
        assert trace.details["num_rows"] == {"ref": 2, "hyp": 2}
        assert trace.details["num_empty_after_normalization"] == {"ref": 0, "hyp": 0}
    finally:
        Path(normalized.ref_file).unlink(missing_ok=True)
        Path(normalized.hyp_file).unlink(missing_ok=True)


def test_wetext_unsupported_profile_fails_before_importing_backend() -> None:
    from sure_eval.evaluation.nodes.normalization.wetext_norm import normalize_wetext_text

    with pytest.raises(ValueError, match="Unsupported wetext_norm profile"):
        normalize_wetext_text("hello", profile="fr_tn")
