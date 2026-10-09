from __future__ import annotations

from pathlib import Path

import pytest


def _require_xnorm_node_env() -> None:
    node_dir = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "sure_eval"
        / "evaluation"
        / "nodes"
        / "normalization"
        / "xnorm"
    )
    if not (node_dir / ".venv" / "bin" / "python").exists():
        pytest.skip(
            "xnorm node-local environment is not prepared. "
            "Run: sure-eval env setup --node normalization/xnorm"
        )


def test_xnorm_exposes_all_supported_profiles() -> None:
    from sure_eval.evaluation.nodes.normalization.xnorm import SUPPORTED_PROFILES

    assert set(SUPPORTED_PROFILES) == {
        "ar",
        "de",
        "en",
        "es",
        "fr",
        "he",
        "hi",
        "hu",
        "hy",
        "id",
        "it",
        "ja",
        "ko",
        "mr",
        "pt",
        "ru",
        "sv",
        "th",
        "vi",
        "zh",
    }


def test_xnorm_normalizes_representative_text() -> None:
    from sure_eval.evaluation.nodes.normalization.xnorm import normalize_xnorm_text

    assert normalize_xnorm_text("Saya punya 123 buku.", language="id") == (
        "Saya punya seratus dua puluh tiga buku"
    )
    assert normalize_xnorm_text("ฉันมี 123 หนังสือ", language="th") == (
        "ฉันมี หนึ่งร้อยยี่สิบสาม หนังสือ"
    )


def test_xnorm_rejects_unknown_language() -> None:
    from sure_eval.evaluation.nodes.normalization.xnorm import normalize_xnorm_text

    with pytest.raises(ValueError, match="Unsupported xnorm language"):
        normalize_xnorm_text("123", language="xx")


def test_xnorm_key_text_adapter_preserves_keys_and_trace(tmp_path: Path) -> None:
    _require_xnorm_node_env()
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.xnorm import normalize_xnorm_key_text_files

    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("utt-1\tSaya punya 123 buku.\nmalformed\n", encoding="utf-8")
    hyp.write_text("utt-1\tSaya punya 123 buku.\n", encoding="utf-8")

    normalized, trace = normalize_xnorm_key_text_files(
        KeyTextFiles(ref_file=str(ref), hyp_file=str(hyp)), language="id"
    )

    assert Path(normalized.ref_file).read_text(encoding="utf-8") == (
        "utt-1\tSaya punya seratus dua puluh tiga buku\n"
    )
    assert Path(normalized.hyp_file).read_text(encoding="utf-8") == (
        "utt-1\tSaya punya seratus dua puluh tiga buku\n"
    )
    assert trace.node_id == "normalization/xnorm"
    assert trace.details["language"] == "id"
    assert trace.details["row_counts"] == {"ref": 1, "hyp": 1}

    Path(normalized.ref_file).unlink()
    Path(normalized.hyp_file).unlink()


def test_xnorm_manifest_declares_profiles() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker
    from sure_eval.evaluation.scripts.contracts import load_node_manifest

    manifest, _ = load_node_manifest("normalization/xnorm")
    assert manifest["runtime"] == "node_local_optional"
    assert set(manifest["profiles"]) == {
        "ar",
        "de",
        "en",
        "es",
        "fr",
        "he",
        "hi",
        "hu",
        "hy",
        "id",
        "it",
        "ja",
        "ko",
        "mr",
        "pt",
        "ru",
        "sv",
        "th",
        "vi",
        "zh",
    }
    result = NodeEnvChecker().check_node("normalization/xnorm")
    assert result.runtime == "node_local_project"
    if not result.ok:
        pytest.skip(result.message)


def test_xnorm_explicit_asr_selector_and_pipeline(tmp_path: Path) -> None:
    _require_xnorm_node_env()
    from sure_eval.evaluation.tasks.asr.pipeline import (
        _normalize_normalizer,
        evaluate_asr_files,
    )

    assert _normalize_normalizer(language="id", metric="wer", normalizer="xnorm:id") == (
        "xnorm:id"
    )
    with pytest.raises(ValueError, match="does not match ASR language"):
        _normalize_normalizer(language="id", metric="wer", normalizer="xnorm:zh")

    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("utt-1\tSaya punya seratus dua puluh tiga buku\n", encoding="utf-8")
    hyp.write_text("utt-1\tSaya punya 123 buku.\n", encoding="utf-8")

    report = evaluate_asr_files(
        str(ref), str(hyp), language="id", metric="wer", normalizer="xnorm:id"
    )

    assert report.score == 0.0
    assert report.pipeline_trace[0].node_id == "normalization/xnorm"
    assert "xnorm_id" in report.pipeline_id


def test_xnorm_routes_are_describable_for_every_profile() -> None:
    from sure_eval.evaluation.nodes.normalization.xnorm import SUPPORTED_PROFILES
    from sure_eval.evaluation.scripts.asr import describe_pipeline
    from sure_eval.evaluation.scripts.contracts import load_task_routes

    routes, _ = load_task_routes("asr")
    route_ids = {
        str(route["pipeline_id"])
        for route in routes["routes"]
        if "xnorm_" in str(route.get("pipeline_id"))
    }
    expected = {
        f"asr.{language}.{'cer' if language in {'ar', 'ja', 'ko', 'th', 'zh'} else 'wer'}."
        f"xnorm_{language}_v1.wenet_{'cer' if language in {'ar', 'ja', 'ko', 'th', 'zh'} else 'wer'}_v1"
        for language in SUPPORTED_PROFILES
    }
    assert route_ids == expected
    for pipeline_id in sorted(expected):
        description = describe_pipeline(pipeline_id=pipeline_id)
        assert description.pipeline_id == pipeline_id
        assert description.node_ids[0] == "normalization/xnorm"

