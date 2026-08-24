from __future__ import annotations

from pathlib import Path

import pytest


def _require_funasr_node_env() -> None:
    node_dir = (
        Path(__file__).resolve().parent.parent
        / "src" / "sure_eval" / "evaluation" / "nodes" / "normalization" / "funasr_itn"
    )
    venv_python = node_dir / ".venv" / "bin" / "python"
    funasr_src = node_dir / "fun_text_processing"
    if not venv_python.exists() or not funasr_src.is_dir():
        pytest.skip(
            "funasr_itn node-local environment is not prepared. "
            "Run: sure-eval env setup --node normalization/funasr_itn"
        )


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
    # Status is "ok" if node-local env is prepared, "warning" for binary runtime
    # without explicit env var, "failed" if env is broken.
    assert result.status in {"ok", "failed", "warning"}


def test_funasr_is_default_normalizer_for_unsupported_languages() -> None:
    """Verify funasr is selected as default normalizer for languages without zh/en default."""
    from sure_eval.evaluation.tasks.asr.pipeline import _normalize_normalizer  # type: ignore[attr-defined]

    funasr_languages = ["ja", "es", "fr", "de", "ko", "ru", "pt", "vi", "id", "tl"]
    for lang in funasr_languages:
        metric = "cer" if lang in {"ja", "ko"} else "wer"
        assert _normalize_normalizer(language=lang, metric=metric, normalizer=None) == f"funasr:{lang}"

    assert _normalize_normalizer(language="zh", metric="cer", normalizer=None) == "wetext:zh_itn"
    assert _normalize_normalizer(language="en", metric="wer", normalizer=None) == "whisper"
    assert _normalize_normalizer(language="cs", metric="mer", normalizer=None) == "aispeech"


@pytest.mark.parametrize(
    ("language", "metric", "text", "expected_pipeline_id"),
    [
        ("ja", "cer", "\u3053\u3093\u306b\u3061\u306f", "asr.ja.cer.funasr_itn_ja_v1.wenet_cer_v1"),
        ("ko", "cer", "\uc548\ub155\ud558\uc138\uc694", "asr.ko.cer.funasr_itn_ko_v1.wenet_cer_v1"),
        ("es", "wer", "hola mundo", "asr.es.wer.funasr_itn_es_v1.wenet_wer_v1"),
        ("fr", "wer", "bonjour monde", "asr.fr.wer.funasr_itn_fr_v1.wenet_wer_v1"),
        ("de", "wer", "hallo welt", "asr.de.wer.funasr_itn_de_v1.wenet_wer_v1"),
        ("ru", "wer", "\u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440", "asr.ru.wer.funasr_itn_ru_v1.wenet_wer_v1"),
        ("pt", "wer", "ola mundo", "asr.pt.wer.funasr_itn_pt_v1.wenet_wer_v1"),
        ("vi", "wer", "xin chao", "asr.vi.wer.funasr_itn_vi_v1.wenet_wer_v1"),
        ("id", "wer", "halo dunia", "asr.id.wer.funasr_itn_id_v1.wenet_wer_v1"),
        ("tl", "wer", "kamusta mundo", "asr.tl.wer.funasr_itn_tl_v1.wenet_wer_v1"),
    ],
)
def test_funasr_default_pipeline_routes(
    language: str, metric: str, text: str, expected_pipeline_id: str, tmp_path: Path
) -> None:
    _require_funasr_node_env()
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", text)])
    _write_key_text(hyp_file, [("utt1", text)])

    report = evaluate_asr_files(str(ref_file), str(hyp_file), language=language, metric=metric)

    assert report.task == "ASR"
    assert report.language == language
    assert report.metric == metric
    assert report.pipeline_id == expected_pipeline_id
    assert report.pipeline_trace[0].node_id == "normalization/funasr_itn"
    assert report.pipeline_trace[0].details["profile"] == language


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
