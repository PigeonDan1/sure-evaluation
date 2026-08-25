from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml


def _require_funasr_node_env() -> None:
    node_dir = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "sure_eval"
        / "evaluation"
        / "nodes"
        / "normalization"
        / "funasr_itn"
    )
    venv_python = node_dir / ".venv" / "bin" / "python"
    funasr_src = node_dir / "runtime" / "fun_text_processing"
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


def test_funasr_zh_profile_runtime_details(monkeypatch: pytest.MonkeyPatch) -> None:
    from sure_eval.evaluation.nodes.normalization.funasr_itn import node

    monkeypatch.setattr(
        node,
        "_source_metadata",
        lambda: {
            "repository": "https://github.com/modelscope/FunASR.git",
            "revision": "a" * 40,
            "subdirectory": "fun_text_processing",
            "source_tree": "b" * 40,
            "license": "Apache-2.0",
        },
    )
    monkeypatch.setattr(node, "_distribution_version", lambda package: f"test-{package}")

    details = node.funasr_runtime_details("zh")
    assert details["node_id"] == "normalization/funasr_itn"
    assert details["profile"] == "zh"
    assert details["language"] == "zh"
    assert details["direction"] == "itn"
    assert "InverseNormalizer" in details["normalizer_class"]
    assert details["source"]["revision"] == "a" * 40
    assert details["runtime_versions"]["python"]
    assert details["runtime_versions"]["pynini"] == "test-pynini"
    assert details["runtime_versions"]["tqdm"] == "test-tqdm"


@pytest.mark.parametrize(
    ("profile", "text", "expected"),
    [
        ("zh", "\u4e8c\u767e\u4e94\u5341", "250"),
        ("ja", "\u4e8c\u5343\u4e8c\u5341\u56db\u5e74", "2024\u5e74"),
        ("ko", "\uc774\uc2ed\uc0bc\uc77c", "23\uc77c "),
        ("es", "veintid\u00f3s euros", "\u20ac22"),
        ("fr", "mille cent vingt-deux", "1122"),
        ("de", "zweiundzwanzig", "22"),
        ("ru", "\u0434\u0432\u0430\u0434\u0446\u0430\u0442\u044c \u0434\u0432\u0430", "22"),
        ("pt", "vinte dois", "22"),
        ("vi", "tr\u1eeb hai m\u01b0\u01a1i ba", "-23"),
        ("id", "dua ribu dua puluh dua", "2022"),
        ("tl", "dalawampu", "20"),
    ],
)
def test_funasr_itn_semantics(profile: str, text: str, expected: str) -> None:
    from sure_eval.evaluation.nodes.normalization.funasr_itn import normalize_funasr_text

    _require_funasr_node_env()
    assert normalize_funasr_text(text, profile=profile) == expected


def test_funasr_english_profile_smoke() -> None:
    from sure_eval.evaluation.nodes.normalization.funasr_itn import normalize_funasr_text

    _require_funasr_node_env()
    assert normalize_funasr_text("twenty dollars", profile="en") == "$20"


def test_funasr_node_env_checker_recognizes_node() -> None:
    """Verify NodeEnvChecker can locate and evaluate the funasr_itn node."""
    from sure_eval.evaluation.env_check import NodeEnvChecker

    result = NodeEnvChecker().check_node("normalization/funasr_itn")
    assert result.node_id == "normalization/funasr_itn"
    assert result.runtime == "node_local_project"
    assert result.status in {"ok", "failed"}


def test_funasr_is_default_normalizer_for_unsupported_languages() -> None:
    """Verify funasr is selected as default normalizer for languages without zh/en default."""
    from sure_eval.evaluation.tasks.asr.pipeline import _normalize_normalizer  # type: ignore[attr-defined]

    funasr_languages = ["ja", "es", "fr", "de", "ko", "ru", "pt", "vi", "id", "tl"]
    for lang in funasr_languages:
        metric = "cer" if lang in {"ja", "ko"} else "wer"
        assert (
            _normalize_normalizer(language=lang, metric=metric, normalizer=None) == f"funasr:{lang}"
        )

    assert _normalize_normalizer(language="zh", metric="cer", normalizer=None) == "wetext:zh_itn"
    assert _normalize_normalizer(language="en", metric="wer", normalizer=None) == "whisper"
    assert _normalize_normalizer(language="cs", metric="mer", normalizer=None) == "aispeech"


def test_funasr_profile_must_match_route_language() -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import _normalize_normalizer  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="does not match ASR language"):
        _normalize_normalizer(language="es", metric="wer", normalizer="funasr:ja")


class _FakeNormalizer:
    def inverse_normalize(self, text: str, *, verbose: bool) -> str:
        assert verbose is False
        print(text)
        return text.upper()


def test_funasr_key_text_parser_preserves_empty_text_and_reports_stats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from sure_eval.evaluation.nodes.normalization.funasr_itn.node import (
        _normalize_key_text_file,
    )

    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text("utt1\thello\n\nutt2\t\n", encoding="utf-8")

    stats = _normalize_key_text_file(str(input_file), str(output_file), _FakeNormalizer())

    assert output_file.read_text(encoding="utf-8") == "utt1\tHELLO\nutt2\t\n"
    assert stats == {
        "num_rows": 2,
        "num_blank_lines": 1,
        "num_empty_text_rows": 1,
        "num_empty_after_normalization": 1,
        "num_dropped_malformed_rows": 0,
    }
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("utt1 missing-tab\n", "line 1"),
        ("\tmissing-key\n", "Empty key"),
        ("utt1\tone\nutt1\ttwo\n", "Duplicate key 'utt1'"),
        ("\n", "No <key><TAB><text> rows"),
    ],
)
def test_funasr_key_text_parser_rejects_invalid_input(
    content: str, message: str, tmp_path: Path
) -> None:
    from sure_eval.evaluation.nodes.normalization.funasr_itn.node import (
        _normalize_key_text_file,
    )

    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _normalize_key_text_file(str(input_file), str(output_file), _FakeNormalizer())


def test_funasr_prepare_installs_verified_source_atomically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation.nodes.normalization.funasr_itn import prepare_funasr_itn

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    lock = {
        "repository": "https://example.test/FunASR.git",
        "revision": "a" * 40,
        "subdirectory": "fun_text_processing",
        "source_tree": "b" * 40,
        "license_file": "LICENSE",
    }
    lock_file = node_dir / "source_lock.json"
    lock_file.write_text(json.dumps(lock), encoding="utf-8")
    runtime_dir = node_dir / "runtime"
    monkeypatch.setattr(prepare_funasr_itn, "NODE_DIR", node_dir)
    monkeypatch.setattr(prepare_funasr_itn, "LOCK_FILE", lock_file)
    monkeypatch.setattr(prepare_funasr_itn, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(
        prepare_funasr_itn,
        "MARKER_FILE",
        runtime_dir / "funasr_revision.json",
    )

    def fake_checkout(checkout_dir: Path, source_lock: dict[str, str]) -> str:
        package_dir = checkout_dir / source_lock["subdirectory"]
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (checkout_dir / source_lock["license_file"]).write_text("Apache", encoding="utf-8")
        return "b" * 40

    monkeypatch.setattr(prepare_funasr_itn, "_checkout_source", fake_checkout)

    first = prepare_funasr_itn.prepare()
    second = prepare_funasr_itn.prepare()

    assert first["status"] == "prepared"
    assert second["status"] == "already_prepared"
    assert first["revision"] == lock["revision"]
    assert first["source_tree"] == "b" * 40
    assert (runtime_dir / "fun_text_processing" / "__init__.py").is_file()
    assert (runtime_dir / "LICENSE").read_text(encoding="utf-8") == "Apache"


def test_funasr_source_lock_matches_manifest() -> None:
    node_dir = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "sure_eval"
        / "evaluation"
        / "nodes"
        / "normalization"
        / "funasr_itn"
    )
    source_lock = json.loads((node_dir / "source_lock.json").read_text(encoding="utf-8"))
    manifest = yaml.safe_load((node_dir / "manifest.yaml").read_text(encoding="utf-8"))

    assert manifest["package"]["repo"] == source_lock["repository"]
    assert manifest["package"]["revision"] == source_lock["revision"]
    assert manifest["package"]["subdirectory"] == source_lock["subdirectory"]
    assert manifest["package"]["source_tree"] == source_lock["source_tree"]


@pytest.mark.parametrize(
    ("language", "metric", "spoken_text", "written_text", "expected_pipeline_id"),
    [
        (
            "ja",
            "cer",
            "\u4e8c\u5343\u4e8c\u5341\u56db\u5e74",
            "2024\u5e74",
            "asr.ja.cer.funasr_itn_ja_v1.wenet_cer_v1",
        ),
        (
            "ko",
            "cer",
            "\uc774\uc2ed\uc0bc\uc77c",
            "23\uc77c",
            "asr.ko.cer.funasr_itn_ko_v1.wenet_cer_v1",
        ),
        (
            "es",
            "wer",
            "veintid\u00f3s euros",
            "\u20ac22",
            "asr.es.wer.funasr_itn_es_v1.wenet_wer_v1",
        ),
        (
            "fr",
            "wer",
            "mille cent vingt-deux",
            "1122",
            "asr.fr.wer.funasr_itn_fr_v1.wenet_wer_v1",
        ),
        (
            "de",
            "wer",
            "zweiundzwanzig",
            "22",
            "asr.de.wer.funasr_itn_de_v1.wenet_wer_v1",
        ),
        (
            "ru",
            "wer",
            "\u0434\u0432\u0430\u0434\u0446\u0430\u0442\u044c \u0434\u0432\u0430",
            "22",
            "asr.ru.wer.funasr_itn_ru_v1.wenet_wer_v1",
        ),
        ("pt", "wer", "vinte dois", "22", "asr.pt.wer.funasr_itn_pt_v1.wenet_wer_v1"),
        (
            "vi",
            "wer",
            "tr\u1eeb hai m\u01b0\u01a1i ba",
            "-23",
            "asr.vi.wer.funasr_itn_vi_v1.wenet_wer_v1",
        ),
        (
            "id",
            "wer",
            "dua ribu dua puluh dua",
            "2022",
            "asr.id.wer.funasr_itn_id_v1.wenet_wer_v1",
        ),
        ("tl", "wer", "dalawampu", "20", "asr.tl.wer.funasr_itn_tl_v1.wenet_wer_v1"),
    ],
)
def test_funasr_default_pipeline_routes(
    language: str,
    metric: str,
    spoken_text: str,
    written_text: str,
    expected_pipeline_id: str,
    tmp_path: Path,
) -> None:
    _require_funasr_node_env()
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", spoken_text)])
    _write_key_text(hyp_file, [("utt1", written_text)])

    report = evaluate_asr_files(str(ref_file), str(hyp_file), language=language, metric=metric)

    assert report.task == "ASR"
    assert report.language == language
    assert report.metric == metric
    assert report.pipeline_id == expected_pipeline_id
    assert report.score == 0.0
    assert report.pipeline_trace[0].node_id == "normalization/funasr_itn"
    assert report.pipeline_trace[0].details["profile"] == language


def test_funasr_key_text_files_preserve_keys_and_trace_runtime(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.funasr_itn import normalize_funasr_files

    _require_funasr_node_env()
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(
        ref_file, [("utt1", "\u4e8c\u767e\u4e94\u5341"), ("utt2", "\u4e09\u5341\u516b")]
    )
    _write_key_text(
        hyp_file, [("utt1", "\u4e8c\u767e\u4e94\u5341"), ("utt2", "\u4e09\u5341\u516b")]
    )

    normalized, trace = normalize_funasr_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)),
        profile="zh",
    )

    try:
        assert trace.node_id == "normalization/funasr_itn"
        assert trace.details["profile"] == "zh"
        assert trace.details["language"] == "zh"
        assert trace.details["num_rows"] == {"ref": 2, "hyp": 2}
        assert trace.details["row_stats"]["ref"]["num_dropped_malformed_rows"] == 0
        assert "ref_rows" not in trace.details
        assert "hyp_rows" not in trace.details
    finally:
        Path(normalized.ref_file).unlink(missing_ok=True)
        Path(normalized.hyp_file).unlink(missing_ok=True)
