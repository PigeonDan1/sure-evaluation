from __future__ import annotations

from pathlib import Path

import pytest


def test_etc_supported_profiles_and_text_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    from sure_eval.evaluation.nodes.normalization.etc_canonicalization import node

    assert set(node.SUPPORTED_PROFILES) == {
        "ar",
        "bn",
        "de",
        "en",
        "fa",
        "fr",
        "hi",
        "id",
        "ja",
        "ru",
        "vi",
        "zh",
    }
    monkeypatch.setattr(node, "canonicalize_batch", lambda texts, **_: [text.replace("two", "2") for text in texts])
    assert node.normalize_etc_text("two eggs", language="en") == "2 eggs"
    with pytest.raises(ValueError, match="Unsupported etc_canonicalization language"):
        node.normalize_etc_text("two", language="xx")


def test_etc_key_text_adapter_processes_ref_and_hyp_independently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.etc_canonicalization import node

    monkeypatch.setattr(node, "canonicalize_batch", lambda texts, **_: [text.replace("two", "2") for text in texts])
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("utt-1\tThey need two eggs.\nmalformed\n", encoding="utf-8")
    hyp.write_text("utt-1\tThey need 2 eggs.\n", encoding="utf-8")

    normalized, trace = node.normalize_etc_key_text_files(
        KeyTextFiles(ref_file=str(ref), hyp_file=str(hyp)), language="en"
    )

    try:
        assert Path(normalized.ref_file).read_text(encoding="utf-8") == (
            "utt-1\tThey need 2 eggs.\n"
        )
        assert Path(normalized.hyp_file).read_text(encoding="utf-8") == (
            "utt-1\tThey need 2 eggs.\n"
        )
        assert trace.node_id == "normalization/etc_canonicalization"
        assert trace.details["language"] == "en"
        assert trace.details["row_counts"] == {"ref": 1, "hyp": 1}
        assert trace.details["changed_counts"] == {"ref": 1, "hyp": 0}
    finally:
        Path(normalized.ref_file).unlink(missing_ok=True)
        Path(normalized.hyp_file).unlink(missing_ok=True)


def test_etc_manifest_declares_runtime_and_profiles() -> None:
    from sure_eval.evaluation.scripts.contracts import load_node_manifest

    manifest, _ = load_node_manifest("normalization/etc_canonicalization")
    assert manifest["runtime"] == "node_local_optional"
    assert manifest["package"]["name"] == "etc-canonicalizer"
    assert set(manifest["profiles"]) == {
        "ar",
        "bn",
        "de",
        "en",
        "fa",
        "fr",
        "hi",
        "id",
        "ja",
        "ru",
        "vi",
        "zh",
    }


def test_etc_routes_are_describable_for_every_profile() -> None:
    from sure_eval.evaluation.scripts.asr import describe_pipeline
    from sure_eval.evaluation.scripts.contracts import load_task_routes

    routes, _ = load_task_routes("asr")
    etc_routes = [
        route for route in routes["routes"]
        if "normalization/etc_canonicalization" in route["nodes"]
    ]
    assert {route["language"] for route in etc_routes} == {
        "ar", "bn", "de", "en", "fa", "fr", "hi", "id", "ja", "ru", "vi", "zh"
    }
    for route in etc_routes:
        description = describe_pipeline(pipeline_id=route["pipeline_id"])
        assert description.node_ids[0] == "normalization/etc_canonicalization"


def test_etc_env_check_requires_user_prepared_runtime() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    result = NodeEnvChecker().check_node("normalization/etc_canonicalization")
    assert result.runtime == "node_local_project"
    assert result.node_id == "normalization/etc_canonicalization"
    if not result.ok:
        assert "sure-eval env setup --node normalization/etc_canonicalization" in result.fix


def test_etc_normalizer_runs_in_asr_pipeline_with_zero_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation.nodes.normalization.etc_canonicalization import node
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    monkeypatch.setattr(node, "canonicalize_batch", lambda texts, **_: [text.replace("two", "2") for text in texts])
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("utt-1\tThey need two eggs.\n", encoding="utf-8")
    hyp.write_text("utt-1\tThey need 2 eggs.\n", encoding="utf-8")

    report = evaluate_asr_files(
        str(ref), str(hyp), language="en", metric="wer", normalizer="etc:en"
    )

    assert report.score == 0.0
    assert report.pipeline_trace[0].node_id == "normalization/etc_canonicalization"
    assert "etc_canonicalization_en" in report.pipeline_id


def test_etc_route_describe_and_run_with_fake_canonicalizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation.nodes.normalization.etc_canonicalization import node
    from sure_eval.evaluation.scripts import asr

    monkeypatch.setattr(
        node, "canonicalize_batch", lambda texts, **_: [text.replace("two", "2") for text in texts]
    )
    pipeline_id = "asr.en.wer.etc_canonicalization_en_v1.wenet_wer_v1"
    description = asr.describe_pipeline(pipeline_id=pipeline_id)
    assert description.node_ids == (
        "normalization/etc_canonicalization", "scoring/wenet_wer"
    )

    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    ref.write_text("utt-1\tThey need two eggs.\n", encoding="utf-8")
    hyp.write_text("utt-1\tThey need 2 eggs.\n", encoding="utf-8")
    result = asr.run(
        str(ref), str(hyp), pipeline_id=pipeline_id, output_dir=str(tmp_path / "report")
    )
    assert result.pipeline_id == pipeline_id
    assert result.score == 0.0
