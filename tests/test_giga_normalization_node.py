from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_GIGA_PACKAGES = ("zhconv", "whisper_normalizer", "num2words")

requires_giga_extra = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in _GIGA_PACKAGES),
    reason='normalization/giga_norm text rules require the [giga] extra (pip install -e ".[giga]")',
)


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def _giga_extra_installed() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in _GIGA_PACKAGES)


# --------------------------------------------------------------------------- #
# Base install: must work without the [giga] extra
# --------------------------------------------------------------------------- #


def test_giga_node_imports_without_the_optional_extra() -> None:
    """The ASR task imports giga_norm eagerly, so the node must import cleanly.

    Profile modules pull their dependencies lazily, so a base install must
    never break because the [giga] extra is absent.
    """

    from sure_eval.evaluation.nodes.normalization.giga_norm import (
        SUPPORTED_PROFILES,
        profile_for_language,
    )
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files  # noqa: F401

    assert len(SUPPORTED_PROFILES) == 17
    assert profile_for_language("ja") == "JPN"


def test_giga_profile_aliases_resolve_to_backing_modules() -> None:
    from sure_eval.evaluation.nodes.normalization.giga_norm.normalization_impl import (
        resolve_profile,
    )

    assert resolve_profile("YUE") == "CHN"
    assert resolve_profile("MED-CH") == "CHN"
    assert resolve_profile("SGP-EN") == "USA"
    assert resolve_profile("JPN_HARD") == "JPN"
    assert resolve_profile("syr") == "SYR"


def test_giga_unknown_profile_raises_instead_of_falling_back() -> None:
    from sure_eval.evaluation.nodes.normalization.giga_norm.normalization_impl import (
        resolve_profile,
    )

    with pytest.raises(ValueError, match="Unsupported giga_norm profile"):
        resolve_profile("XYZ")


def test_giga_language_to_profile_mapping() -> None:
    from sure_eval.evaluation.nodes.normalization.giga_norm import profile_for_language

    assert profile_for_language("zh") == "CHN"
    assert profile_for_language("en") == "USA"
    assert profile_for_language("ar-eg") == "EGY"
    assert profile_for_language("AR_SY") == "SYR"
    with pytest.raises(ValueError, match="not mapped for ASR language"):
        profile_for_language("cs")


def test_giga_profile_must_match_route_language(tmp_path: Path) -> None:
    """Illegal route combinations must fail before any normalization runs."""

    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "hello world")])
    _write_key_text(hyp_file, [("utt1", "hello world")])

    with pytest.raises(ValueError, match="does not match ASR language"):
        evaluate_asr_files(
            str(ref_file),
            str(hyp_file),
            language="en",
            metric="wer",
            normalizer="giga:CHN",
        )


# --------------------------------------------------------------------------- #
# Route identity and catalog
# --------------------------------------------------------------------------- #


def test_giga_routes_are_selectable_without_changing_defaults() -> None:
    from sure_eval.evaluation.scripts import describe_pipeline

    assert (
        describe_pipeline("asr", language="zh", metric="cer").pipeline_id
        == "asr.zh.cer.wetext_norm_zh_itn_v1.wenet_cer_v1"
    )
    assert (
        describe_pipeline("asr", language="en", metric="wer").pipeline_id
        == "asr.en.wer.whisper_norm_english_v1.wenet_wer_v1"
    )

    giga_zh = describe_pipeline("asr", pipeline_id="asr.zh.cer.giga_norm_chn_v1.wenet_cer_v1")
    assert giga_zh.node_ids == ("normalization/giga_norm", "scoring/wenet_cer")

    giga_en = describe_pipeline("asr", pipeline_id="asr.en.wer.giga_norm_usa_v1.wenet_wer_v1")
    assert giga_en.node_ids == ("normalization/giga_norm", "scoring/wenet_wer")

    assert (
        describe_pipeline("asr", language="ja", metric="cer").pipeline_id
        == "asr.ja.cer.funasr_itn_ja_v1.wenet_cer_v1"
    )
    assert (
        describe_pipeline("asr", language="ar-eg", metric="wer").pipeline_id
        == "asr.ar_eg.wer.giga_norm_egy_v1.wenet_wer_v1"
    )


def test_giga_node_is_registered_in_the_pipeline_catalog() -> None:
    """The regenerated catalog must contain every new giga route."""

    catalog_path = Path(__file__).resolve().parents[1] / "docs" / "pipeline_catalog.jsonl"
    rows = [
        json.loads(line)
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    giga_rows = {
        row["pipeline_id"]: row
        for row in rows
        if "normalization/giga_norm" in row.get("nodes", [])
    }

    assert "asr.zh.cer.giga_norm_chn_v1.wenet_cer_v1" in giga_rows
    assert "asr.en.wer.giga_norm_usa_v1.wenet_wer_v1" in giga_rows
    assert len(giga_rows) == 17

    for row in giga_rows.values():
        assert row["task"] == "ASR"
        assert row["metric"] in {"cer", "wer"}
        assert row["pipeline_kind"] == "atomic"
        assert row["route_config_path"] == "src/sure_eval/evaluation/tasks/asr/routes.yaml"


def test_giga_routes_resolve_through_agent_plan() -> None:
    """docs/agent_contract.md requires new routes to resolve via `agent plan`."""

    from sure_eval.evaluation import agent_plan

    payload = agent_plan.build_agent_plan(
        "asr",
        language="ja",
        pipeline_id="asr.ja.cer.giga_norm_jpn_v1.wenet_cer_v1",
    )

    assert payload["schema"] == "sure.eval.agent_plan.v1"
    route = payload["selected_routes"][0]
    assert route["pipeline_id"] == "asr.ja.cer.giga_norm_jpn_v1.wenet_cer_v1"
    assert route["resolved_metric"] == "cer"
    assert route["pipeline_kind"] == "atomic"
    assert route["member_pipeline_ids"] == []
    assert route["computation_node_ids"] == [
        "normalization/giga_norm",
        "scoring/wenet_cer",
    ]
    assert route["route_config_path"] == "src/sure_eval/evaluation/tasks/asr/routes.yaml"
    assert route["required_roles"] == ["hyp", "ref"]

    giga_node = next(node for node in route["nodes"] if node["node_id"] == "normalization/giga_norm")
    assert giga_node["runtime"] == "pip_optional"

    giga_check = next(
        check for check in route["env_checks"] if check["name"] == "normalization/giga_norm"
    )
    assert giga_check["required_for_selected_route"] is True
    if _giga_extra_installed():
        assert giga_check["status"] == "ok"
        assert payload["can_run_now"] is True
    else:
        # A base install must block with an actionable fix instead of scoring.
        assert giga_check["status"] == "failed"
        assert giga_check["blocking"] is True
        assert "zhconv" in giga_check["fix"]
        assert payload["can_run_now"] is False
        assert payload["blocking_issues"]


# --------------------------------------------------------------------------- #
# Vendoring and license
# --------------------------------------------------------------------------- #


def test_vendored_license_is_present_and_declared() -> None:
    """MIT requires the upstream notice to travel with the vendored sources."""

    import yaml

    node_dir = (
        Path(__file__).resolve().parents[1]
        / "src/sure_eval/evaluation/nodes/normalization/giga_norm"
    )
    license_file = node_dir / "normalization_impl" / "LICENSE.gigaspeechbench"
    assert license_file.exists()

    license_text = license_file.read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "THIRD-PARTY NOTICES" in license_text
    assert "speechio/chinese_text_normalization" in license_text
    assert "tsroten/zhon" in license_text

    manifest = yaml.safe_load((node_dir / "manifest.yaml").read_text(encoding="utf-8"))
    upstream = manifest["upstream"]
    assert upstream["license"] == "MIT"
    assert upstream["license_file"] == "normalization_impl/LICENSE.gigaspeechbench"
    assert {entry["project"] for entry in upstream["third_party"]} >= {
        "speechio/chinese_text_normalization",
        "Zhon",
    }


def test_vendored_sources_keep_upstream_attribution_headers() -> None:
    impl_dir = (
        Path(__file__).resolve().parents[1]
        / "src/sure_eval/evaluation/nodes/normalization/giga_norm/normalization_impl"
    )
    for name in ("CHN.py", "USA.py"):
        header = (impl_dir / name).read_text(encoding="utf-8")[:2000]
        assert "speechio/chinese_text_normalization" in header
        assert "tsroten/zhon" in header
        assert "MIT License" in header


def test_declared_node_env_verify_files_exist() -> None:
    """A typo in node_env.yaml would silently disable the env check."""

    import yaml

    node_dir = (
        Path(__file__).resolve().parents[1]
        / "src/sure_eval/evaluation/nodes/normalization/giga_norm"
    )
    node_env = yaml.safe_load((node_dir / "node_env.yaml").read_text(encoding="utf-8"))

    declared = node_env["verify"]["files"]
    assert declared, "the IDN profile depends on the ref_code tables"
    for relative_path in declared:
        assert (node_dir / relative_path).is_file(), relative_path


# --------------------------------------------------------------------------- #
# Node-local environment
# --------------------------------------------------------------------------- #


def test_giga_norm_env_check_declares_optional_pip_runtime() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    result = NodeEnvChecker().check_node("normalization/giga_norm")

    assert result.runtime == "pip_optional"
    assert result.status == ("ok" if _giga_extra_installed() else "failed")


def test_giga_norm_env_check_fails_loudly_when_a_dependency_is_missing(monkeypatch) -> None:
    from sure_eval.evaluation import env_check
    from sure_eval.evaluation.env_check import NodeEnvChecker

    original_find_spec = env_check.importlib.util.find_spec

    def _fake_find_spec(name: str):
        if name == "zhconv":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(env_check.importlib.util, "find_spec", _fake_find_spec)

    result = NodeEnvChecker().check_node("normalization/giga_norm")

    assert result.status == "failed"
    assert "zhconv" in result.details["missing_imports"]
    assert "pip install" in result.fix


def test_giga_norm_env_setup_dry_run_builds_pip_command() -> None:
    from typer.testing import CliRunner

    from sure_eval.cli import app

    result = CliRunner().invoke(
        app, ["env", "setup", "--node", "normalization/giga_norm", "--dry-run", "--json"]
    )

    assert result.exit_code == 0, result.stdout
    action = json.loads(result.stdout)["actions"][0]
    assert action["node_id"] == "normalization/giga_norm"
    assert action["runtime"] == "pip"
    assert action["packages"] == [
        "whisper_normalizer>=0.0.10",
        "zhconv>=1.4.3",
        "num2words>=0.5.13",
        "underthesea>=6.8.0",
    ]
    assert "python -m pip install" in action["command"]


# --------------------------------------------------------------------------- #
# Text rules (require the [giga] extra)
# --------------------------------------------------------------------------- #


@requires_giga_extra
def test_giga_chn_profile_applies_itn_and_character_tokenization() -> None:
    from sure_eval.evaluation.nodes.normalization.giga_norm import normalize_giga_text

    assert normalize_giga_text("我有123个苹果", profile="CHN") == "我 有 1 2 3 个 苹 果"
    assert normalize_giga_text("我有一百二十三个苹果", profile="CHN") == "我 有 1 2 3 个 苹 果"


@requires_giga_extra
def test_giga_chn_profile_keeps_parenthesized_content_like_upstream() -> None:
    """Upstream replaces the brackets with spaces and keeps the inner text."""

    from sure_eval.evaluation.nodes.normalization.giga_norm import normalize_giga_text

    assert normalize_giga_text("自己会去处理它，(noise) 然后继续。", profile="CHN") == (
        "自 己 会 去 处 理 它 noise 然 后 继 续"
    )


@requires_giga_extra
def test_giga_usa_profile_applies_whisper_english_rules() -> None:
    from sure_eval.evaluation.nodes.normalization.giga_norm import normalize_giga_text

    assert normalize_giga_text("I have 123 apples", profile="USA") == "i have 1 2 3 apples"
    assert normalize_giga_text("I've got TWO apples.", profile="USA") == "i have got 2 apples"


@requires_giga_extra
def test_giga_normalization_preserves_keys_and_records_provenance(tmp_path: Path) -> None:
    from sure_eval.evaluation.core.types import KeyTextFiles
    from sure_eval.evaluation.nodes.normalization.giga_norm import normalize_giga_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "我有123个苹果"), ("utt2", "[laugh]")])
    _write_key_text(hyp_file, [("utt1", "我有一百二十三个苹果"), ("utt2", "")])

    normalized, trace = normalize_giga_asr_files(
        KeyTextFiles(ref_file=str(ref_file), hyp_file=str(hyp_file)),
        language="zh",
    )

    try:
        expected = "utt1\t我 有 1 2 3 个 苹 果\nutt2\t\n"
        assert Path(normalized.ref_file).read_text(encoding="utf-8") == expected
        assert Path(normalized.hyp_file).read_text(encoding="utf-8") == expected
        assert trace.node_id == "normalization/giga_norm"
        assert trace.version == "v1"
        assert trace.details["profile"] == "CHN"
        assert trace.details["num_rows"] == {"ref": 2, "hyp": 2}
        assert trace.details["num_empty_after_normalization"] == {"ref": 1, "hyp": 1}
        normalization = trace.details["normalization"]
        assert normalization["backend"] == "gigaspeechbench_text_norm"
        assert normalization["upstream_project"] == "GigaSpeechBench"
        assert normalization["vendored"] is True
        assert "whisper_normalizer" in normalization["package_versions"]
        assert "zhconv" in normalization["package_versions"]
    finally:
        Path(normalized.ref_file).unlink(missing_ok=True)
        Path(normalized.hyp_file).unlink(missing_ok=True)


@requires_giga_extra
@pytest.mark.parametrize(
    ("language", "metric", "normalizer", "expected_pipeline_id", "rows"),
    [
        # Explicit bare selector on a language that has a different default route.
        (
            "zh",
            "cer",
            "giga",
            "asr.zh.cer.giga_norm_chn_v1.wenet_cer_v1",
            [("utt1", "我有123个苹果", "我有一百二十三个苹果")],
        ),
        (
            "en",
            "wer",
            "giga",
            "asr.en.wer.giga_norm_usa_v1.wenet_wer_v1",
            [("utt1", "I have 123 apples", "I have one hundred twenty three apples")],
        ),
        # Explicit selector on a language whose default route is FunASR.
        (
            "ja",
            "cer",
            "giga",
            "asr.ja.cer.giga_norm_jpn_v1.wenet_cer_v1",
            [("utt1", "今日は123円です。", "今日は一二三円です")],
        ),
        (
            "ar-eg",
            "wer",
            None,
            "asr.ar_eg.wer.giga_norm_egy_v1.wenet_wer_v1",
            [("utt1", "مرحبا بالعالم", "مرحبا بالعالم")],
        ),
    ],
)
def test_giga_normalizer_selector_resolves_to_the_expected_route(
    tmp_path: Path,
    language: str,
    metric: str,
    normalizer: str | None,
    expected_pipeline_id: str,
    rows: list[tuple[str, str, str]],
) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [(key, ref) for key, ref, _ in rows])
    _write_key_text(hyp_file, [(key, hyp) for key, _, hyp in rows])

    report = evaluate_asr_files(
        str(ref_file),
        str(hyp_file),
        language=language,
        metric=metric,
        normalizer=normalizer,
    )

    assert report.pipeline_id == expected_pipeline_id
    assert report.computation_node_ids[0] == "normalization/giga_norm"
    assert report.score == pytest.approx(0.0)


@requires_giga_extra
@pytest.mark.parametrize(
    ("pipeline_id", "metric", "nodes", "rows"),
    [
        (
            "asr.zh.cer.giga_norm_chn_v1.wenet_cer_v1",
            "cer",
            ("normalization/giga_norm", "scoring/wenet_cer"),
            [
                ("utt1", "我有123个苹果", "我有一百二十三个苹果"),
                ("utt2", "今天天气不错", "今天天气不错"),
            ],
        ),
        (
            "asr.en.wer.giga_norm_usa_v1.wenet_wer_v1",
            "wer",
            ("normalization/giga_norm", "scoring/wenet_wer"),
            [
                ("utt1", "I have 123 apples", "I have one hundred twenty three apples"),
                ("utt2", "hello world", "hello world"),
            ],
        ),
        (
            "asr.ja.cer.giga_norm_jpn_v1.wenet_cer_v1",
            "cer",
            ("normalization/giga_norm", "scoring/wenet_cer"),
            [
                ("utt1", "今日は123円です。", "今日は一二三円です"),
                ("utt2", "ありがとう", "ありがとう"),
            ],
        ),
    ],
)
def test_giga_route_describe_run_report_preserves_identity(
    tmp_path: Path,
    pipeline_id: str,
    metric: str,
    nodes: tuple[str, ...],
    rows: list[tuple[str, str, str]],
) -> None:
    """describe -> run -> report must preserve the selected route identity."""

    from sure_eval.evaluation.scripts import describe_pipeline, run_task

    description = describe_pipeline("asr", pipeline_id=pipeline_id)
    assert description.pipeline_id == pipeline_id
    assert description.metric == metric
    assert description.execution_metrics == (metric,)
    assert description.pipeline_kind == "atomic"
    assert description.member_pipeline_ids == ()
    assert description.node_ids == nodes
    assert description.computation_node_ids == nodes

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [(key, ref) for key, ref, _ in rows])
    _write_key_text(hyp_file, [(key, hyp) for key, _, hyp in rows])

    out_dir = tmp_path / "eval"
    report = run_task(
        "asr",
        ref_file=str(ref_file),
        hyp_file=str(hyp_file),
        pipeline_id=pipeline_id,
        output_dir=str(out_dir),
    )

    assert report.pipeline_id == pipeline_id
    assert report.metric == metric
    assert report.computation_node_ids == nodes
    assert report.score == pytest.approx(0.0)
    # The report trace must contain every selected node, in route order.
    assert tuple(entry.node_id for entry in report.pipeline_trace) == nodes
    assert report.pipeline_trace[0].stage == "normalization"
    assert report.pipeline_trace[0].version == "v1"
    assert report.pipeline_trace[-1].stage == "scoring"

    assert (out_dir / "report.json").exists()
    assert (out_dir / "pipeline_description.json").exists()
    written_report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    written_description = json.loads(
        (out_dir / "pipeline_description.json").read_text(encoding="utf-8")
    )
    assert written_report["pipeline_id"] == pipeline_id
    assert written_description["pipeline_id"] == pipeline_id
    assert written_description["pipeline_kind"] == "atomic"
    assert tuple(written_description["computation_node_ids"]) == nodes
