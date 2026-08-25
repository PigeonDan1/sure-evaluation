from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _require_nemo_node_env() -> None:
    node_dir = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "sure_eval"
        / "evaluation"
        / "nodes"
        / "normalization"
        / "nemo_norm"
    )
    if not (node_dir / ".venv" / "bin" / "python").exists():
        pytest.skip(
            "nemo_norm node-local environment is not prepared. "
            "Run: sure-eval env setup --node normalization/nemo_norm"
        )


def test_nemo_norm_uses_pinned_package_without_vendored_source() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    node_dir = repo_root / "src/sure_eval/evaluation/nodes/normalization/nemo_norm"
    node_pyproject = (node_dir / "pyproject.toml").read_text(encoding="utf-8")
    node_env = yaml.safe_load((node_dir / "node_env.yaml").read_text(encoding="utf-8"))

    assert '"nemo-text-processing==1.2.0"' in node_pyproject
    assert node_env["runtime"]["frozen"] is True
    assert (node_dir / "uv.lock").is_file()
    assert not (node_dir / "vendor").exists()
    assert not (node_dir / "ar").exists()


def test_asr_ar_route_describes_nemo_tn_and_wenet_cer() -> None:
    from sure_eval.evaluation.scripts.asr import describe_pipeline

    description = describe_pipeline(language="ar")

    assert description.pipeline_id == "asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1"
    assert description.language == "ar"
    assert description.metric == "cer"
    assert description.node_ids == (
        "normalization/nemo_norm",
        "scoring/wenet_cer",
    )


def test_asr_ar_nemo_normalizer_converts_written_number_to_spoken_form() -> None:
    from sure_eval.evaluation.nodes.normalization.nemo_norm import normalize_nemo_text

    _require_nemo_node_env()
    normalized = normalize_nemo_text("21", cache_dir=None)
    assert normalized == "واحد وعشرون"


def test_asr_ar_route_scores_raw_key_text_with_nemo_normalization(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    _require_nemo_node_env()
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1\t21 كتابا\n", encoding="utf-8")
    hyp_file.write_text("utt1\tواحد وعشرون كتابا\n", encoding="utf-8")

    report = evaluate_asr_files(
        str(ref_file),
        str(hyp_file),
        language="ar",
        metric="cer",
    )

    assert report.score == 0.0
    assert report.pipeline_id == "asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1"
    assert report.pipeline_trace[0].node_id == "normalization/nemo_norm"


class _FakeNormalizer:
    def normalize(self, text: str, *, verbose: bool) -> str:
        assert verbose is False
        return text.upper()


def test_nemo_key_text_parser_preserves_empty_text_and_reports_stats(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.normalization.nemo_norm.node import (
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


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("utt1 missing-tab\n", "line 1"),
        ("\tmissing-key\n", "Empty key"),
        ("utt1\tone\nutt1\ttwo\n", "Duplicate key 'utt1'"),
        ("\n", "No <key><TAB><text> rows"),
    ],
)
def test_nemo_key_text_parser_rejects_invalid_input(
    content: str,
    message: str,
    tmp_path: Path,
) -> None:
    from sure_eval.evaluation.nodes.normalization.nemo_norm.node import (
        _normalize_key_text_file,
    )

    input_file = tmp_path / "input.txt"
    output_file = tmp_path / "output.txt"
    input_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _normalize_key_text_file(str(input_file), str(output_file), _FakeNormalizer())


def test_nemo_node_env_checker_recognizes_node() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    result = NodeEnvChecker().check_node("normalization/nemo_norm")
    assert result.node_id == "normalization/nemo_norm"
    assert result.runtime == "node_local_project"
    assert result.status in {"ok", "failed"}
