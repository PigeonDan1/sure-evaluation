from __future__ import annotations

import json
from pathlib import Path
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sure_eval.cli import app


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _require_wetext_node_env() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    result = NodeEnvChecker().check_node("normalization/wetext_norm")
    if result.status != "ok":
        pytest.skip(f"wetext_norm node-local environment is not prepared: {result.message}")


@dataclass
class _FakeErrorRate:
    error_rate: float
    errors: int = 0
    length: int = 1
    missed_speaker_time: float = 0.0
    falarm_speaker_time: float = 0.0
    speaker_error_time: float = 0.0


def _install_fake_meeteval(monkeypatch) -> None:
    def load(path: str):
        return f"loaded:{Path(path).name}"

    def cpwer(reference, hypothesis):
        return {"session": _FakeErrorRate(error_rate=0.5, errors=1, length=2)}

    def combine_error_rates(error_rates):
        rates = list(error_rates)
        return _FakeErrorRate(
            error_rate=sum(rate.errors for rate in rates) / sum(rate.length for rate in rates),
            errors=sum(rate.errors for rate in rates),
            length=sum(rate.length for rate in rates),
        )

    def dscore(reference, hypothesis, *, collar):
        return {"session": _FakeErrorRate(error_rate=0.25)}

    monkeypatch.setitem(
        sys.modules,
        "meeteval",
        SimpleNamespace(
            io=SimpleNamespace(load=load),
            wer=SimpleNamespace(cpwer=cpwer, combine_error_rates=combine_error_rates),
            der=SimpleNamespace(dscore=dscore),
        ),
    )


def test_metric_describe_outputs_route_backed_pipeline_json(tmp_path: Path) -> None:
    runner = CliRunner()
    pipeline_path = tmp_path / "asr_pipeline.json"

    result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "asr",
            "--language",
            "zh",
            "--metric",
            "cer",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert payload["task"] == "asr"
    assert payload["pipeline_id"] == "asr.zh.cer.wetext_norm_zh_itn_v1.wenet_cer_v1"
    assert payload["run_args"]["output_dir"] is None
    assert payload["run_args"]["ref_file"] is None
    assert payload["run_args"]["hyp_file"] is None
    assert [slot["slot"] for slot in payload["pipeline"]] == ["normalization", "scoring"]
    assert payload["pipeline"][0]["nullable"] is True
    assert payload["pipeline"][0]["selected"] == "default"
    assert payload["pipeline"][0]["default"] == "normalization/wetext_norm"
    assert payload["pipeline"][1]["nullable"] is False
    assert payload["pipeline"][1]["metric"] == "cer"
    assert "scoring/wenet_cer" in payload["pipeline"][1]["choices"]
    assert payload["task_config_path"] == "src/sure_eval/evaluation/tasks/asr/manifest.yaml"
    assert payload["route_config_path"] == "src/sure_eval/evaluation/tasks/asr/routes.yaml"
    assert payload["describe_entrypoint"] == "sure_eval.evaluation.scripts.asr.describe_pipeline"
    assert payload["script_entrypoint"] == "sure_eval.evaluation.scripts.asr.run"
    assert payload["executor"] == "sure_eval.evaluation.tasks.asr.pipeline.evaluate_asr_files"


def test_metric_describe_can_select_asr_pipeline_id(tmp_path: Path) -> None:
    runner = CliRunner()
    pipeline_path = tmp_path / "asr_canonical_pipeline.json"

    result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "asr",
            "--pipeline-id",
            "asr.cs.mer.canonical_itn_cs_v1.token_mer_v1",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert payload["metric"] == "mer"
    assert payload["requested_metric"] == "mer"
    assert payload["metrics"] == ["mer"]
    assert payload["pipeline_id"] == "asr.cs.mer.canonical_itn_cs_v1.token_mer_v1"
    assert payload["computation_node_ids"] == ["normalization/canonical_itn", "scoring/token_mer"]
    assert payload["route_config_path"] == "src/sure_eval/evaluation/tasks/asr/routes.yaml"
    assert "internal_executor_metric" not in json.dumps(payload, ensure_ascii=False)


def test_metric_run_executes_pipeline_file_and_writes_outputs(tmp_path: Path) -> None:
    _require_wetext_node_env()
    runner = CliRunner()
    pipeline_path = tmp_path / "asr_pipeline.json"
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    output_dir = tmp_path / "metric_out"
    _write_key_text(ref_file, [("utt1", "你好世界"), ("utt2", "今天天气")])
    _write_key_text(hyp_file, [("utt1", "你好世界"), ("utt2", "今天天气")])

    describe_result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "asr",
            "--language",
            "zh",
            "--metric",
            "cer",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe_result.exit_code == 0, describe_result.stdout

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--ref-file",
            str(ref_file),
            "--hyp-file",
            str(hyp_file),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert run_result.exit_code == 0, run_result.stdout
    stdout_payload = json.loads(run_result.stdout)
    assert stdout_payload["status"] == "ok"
    assert stdout_payload["pipeline_id"] == "asr.zh.cer.wetext_norm_zh_itn_v1.wenet_cer_v1"
    assert stdout_payload["report_path"] == str(output_dir / "report.json")
    assert "node-local environments are not validated" in stdout_payload["environment_note"]
    assert stdout_payload["node_config_paths"] == [
        "src/sure_eval/evaluation/nodes/normalization/wetext_norm/manifest.yaml",
        "src/sure_eval/evaluation/nodes/scoring/wenet_wer/manifest.yaml",
    ]
    assert (output_dir / "report.json").exists()
    assert (output_dir / "pipeline_description.json").exists()
    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report_payload["score"] == 0.0
    assert report_payload["pipeline_trace"][0]["node_id"] == "normalization/wetext_norm"
    assert report_payload["pipeline_trace"][0]["details"]["profile"] == "zh_itn"


def test_metric_run_rejects_node_choice_not_declared_by_describe(tmp_path: Path) -> None:
    runner = CliRunner()
    pipeline_path = tmp_path / "asr_pipeline.json"
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好世界")])
    _write_key_text(hyp_file, [("utt1", "你好世界")])

    result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "asr",
            "--language",
            "zh",
            "--metric",
            "cer",
            "--output",
            str(pipeline_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    payload["pipeline"][0]["selected"] = "normalization/unknown_norm"
    pipeline_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--ref-file",
            str(ref_file),
            "--hyp-file",
            str(hyp_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--json",
        ],
    )

    assert run_result.exit_code == 1
    assert "not declared in choices" in run_result.stdout


def test_metric_describe_lists_dynamic_route_choices_from_routes_yaml() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["metric", "describe", "s2tt", "--language", "zh", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    route_metrics = {route["metric"] for route in payload["route_choices"]}
    assert {"bleu", "bleu_char", "chrf", "xcomet_xl", "bleurt_20"}.issubset(route_metrics)
    scoring_choices = next(slot for slot in payload["pipeline"] if slot["slot"] == "scoring")[
        "choices"
    ]
    assert "scoring/sacrebleu" in scoring_choices


def test_metric_describe_outputs_sa_asr_meeteval_pipeline() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["metric", "describe", "sa-asr", "--metric", "cpwer", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["task"] == "sa_asr"
    assert payload["pipeline_id"] == "sa_asr.en.cpwer.conversion_sa_asr_cpwer_v1.gstar_norm_v1.meeteval_v1"
    assert payload["run_args"]["ref_file"] is None
    assert payload["run_args"]["hyp_file"] is None
    assert payload["pipeline"][0]["default"] == "normalization/gstar_norm"
    assert payload["pipeline"][0]["nullable"] is True
    assert payload["pipeline"][1]["default"] == "scoring/meeteval"
    assert payload["pipeline"][1]["nullable"] is False
    assert payload["conversion_steps"][0]["id"] == "sa_asr__cpwer"
    assert "model" not in payload["conversion_steps"][0]
    assert payload["conversion_steps"][0]["script"].endswith("conversion/sa_asr__cpwer/convert.py")


def test_metric_run_executes_sd_meeteval_pipeline(monkeypatch, tmp_path: Path) -> None:
    _install_fake_meeteval(monkeypatch)
    runner = CliRunner()
    pipeline_path = tmp_path / "sd_pipeline.json"
    ref_file = tmp_path / "ref.rttm"
    hyp_file = tmp_path / "hyp.rttm"
    output_dir = tmp_path / "sd_out"
    ref_file.write_text("SPEAKER rec1 1 0.00 1.00 <NA> <NA> spk1 <NA> <NA>\n", encoding="utf-8")
    hyp_file.write_text("SPEAKER rec1 1 0.00 1.00 <NA> <NA> hyp1 <NA> <NA>\n", encoding="utf-8")

    describe_result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "sd",
            "--metric",
            "der",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe_result.exit_code == 0, describe_result.stdout

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--ref-file",
            str(ref_file),
            "--hyp-file",
            str(hyp_file),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert run_result.exit_code == 0, run_result.stdout
    payload = json.loads(run_result.stdout)
    assert payload["pipeline_id"] == "sd.any.der.meeteval_v1"
    assert payload["score"] == 0.25
    assert (output_dir / "report.json").exists()


def test_metric_run_executes_ser_with_builtin_label_spec(tmp_path: Path) -> None:
    runner = CliRunner()
    pipeline_path = tmp_path / "ser_pipeline.json"
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    output_dir = tmp_path / "ser_out"
    _write_key_text(ref_file, [("utt1", "hap"), ("utt2", "ang")])
    _write_key_text(hyp_file, [("utt1", "happy"), ("utt2", "2")])

    describe_result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "ser",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe_result.exit_code == 0, describe_result.stdout
    description_payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert description_payload["required_roles"] == ["hyp", "ref"]
    assert description_payload["optional_roles"] == ["label_spec"]

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--ref-file",
            str(ref_file),
            "--hyp-file",
            str(hyp_file),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert run_result.exit_code == 0, run_result.stdout
    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report_payload["details"]["label_spec"]["id"] == "ser_default"


def test_metric_run_executes_slu_prompt_norm_pipeline(tmp_path: Path) -> None:
    runner = CliRunner()
    pipeline_path = tmp_path / "slu_pipeline.json"
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    prompt_jsonl = tmp_path / "prompt.jsonl"
    output_dir = tmp_path / "slu_out"
    _write_key_text(ref_file, [("utt1", "B"), ("utt2", "C")])
    _write_key_text(hyp_file, [("utt1", "B"), ("utt2", "option C")])
    prompt_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "key": "utt1",
                        "choices": [
                            {"id": "A", "text": "cat"},
                            {"id": "B", "text": "dog"},
                            {"id": "C", "text": "bird"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "key": "utt2",
                        "choices": [
                            {"id": "A", "text": "red"},
                            {"id": "B", "text": "blue"},
                            {"id": "C", "text": "option C"},
                        ],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    describe_result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "slu",
            "--metric",
            "accuracy",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe_result.exit_code == 0, describe_result.stdout

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--ref-file",
            str(ref_file),
            "--hyp-file",
            str(hyp_file),
            "--prompt-jsonl",
            str(prompt_jsonl),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert run_result.exit_code == 0, run_result.stdout
    payload = json.loads(run_result.stdout)
    assert payload["pipeline_id"] == "slu.any.accuracy.prompt_norm_choice_id_v1.classify_v1"
    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report_payload["pipeline_trace"][0]["node_id"] == "normalization/prompt_norm"


def test_metric_run_executes_kws_macro_recall_pipeline(tmp_path: Path) -> None:
    runner = CliRunner()
    pipeline_path = tmp_path / "kws_pipeline.json"
    reference_jsonl = tmp_path / "reference.jsonl"
    sample_output = tmp_path / "sample_output.json"
    output_dir = tmp_path / "kws_out"
    _write_jsonl(
        reference_jsonl,
        [
            {"key": "pos_high", "expected": "detect", "text": "嗨小问", "duration": 1.0},
            {"key": "pos_low", "expected": "detect", "text": "嗨小问", "duration": 1.0},
            {"key": "neg_high", "expected": "reject", "duration": 3600.0},
        ],
    )
    sample_output.write_text(
        json.dumps(
            [
                {
                    "key": "pos_high",
                    "result": {"detected": True, "keyword": "嗨小问", "score": 0.9},
                },
                {"key": "pos_low", "result": {"detected": True, "keyword": "嗨小问", "score": 0.7}},
                {
                    "key": "neg_high",
                    "result": {"detected": True, "keyword": "嗨小问", "score": 0.8},
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    describe_result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "kws",
            "--metric",
            "macro-recall",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe_result.exit_code == 0, describe_result.stdout
    description_payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert description_payload["metric"] == "macro_recall"
    assert description_payload["pipeline_id"] == (
        "kws.any.macro_recall.conversion_kws_sure_json_to_samples_v1.wekws_det_v1"
    )

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--reference-jsonl",
            str(reference_jsonl),
            "--sample-output",
            str(sample_output),
            "--macro-recall-false-alarms",
            "0",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert run_result.exit_code == 0, run_result.stdout
    payload = json.loads(run_result.stdout)
    assert payload["metric"] == "macro_recall"
    assert payload["score"] == 0.5
    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report_payload["score"] == 0.5
    assert report_payload["details"]["results"]["accuracy"]["score"] == pytest.approx(2 / 3)
    assert report_payload["details"]["results"]["macro-recall"]["score"] == 0.5


def test_metric_run_human_mode_prints_environment_note(tmp_path: Path) -> None:
    runner = CliRunner()
    pipeline_path = tmp_path / "asr_pipeline.json"
    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好世界")])
    _write_key_text(hyp_file, [("utt1", "你好世界")])

    describe_result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "asr",
            "--language",
            "zh",
            "--metric",
            "cer",
            "--output",
            str(pipeline_path),
        ],
    )
    assert describe_result.exit_code == 0, describe_result.stdout

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--ref-file",
            str(ref_file),
            "--hyp-file",
            str(hyp_file),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert run_result.exit_code == 0, run_result.stdout
    assert "Environment note" in run_result.stdout
    assert "pyproject.toml or uv.lock" in run_result.stdout


def test_metric_describe_tts_accepts_metrics_alias_and_samples_role(tmp_path: Path) -> None:
    runner = CliRunner()
    pipeline_path = tmp_path / "tts_pipeline.json"

    result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "tts",
            "--language",
            "zh",
            "--metrics",
            "tts_cer,sim/wavlm-large,dnsmos",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert payload["task"] == "tts"
    assert payload["metric"] == "multi"
    assert payload["pipeline_id"] == (
        "tts.zh.multi.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1."
        "punctuation_strip_norm_v1.wenet_cer_v1__spk_sim.wavlm_large_sim_v1__"
        "dnsmos.dnsmos_v1"
    )
    assert payload["pipeline_kind"] == "bundle"
    assert payload["member_pipeline_ids"] == [
        "tts.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1.punctuation_strip_norm_v1.wenet_cer_v1",
        "tts.zh.spk_sim.wavlm_large_sim_v1",
        "tts.zh.dnsmos.dnsmos_v1",
    ]
    assert payload["run_args"]["samples_jsonl"] is None
    assert payload["required_roles"] == ["samples_jsonl"]
    assert "frontend/funasr_loader_16k_mono" in [slot["default"] for slot in payload["pipeline"]]
    assert "transcription/paraformer_zh" in [slot["default"] for slot in payload["pipeline"]]


def test_metric_run_executes_tts_samples_jsonl_with_standard_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation.nodes.transcription import StaticTranscriber

    import sure_eval.evaluation.audio_runtime as audio_runtime

    runner = CliRunner()
    pipeline_path = tmp_path / "tts_pipeline.json"
    samples_jsonl = tmp_path / "samples.jsonl"
    output_dir = tmp_path / "tts_out"
    (tmp_path / "generated.wav").write_bytes(b"fake")
    (tmp_path / "reference.wav").write_bytes(b"fake")
    _write_jsonl(
        samples_jsonl,
        [
            {
                "sample_id": "utt1",
                "prediction_audio": "generated.wav",
                "reference_audio": "reference.wav",
                "reference_text": "你好世界",
                "language": "zh",
            }
        ],
    )
    monkeypatch.setattr(
        audio_runtime,
        "build_tts_runtime",
        lambda **kwargs: {
            "transcribers": {"zh": StaticTranscriber("你好世界")},
            "speaker_providers": {"wavlm-large": lambda prediction, reference, **_: {"ASV": 0.7}},
            "mos_providers": {"dnsmos": lambda prediction, reference="", **_: {"OVRL": 3.1}},
        },
    )

    describe_result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "tts",
            "--language",
            "zh",
            "--metrics",
            "tts_cer,sim/wavlm-large,dnsmos",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe_result.exit_code == 0, describe_result.stdout

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--samples-jsonl",
            str(samples_jsonl),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert run_result.exit_code == 0, run_result.stdout
    payload = json.loads(run_result.stdout)
    assert payload["pipeline_id"] == (
        "tts.zh.multi.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1."
        "punctuation_strip_norm_v1.wenet_cer_v1__spk_sim.wavlm_large_sim_v1__"
        "dnsmos.dnsmos_v1"
    )
    assert payload["score"] == 0.0
    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report_payload["details"]["results"]["spk_sim"]["score"] == 0.7
    assert report_payload["details"]["results"]["dnsmos"]["score"] == 3.1
    assert (output_dir / "pipeline_description.json").exists()


def test_metric_run_executes_vc_samples_jsonl_with_standard_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation.nodes.transcription import StaticTranscriber

    import sure_eval.evaluation.audio_runtime as audio_runtime

    runner = CliRunner()
    pipeline_path = tmp_path / "vc_pipeline.json"
    samples_jsonl = tmp_path / "samples.jsonl"
    output_dir = tmp_path / "vc_out"
    for name in ("converted.wav", "reference.wav", "source.wav"):
        (tmp_path / name).write_bytes(b"fake")
    _write_jsonl(
        samples_jsonl,
        [
            {
                "sample_id": "utt1",
                "converted_audio": "converted.wav",
                "reference_audio": "reference.wav",
                "source_audio": "source.wav",
                "reference_text": "你好世界",
                "language": "zh",
            }
        ],
    )
    monkeypatch.setattr(
        audio_runtime,
        "build_vc_runtime",
        lambda **kwargs: {
            "transcribers": {"zh": StaticTranscriber("你好世界")},
            "speaker_providers": {"ecapa-tdnn": lambda prediction, reference, **_: {"ASV": 0.8}},
            "mos_providers": {"utmos": lambda prediction, reference="", **_: {"utmos": 3.7}},
        },
    )

    describe_result = runner.invoke(
        app,
        [
            "metric",
            "describe",
            "vc",
            "--language",
            "zh",
            "--metrics",
            "vc_cer,sim/ecapa-tdnn,utmos",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe_result.exit_code == 0, describe_result.stdout

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--samples-jsonl",
            str(samples_jsonl),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert run_result.exit_code == 0, run_result.stdout
    payload = json.loads(run_result.stdout)
    assert payload["pipeline_id"] == (
        "vc.zh.multi.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1."
        "punctuation_strip_norm_v1.wenet_cer_v1__spk_sim.ecapa_tdnn_sim_v1__"
        "utmos.utmos_v1"
    )
    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report_payload["details"]["results"]["spk_sim"]["score"] == 0.8
    assert report_payload["details"]["results"]["utmos"]["score"] == 3.7
    assert (output_dir / "pipeline_description.json").exists()
