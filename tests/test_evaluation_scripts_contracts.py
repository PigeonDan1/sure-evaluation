from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml


def _write_key_text(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{text}\n" for key, text in rows), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


@dataclass
class _FakeErrorRate:
    error_rate: float
    errors: int = 0
    length: int = 1
    missed_speaker_time: float = 0.0
    falarm_speaker_time: float = 0.0
    speaker_error_time: float = 0.0


def _install_fake_meeteval(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    calls: dict[str, object] = {"loaded": [], "cpwer": [], "dscore": []}

    def load(path: str):
        calls["loaded"].append(path)
        return f"loaded:{Path(path).name}"

    def cpwer(reference, hypothesis):
        calls["cpwer"].append((reference, hypothesis))
        return {
            "session_a": _FakeErrorRate(error_rate=0.25, errors=1, length=4),
            "session_b": _FakeErrorRate(error_rate=0.5, errors=2, length=4),
        }

    def combine_error_rates(error_rates):
        rates = list(error_rates)
        errors = sum(rate.errors for rate in rates)
        length = sum(rate.length for rate in rates)
        return _FakeErrorRate(error_rate=errors / length, errors=errors, length=length)

    def dscore(reference, hypothesis, *, collar):
        calls["dscore"].append((reference, hypothesis, collar))
        return {
            "session_a": _FakeErrorRate(
                error_rate=0.1,
                missed_speaker_time=0.2,
                falarm_speaker_time=0.1,
                speaker_error_time=0.0,
            ),
            "session_b": _FakeErrorRate(
                error_rate=0.3,
                missed_speaker_time=0.0,
                falarm_speaker_time=0.1,
                speaker_error_time=0.2,
            ),
        }

    fake_meeteval = SimpleNamespace(
        io=SimpleNamespace(load=load),
        wer=SimpleNamespace(cpwer=cpwer, combine_error_rates=combine_error_rates),
        der=SimpleNamespace(dscore=dscore),
    )
    monkeypatch.setitem(sys.modules, "meeteval", fake_meeteval)
    return calls


def _write_annotation(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_asr_script_describes_pipeline_contract() -> None:
    from sure_eval.evaluation.scripts.asr import describe_pipeline

    description = describe_pipeline(language="en", metric="wer")

    assert description.pipeline_id == "asr.en.wer.whisper_norm.wenet_wer"
    assert description.node_ids == ("normalization/whisper_norm", "scoring/wenet_wer")
    assert description.required_roles == ("hyp", "ref")
    assert description.output_dir_required is True


def test_asr_routes_declare_nodes_contract_and_executor() -> None:
    routes = yaml.safe_load(
        Path("src/sure_eval/evaluation/tasks/asr/routes.yaml").read_text(encoding="utf-8")
    )
    route = next(
        item
        for item in routes["routes"]
        if item["language"] == "zh" and item["metric"] == "cer"
    )

    assert route["pipeline_id"] == "asr.zh.cer.aispeech_norm.wenet_cer"
    assert route["nodes"] == ["normalization/aispeech_norm", "scoring/wenet_cer"]
    assert route["input_contract"] == "scoring/wenet_cer"
    assert route["executor"] == "sure_eval.evaluation.tasks.asr.pipeline.evaluate_asr_files"


def test_s2tt_script_describes_model_metric_inputs() -> None:
    from sure_eval.evaluation.scripts.s2tt import describe_pipeline

    description = describe_pipeline(metric="xcomet_xl")

    assert description.node_ids == ("scoring/xcomet_xl",)
    assert description.required_roles == ("src", "hyp", "ref")
    assert description.contracts[0]["model"] == "Unbabel/XCOMET-XL"


def test_s2tt_routes_declare_nodes_contract_and_executor() -> None:
    routes = yaml.safe_load(
        Path("src/sure_eval/evaluation/tasks/s2tt/routes.yaml").read_text(encoding="utf-8")
    )
    route = next(
        item
        for item in routes["routes"]
        if item["language"] == "zh" and item["metric"] == "xcomet_xl"
    )

    assert route["pipeline_id"] == "s2tt.zh.xcomet_xl.xcomet_xl"
    assert route["nodes"] == ["scoring/xcomet_xl"]
    assert route["input_contract"] == "scoring/xcomet_xl"
    assert route["executor"] == "sure_eval.evaluation.tasks.s2tt.pipeline.evaluate_s2tt_files"


def test_slu_script_unions_prompt_norm_and_classify_inputs() -> None:
    from sure_eval.evaluation.scripts.slu import describe_pipeline

    description = describe_pipeline()

    assert description.node_ids == ("normalization/prompt_norm", "scoring/classify")
    assert description.required_roles == ("hyp", "ref", "prompt_jsonl")
    assert description.optional_roles == ("label_spec",)


def test_sd_script_describes_meeteval_der_route() -> None:
    from sure_eval.evaluation.scripts.sd import describe_pipeline

    description = describe_pipeline(metric="der")

    assert description.pipeline_id == "sd.der.meeteval"
    assert description.node_ids == ("scoring/meeteval",)
    assert description.required_roles == ("hyp", "ref")
    assert description.contracts[0]["row_format"] == "meeteval_annotation"
    assert description.contracts[0]["aggregation"] == "session_mean_error_rate"


def test_sa_asr_script_describes_meeteval_cpwer_route_with_der_companion() -> None:
    from sure_eval.evaluation.scripts.sa_asr import describe_pipeline

    description = describe_pipeline(metric="cpwer")

    assert description.pipeline_id == "sa_asr.cpwer.gstar_norm.meeteval"
    assert description.node_ids == ("normalization/gstar_norm", "scoring/meeteval")
    assert description.required_roles == ("hyp", "ref")
    assert description.contracts[0]["row_format"] == "meeteval_annotation"
    assert description.contracts[0]["aggregation"] == "meeteval_combined_error_rate"
    assert any(item["id"] == "sa_asr__cpwer" for item in description.conversion_steps)


def test_simple_task_routes_declare_nodes_contract_and_executor() -> None:
    expected = {
        "kws": {
            "metric": "accuracy",
            "pipeline_id": "kws.sure_json.accuracy.wekws_det",
            "nodes": ["scoring/wekws_det"],
            "input_contract": "sure_json",
            "executor": "sure_eval.evaluation.tasks.kws.pipeline.evaluate_kws_files",
        },
        "classification": {
            "metric": "accuracy",
            "pipeline_id": "classification.accuracy.classify",
            "nodes": ["scoring/classify"],
            "input_contract": "scoring/classify",
            "executor": "sure_eval.evaluation.tasks.classification.pipeline.evaluate_classification_files",
        },
        "slu": {
            "metric": "accuracy",
            "pipeline_id": "slu.accuracy.prompt_norm.classify.choice_id",
            "nodes": ["normalization/prompt_norm", "scoring/classify"],
            "input_contract": "prompt_norm_classify",
            "executor": "sure_eval.evaluation.tasks.slu.pipeline.evaluate_slu_files",
        },
    }
    for task, expected_route in expected.items():
        routes = yaml.safe_load(
            Path(f"src/sure_eval/evaluation/tasks/{task}/routes.yaml").read_text(encoding="utf-8")
        )
        route = routes["routes"][0]
        for key, value in expected_route.items():
            assert route[key] == value


def test_tts_script_merges_multi_metric_input_contracts() -> None:
    from sure_eval.evaluation.scripts.tts import describe_pipeline

    description = describe_pipeline(language="zh", metrics=["tts_cer", "sim/wavlm-large", "dnsmos"])

    assert description.node_ids == (
        "frontend/funasr_loader_16k_mono",
        "transcription/paraformer_zh",
        "normalization/punctuation_strip_norm",
        "scoring/wenet_cer",
        "scoring/wavlm_large_sim",
        "scoring/dnsmos",
    )
    assert description.required_roles == ("prediction_audio", "reference_text", "reference_audio")


def test_tts_routes_declare_semantic_speaker_and_mos_nodes() -> None:
    routes = yaml.safe_load(
        Path("src/sure_eval/evaluation/tasks/tts/routes.yaml").read_text(encoding="utf-8")
    )
    by_metric = {route["metric"]: route for route in routes["routes"]}
    zh_semantic = next(
        route
        for route in routes["routes"]
        if route.get("language") == "zh" and route["metric"] == "tts_cer"
    )

    assert zh_semantic["nodes"] == [
        "frontend/funasr_loader_16k_mono",
        "transcription/paraformer_zh",
        "normalization/punctuation_strip_norm",
        "scoring/wenet_cer",
    ]
    en_semantic = next(
        route
        for route in routes["routes"]
        if route.get("language") == "en" and route["metric"] == "tts_wer"
    )
    assert en_semantic["nodes"] == [
        "transcription/whisper_large_v3",
        "normalization/whisper_norm",
        "scoring/wenet_wer",
    ]
    assert zh_semantic["input_contract"] == "semantic/asr_error_rate"
    assert by_metric["sim/wavlm-large"]["nodes"] == ["scoring/wavlm_large_sim"]
    assert by_metric["sim/wavlm-large"]["input_contract"] == "scoring/wavlm_large_sim"
    assert by_metric["dnsmos"]["nodes"] == ["scoring/dnsmos"]
    assert by_metric["dnsmos"]["input_contract"] == "scoring/dnsmos"


def test_main_flow_audio_handoff_defaults_tts_full_metric_suite_by_language() -> None:
    if not Path("docs/agents/main_flow_agent/templates/run_single_model.sh").exists():
        pytest.skip("main-flow templates are not part of the standalone evaluation package")
    template = Path("docs/agents/main_flow_agent/templates/run_single_model.sh").read_text(
        encoding="utf-8"
    )
    single_dataset_template = Path(
        "docs/agents/main_flow_agent/templates/run_single_model_single_dataset.sh"
    ).read_text(encoding="utf-8")

    for content in (template, single_dataset_template):
        assert "AUDIO_EVAL_METRICS" in content
        assert 'metrics.append("tts_cer" if language.startswith(("zh", "cmn", "yue")) else "tts_wer")' in content
        assert 'metrics.extend(["sim/wavlm-large", "sim/ecapa-tdnn", "sim/eres2net", "dnsmos", "wv-mos", "utmos"])' in content
        assert 'metrics.append("vc_cer" if language.startswith(("zh", "cmn", "yue")) else "vc_wer")' in content
        assert '"metrics": "$AUDIO_EVAL_METRICS"' in content


def test_audio_evaluation_only_probe_uses_language_matched_transcriber() -> None:
    if not Path("docs/agents/main_flow_agent/templates/run_audio_evaluation_only.sh").exists():
        pytest.skip("main-flow templates are not part of the standalone evaluation package")
    template = Path("docs/agents/main_flow_agent/templates/run_audio_evaluation_only.sh").read_text(
        encoding="utf-8"
    )

    assert "ParaformerZHTranscriber" in template
    assert "WhisperLargeV3Transcriber" in template
    assert 'metric in {"tts_cer", "vc_cer"}' in template
    assert "paraformer_zh/checkpoints" in template
    assert "whisper_large_v3/checkpoints" in template
    assert "semantic_probe_cases" in template
    assert "_metric_applies_to_language" in template
    assert 'workspace_root = Path("/workspace/sure-eval")' in template
    assert "Path.cwd() / relative_to_workspace" in template
    assert '"metric": metric' in template
    probe_marker = 'echo "[preflight] one-sample semantic transcription"'
    probe_start = template.index(probe_marker)
    heredoc_start = template.index("<<'PY'\n", probe_start) + len("<<'PY'\n")
    heredoc_end = template.index("\nPY\nfi", heredoc_start)
    compile(template[heredoc_start:heredoc_end], "run_audio_evaluation_only_probe.py", "exec")


def test_f5tts_materialized_audio_evaluation_does_not_pin_semantic_only_metrics() -> None:
    model_dir = Path("src/sure_eval/models/SWivid__F5-TTS_Emilia-ZH-EN")
    if not model_dir.exists():
        pytest.skip("model eval_runs are not part of the standalone evaluation package")
    expected_metrics = {
        "main_agent_f5tts_seedtts_en_005": "tts_wer sim/wavlm-large sim/ecapa-tdnn sim/eres2net dnsmos wv-mos utmos",
        "main_agent_f5tts_seedtts_zh_hard_005": "tts_cer sim/wavlm-large sim/ecapa-tdnn sim/eres2net dnsmos wv-mos utmos",
        "main_agent_f5tts_seedtts_zh_005": "tts_cer sim/wavlm-large sim/ecapa-tdnn sim/eres2net dnsmos wv-mos utmos",
    }

    for run_id, metrics in expected_metrics.items():
        run_dir = model_dir / "eval_runs" / run_id
        handoff_path = run_dir / "evaluation_handoff.json"
        script = (run_dir / "run_audio_evaluation_only.sh").read_text(encoding="utf-8")
        run_script = (run_dir / "run_evaluation.sh").read_text(encoding="utf-8")
        if handoff_path.exists():
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            assert handoff["metrics"] == metrics
        assert f'METRICS="${{METRICS:-{metrics}}}"' in script
        assert f'METRICS="${{METRICS:-{metrics}}}"' in run_script
        assert 'METRICS="${METRICS:-tts_wer}"' not in script
        assert 'METRICS="${METRICS:-tts_cer}"' not in script


def test_audio_evaluation_only_template_supports_segmented_tts_metric_merge() -> None:
    if not Path("docs/agents/main_flow_agent/templates/run_audio_evaluation_only.sh").exists():
        pytest.skip("main-flow templates are not part of the standalone evaluation package")
    template = Path("docs/agents/main_flow_agent/templates/run_audio_evaluation_only.sh").read_text(
        encoding="utf-8"
    )

    assert "AUDIO_EVAL_MODE" in template
    assert "AUDIO_EVAL_SEGMENT" in template
    assert "SEGMENT_PAYLOADS" in template
    assert "--merge-payload" in template
    assert "segment_tts_semantic" in template
    assert "segment_tts_speaker_eres2net" in template
    assert "segment_tts_mos_dnsmos" in template
    assert "segment_tts_mos_wvmos" in template
    assert "segment_tts_mos_utmos" in template
    assert "segment_tts_mos_dnsmos_wvmos" not in template


def test_qwen_materialized_audio_evaluation_uses_language_matched_probe() -> None:
    if not Path("src/sure_eval/models/Qwen__Qwen3-TTS-12Hz-1.7B-Base").exists():
        pytest.skip("model eval_runs are not part of the standalone evaluation package")
    script = Path(
        "src/sure_eval/models/Qwen__Qwen3-TTS-12Hz-1.7B-Base/"
        "eval_runs/main_agent_Qwen__Qwen3-TTS-12Hz-1.7B-Base_001/"
        "run_audio_evaluation_only.sh"
    ).read_text(encoding="utf-8")

    assert 'SURE_TTS_AUDIO_RUNTIME="${SURE_TTS_AUDIO_RUNTIME:-node_local}"' in script
    assert "semantic_probe_cases" in script
    assert "_metric_applies_to_language" in script
    assert 'workspace_root = Path("/workspace/sure-eval")' in script
    assert "Path.cwd() / relative_to_workspace" in script
    assert '"dataset": dataset' in script


def test_audio_evaluation_only_segment_preflight_does_not_require_orchestration_torch() -> None:
    if not Path("docs/agents/main_flow_agent/templates/run_audio_evaluation_only.sh").exists():
        pytest.skip("main-flow templates are not part of the standalone evaluation package")
    template = Path("docs/agents/main_flow_agent/templates/run_audio_evaluation_only.sh").read_text(
        encoding="utf-8"
    )

    assert 'for name in ("yaml",):' in template
    assert 'for name in ("torch", "yaml"):' not in template
    assert 'for name in ("pydantic", "torch", "yaml"):' not in template


def test_audio_evaluation_only_forwards_device_to_evaluate_predictions() -> None:
    if not Path("docs/agents/main_flow_agent/templates/run_audio_evaluation_only.sh").exists():
        pytest.skip("main-flow templates are not part of the standalone evaluation package")
    template = Path("docs/agents/main_flow_agent/templates/run_audio_evaluation_only.sh").read_text(
        encoding="utf-8"
    )
    script = Path("scripts/evaluate_predictions.py").read_text(encoding="utf-8")

    assert '--device "$DEVICE"' in template
    assert 'SURE_TTS_AUDIO_RUNTIME="${SURE_TTS_AUDIO_RUNTIME:-node_local}"' in template
    assert 'SURE_EVAL_MINIMAL_DATASET_MANAGER="${SURE_EVAL_MINIMAL_DATASET_MANAGER:-1}"' in template
    assert 'parser.add_argument("--device", default="cuda"' in script
    assert 'device=args.device' in script
    assert 'build_tts_runtime(metrics=(metric,), language=language, device=device' in script


def test_evaluate_predictions_imports_without_pydantic_config_dependency(monkeypatch) -> None:
    import builtins
    import importlib.util
    import sys

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sure_eval.core.config" or name.startswith("pydantic"):
            raise ModuleNotFoundError("No module named 'pydantic'", name="pydantic")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module_path = Path("scripts/evaluate_predictions.py")
    spec = importlib.util.spec_from_file_location("evaluate_predictions_no_pydantic_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "evaluate_predictions_no_pydantic_test", module)
    spec.loader.exec_module(module)

    manager, rps = module._build_dataset_and_rps_managers(None)
    assert manager.normalize_dataset_name("seedtts_test_eval_en") == "seedtts_test_eval_en"
    assert getattr(rps, "database", None) is None


def test_evaluate_predictions_can_force_minimal_dataset_manager(monkeypatch) -> None:
    import importlib.util
    import sys

    module_path = Path("scripts/evaluate_predictions.py")
    spec = importlib.util.spec_from_file_location("evaluate_predictions_minimal_dataset_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    monkeypatch.setenv("SURE_EVAL_MINIMAL_DATASET_MANAGER", "1")

    manager, rps = module._build_dataset_and_rps_managers(None)

    assert type(manager).__name__ == "_SimpleDatasetManager"
    assert getattr(rps, "database", None) is None


def test_evaluate_predictions_can_merge_segment_payloads_into_standard_artifacts(tmp_path: Path) -> None:
    import importlib.util
    import sys

    module_path = Path("scripts/evaluate_predictions.py")
    spec = importlib.util.spec_from_file_location("evaluate_predictions_for_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    dataset = "tiny_tts_en"
    run_dir = tmp_path / "run"
    model_dir = tmp_path / "model"
    jsonl_path = tmp_path / f"{dataset}.jsonl"
    prediction_path = run_dir / "predictions" / f"{dataset}.txt"
    metric_dir = run_dir / "segments" / "semantic" / "metrics" / dataset / "tts_wer"
    metric_dir.mkdir(parents=True)
    prediction_path.parent.mkdir(parents=True)
    model_dir.mkdir()
    jsonl_path.write_text(
        json.dumps(
            {
                "key": "utt1",
                "task": "TTS",
                "language": "en",
                "target": "hello",
                "reference_audio": "ref.wav",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_path.write_text("utt1\thyp.wav\n", encoding="utf-8")
    (metric_dir / "report.json").write_text(
        json.dumps(
            {
                "task": "TTS",
                "language": "en",
                "metric": "tts_wer",
                "score": 0.25,
                "pipeline_id": "tts.en.tts_wer.whisper_large_v3.whisper_norm.wenet_wer",
                "details": {"results": {"tts_wer": {"score": 0.25}}},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metric_dir / "pipeline_description.json").write_text(
        json.dumps(
            {
                "task": "TTS",
                "metric": "tts_wer",
                "language": "en",
                "pipeline_id": "tts.en.tts_wer.whisper_large_v3.whisper_norm.wenet_wer",
                "nodes": [{"node_id": "transcription/whisper_large_v3", "stage": "transcription"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    segment_payload = tmp_path / "semantic_payload.json"
    segment_payload.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "dataset": dataset,
                        "jsonl_path": str(jsonl_path),
                        "prediction_path": str(prediction_path),
                        "task": "TTS",
                        "language": "en",
                        "metric": "tts_wer",
                        "baseline_dataset": dataset,
                        "score": 0.25,
                        "rps": {"status": "metric_not_comparable_to_baseline"},
                        "rps_is_unbounded": False,
                        "num_samples": 1,
                        "evaluation_context": {"route": "segment"},
                        "pipeline_id": "tts.en.tts_wer.whisper_large_v3.whisper_norm.wenet_wer",
                        "metric_artifact_dir": str(metric_dir),
                        "metric_report_path": str(metric_dir / "report.json"),
                        "pipeline_description_path": str(metric_dir / "pipeline_description.json"),
                        "pipeline_description": {},
                        "details": {"score": 0.25},
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    results = module.merge_payload_results([segment_payload])
    module.write_standard_evaluation_artifacts(
        results=results,
        run_dir=run_dir,
        results_dir=tmp_path / "results",
        tool_name="f5tts-test",
        protocol_id="strict_core",
        model_dir=model_dir,
        validation_payload={
            "results": [
                {
                    "dataset": dataset,
                    "expected_samples": 1,
                    "provided_predictions": 1,
                    "is_valid": True,
                }
            ]
        },
        output_path=run_dir / "evaluation_payload.json",
    )

    payload = json.loads((run_dir / "evaluation_payload.json").read_text(encoding="utf-8"))
    payload_text = (run_dir / "evaluation_payload.json").read_text(encoding="utf-8")
    result_row = payload["results"][0]
    assert payload["schema"] == "sure.eval.payload.v2"
    assert payload["results"][0]["metric"] == "tts_wer"
    assert result_row["result"]["score"] == 0.25
    assert result_row["result"]["wer"] == 0.25
    assert result_row["result"]["score_key"] == "wer"
    assert "details" not in result_row
    assert "per_sample" not in payload_text
    assert '"rows"' not in payload_text
    assert (run_dir / "metrics" / dataset / "tts_wer" / "report.json").exists()
    assert (run_dir / "metrics" / dataset / "tts_wer" / "pipeline_description.json").exists()
    assert (run_dir / "report.jsonl").exists()
    assert (run_dir / "protocol.yaml").exists()
    assert (run_dir / "sample_reports" / dataset / "tts_wer.jsonl").exists()
    assert json.loads((run_dir / "report.jsonl").read_text(encoding="utf-8").splitlines()[0])["metric"]["score"] == 0.25


def test_evaluate_predictions_payload_v2_places_tts_sim_and_mos_values_in_result() -> None:
    import importlib.util
    import sys

    module_path = Path("scripts/evaluate_predictions.py")
    spec = importlib.util.spec_from_file_location("evaluate_predictions_payload_v2_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    base = {
        "dataset": "tiny_tts_en",
        "jsonl_path": "data/datasets/sure_benchmark/jsonl/tiny_tts_en.jsonl",
        "prediction_path": "run/predictions/tiny_tts_en.txt",
        "task": "TTS",
        "language": "en",
        "baseline_dataset": "tiny_tts_en",
        "rps": {"status": "missing_baseline"},
        "num_samples": 2,
        "evaluation_context": {"metric_source": "cli_override"},
        "pipeline_id": "pipeline",
        "metric_artifact_dir": "run/metrics/tiny_tts_en/metric",
        "metric_report_path": "run/metrics/tiny_tts_en/metric/report.json",
        "pipeline_description_path": "run/metrics/tiny_tts_en/metric/pipeline_description.json",
        "pipeline_description": {"nodes": []},
        "sample_report_path": "run/sample_reports/tiny_tts_en/metric.jsonl",
    }
    payload = module._build_evaluation_payload(
        [
            {
                **base,
                "metric": "sim/wavlm-large",
                "score": 0.7,
                "details": {
                    "source": {"score_key": "ASV"},
                    "per_sample": [
                        {"ASV": 0.6, "score": 0.6, "backend": "wavlm-large-cosine"},
                        {"ASV": 0.8, "score": 0.8, "backend": "wavlm-large-cosine"},
                    ],
                },
            },
            {
                **base,
                "metric": "dnsmos",
                "score": 3.1,
                "details": {
                    "source": {"score_key": "OVRL"},
                    "per_sample": [
                        {"OVRL": 3.0, "SIG": 3.4, "BAK": 4.0, "P808_MOS": 3.8},
                        {"OVRL": 3.2, "SIG": 3.6, "BAK": 4.2, "P808_MOS": 4.0},
                    ],
                },
            },
            {
                **base,
                "metric": "wv-mos",
                "score": 4.0,
                "details": {"source": {"score_key": "mos"}},
            },
            {
                **base,
                "metric": "utmos",
                "score": 3.7,
                "details": {"source": {"score_key": "utmos"}},
            },
        ]
    )

    rows = {row["metric"]: row for row in payload["results"]}
    rendered = json.dumps(payload, ensure_ascii=False)
    assert payload["schema"] == "sure.eval.payload.v2"
    assert rows["sim/wavlm-large"]["result"]["similarity"] == 0.7
    assert rows["sim/wavlm-large"]["result"]["backend"] == "wavlm-large-cosine"
    assert rows["dnsmos"]["result"]["OVRL"] == 3.1
    assert rows["dnsmos"]["result"]["mos"] == 3.1
    assert rows["dnsmos"]["result"]["mean_SIG"] == pytest.approx(3.5)
    assert rows["dnsmos"]["result"]["mean_BAK"] == pytest.approx(4.1)
    assert rows["dnsmos"]["result"]["mean_P808_MOS"] == pytest.approx(3.9)
    assert rows["wv-mos"]["result"]["mos"] == 4.0
    assert rows["utmos"]["result"]["mos"] == 3.7
    assert all("details" not in row for row in payload["results"])
    assert "per_sample" not in rendered
    assert '"rows"' not in rendered


def test_evaluate_predictions_payload_v2_can_recover_summary_from_metric_artifact(tmp_path: Path) -> None:
    import importlib.util
    import sys

    module_path = Path("scripts/evaluate_predictions.py")
    spec = importlib.util.spec_from_file_location("evaluate_predictions_payload_artifact_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    sim_report = tmp_path / "sim_report.json"
    sim_report.write_text(
        json.dumps(
            {
                "details": {
                    "results": {
                        "sim/wavlm-large": {
                            "source": {"score_key": "ASV"},
                            "per_sample": [
                                {"ASV": 0.6, "score": 0.6, "backend": "wavlm-large-cosine"},
                                {"ASV": 0.8, "score": 0.8, "backend": "wavlm-large-cosine"},
                            ],
                        }
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    dnsmos_report = tmp_path / "dnsmos_report.json"
    dnsmos_report.write_text(
        json.dumps(
            {
                "details": {
                    "results": {
                        "dnsmos": {
                            "source": {"score_key": "OVRL"},
                            "per_sample": [
                                {"OVRL": 3.0, "SIG": 3.4, "BAK": 4.0, "P808_MOS": 3.8},
                                {"OVRL": 3.2, "SIG": 3.6, "BAK": 4.2, "P808_MOS": 4.0},
                            ],
                        }
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    sim_result = module._compact_metric_result(
        {
            "metric": "sim/wavlm-large",
            "score": 0.7,
            "num_samples": 2,
            "details": {"metric_name": "sim/wavlm-large", "score": 0.7, "similarity": 0.7},
            "metric_report_path": str(sim_report),
        }
    )
    dnsmos_result = module._compact_metric_result(
        {
            "metric": "dnsmos",
            "score": 3.1,
            "num_samples": 2,
            "details": {"metric_name": "dnsmos", "score": 3.1, "OVRL": 3.1},
            "metric_report_path": str(dnsmos_report),
        }
    )

    assert sim_result["similarity"] == 0.7
    assert sim_result["backend"] == "wavlm-large-cosine"
    assert dnsmos_result["OVRL"] == 3.1
    assert dnsmos_result["mos"] == 3.1
    assert dnsmos_result["mean_SIG"] == pytest.approx(3.5)
    assert dnsmos_result["mean_BAK"] == pytest.approx(4.1)
    assert dnsmos_result["mean_P808_MOS"] == pytest.approx(3.9)


def test_main_flow_tts_experience_is_not_generalized_to_all_tasks_or_vc() -> None:
    if not Path("docs/agents/main_flow_agent/AGENTS.md").exists():
        pytest.skip("main-flow docs are not part of the standalone evaluation package")
    agents = Path("docs/agents/main_flow_agent/AGENTS.md").read_text(encoding="utf-8")
    readiness = Path(
        "docs/agents/main_flow_agent/contracts/main_agent_execution_readiness_unit.md"
    ).read_text(encoding="utf-8")
    prediction_contract = Path(
        "docs/agents/main_flow_agent/contracts/prediction_generation_contract.md"
    ).read_text(encoding="utf-8")
    audio_eval_contract = Path(
        "docs/agents/main_flow_agent/contracts/tts_vc_audio_evaluation_surface.md"
    ).read_text(encoding="utf-8")
    normalized_agents = " ".join(agents.split())

    assert "[SYSTEM_CONSTRAINT: TTS_SINGLE_PROCESS_INFERENCE_ENV]" in agents
    assert "[SYSTEM_CONSTRAINT: TTS_INFERENCE_RUNTIME_GUARDRAILS]" in agents
    assert "[SYSTEM_CONSTRAINT: TTS_VC_SINGLE_PROCESS_INFERENCE_ENV]" not in agents
    assert "[SYSTEM_CONSTRAINT: TTS_VC_INFERENCE_RUNTIME_GUARDRAILS]" not in agents
    assert "MUST NOT be generalized to non-TTS tasks" in normalized_agents
    assert "MUST NOT be applied to VC without separate VC run evidence" in normalized_agents
    assert "本次 TTS 音频生成经验" in readiness
    assert "不应泛化到所有 main-flow 任务" in readiness
    assert "不应泛化到 VC" in readiness
    assert "classified as a TTS inference-input or runtime-surface issue" in prediction_contract
    assert "classified as a TTS/VC inference-input or runtime-surface issue" not in prediction_contract
    assert "TTS and VC runs have two independent execution surfaces" in audio_eval_contract


def test_evaluate_predictions_tts_runtime_uses_canonical_audio_runtime(monkeypatch) -> None:
    import importlib.util
    import sys

    import sure_eval.evaluation.audio_runtime as audio_runtime

    module_path = Path("scripts/evaluate_predictions.py")
    spec = importlib.util.spec_from_file_location("evaluate_predictions_for_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    from sure_eval.evaluation.nodes.transcription.common.providers import (
        NodeLocalTranscriber,
    )

    calls: list[dict[str, object]] = []
    original = audio_runtime.build_tts_runtime

    def wrapped_build_tts_runtime(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(audio_runtime, "build_tts_runtime", wrapped_build_tts_runtime)

    zh_runtime = module.build_tts_runtime(metrics=("tts_cer",), language="zh", device="cpu")
    en_runtime = module.build_tts_runtime(metrics=("tts_wer",), language="en", device="cpu")

    assert calls == [
        {"metrics": ("tts_cer",), "language": "zh", "device": "cpu", "cache_dir": None},
        {"metrics": ("tts_wer",), "language": "en", "device": "cpu", "cache_dir": None},
    ]
    assert isinstance(zh_runtime["transcribers"]["zh"], NodeLocalTranscriber)
    assert zh_runtime["transcribers"]["zh"].node_id == "transcription/paraformer_zh"
    assert isinstance(en_runtime["transcribers"]["en"], NodeLocalTranscriber)
    assert en_runtime["transcribers"]["en"].node_id == "transcription/whisper_large_v3"


def test_evaluate_predictions_tts_runtime_supports_in_process_batch_mode(monkeypatch) -> None:
    import importlib.util
    import sys

    module_path = Path("scripts/evaluate_predictions.py")
    spec = importlib.util.spec_from_file_location("evaluate_predictions_for_in_process_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    class FakeProvider:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setenv("SURE_TTS_AUDIO_RUNTIME", "in_process")
    monkeypatch.setattr(
        "sure_eval.evaluation.nodes.scoring.common.speaker_providers.ERes2NetSimilarityProvider",
        FakeProvider,
    )
    monkeypatch.setattr(
        "sure_eval.evaluation.nodes.scoring.common.speaker_providers.ERes2NetEmbeddingProvider",
        FakeProvider,
    )

    runtime = module.build_tts_runtime(metrics=("sim/eres2net",), language="en", device="cpu")

    assert isinstance(runtime["speaker_providers"]["eres2net"], FakeProvider)


def test_evaluate_predictions_remaps_workspace_audio_paths(monkeypatch) -> None:
    import importlib.util
    import sys

    module_path = Path("scripts/evaluate_predictions.py")
    spec = importlib.util.spec_from_file_location("evaluate_predictions_path_remap_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPO_ROOT", Path.cwd())
    target = Path.cwd() / "src/sure_eval/models/model/run.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"fake")

    resolved = module._resolve_prediction_audio(
        "/workspace/sure-eval/src/sure_eval/models/model/run.wav",
        Path("src/sure_eval/models/model/eval_runs/run/predictions/dataset.txt"),
    )

    assert resolved == str(target.resolve())
    target.unlink()


def test_vc_script_can_describe_text_or_audio_reference_semantic_routes() -> None:
    from sure_eval.evaluation.scripts.vc import describe_pipeline

    text_route = describe_pipeline(language="en", metrics=["vc_wer"], reference_mode="text")
    audio_route = describe_pipeline(language="en", metrics=["vc_wer"], reference_mode="audio")

    assert text_route.required_roles == ("converted_audio", "reference_text")
    assert audio_route.required_roles == ("converted_audio", "reference_audio")


def test_vc_routes_declare_reference_mode_specific_semantic_nodes() -> None:
    routes = yaml.safe_load(
        Path("src/sure_eval/evaluation/tasks/vc/routes.yaml").read_text(encoding="utf-8")
    )
    text_route = next(
        route
        for route in routes["routes"]
        if route.get("language") == "en" and route["metric"] == "vc_wer" and route["reference_mode"] == "text"
    )
    audio_route = next(
        route
        for route in routes["routes"]
        if route.get("language") == "en" and route["metric"] == "vc_wer" and route["reference_mode"] == "audio"
    )

    assert text_route["nodes"] == [
        "transcription/whisper_large_v3",
        "normalization/whisper_norm",
        "scoring/wenet_wer",
    ]
    assert text_route["input_contract"] == "semantic/asr_error_rate_with_text"
    assert audio_route["nodes"] == [
        "transcription/whisper_large_v3",
        "transcription/whisper_large_v3",
        "normalization/whisper_norm",
        "scoring/wenet_wer",
    ]
    assert audio_route["input_contract"] == "semantic/asr_error_rate_with_audio_reference"


def test_asr_script_run_writes_report_and_pipeline_description(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts.asr import run

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    output_dir = tmp_path / "out"
    _write_key_text(ref_file, [("utt1", "我有2个苹果。")])
    _write_key_text(hyp_file, [("utt1", "我有两个苹果")])

    report = run(
        ref_file=str(ref_file),
        hyp_file=str(hyp_file),
        language="zh",
        metric="cer",
        output_dir=str(output_dir),
    )

    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    description_payload = json.loads((output_dir / "pipeline_description.json").read_text(encoding="utf-8"))
    assert report_payload["pipeline_id"] == report.pipeline_id
    assert report_payload["task"] == "ASR"
    assert report_payload["pipeline_trace"][0]["node_id"] == "normalization/aispeech_norm"
    assert description_payload["pipeline_id"] == "asr.zh.cer.aispeech_norm.wenet_cer"
    assert description_payload["nodes"] == [
        {
            "node_id": "normalization/aispeech_norm",
            "stage": "normalization",
            "version": "v1",
            "manifest_path": "src/sure_eval/evaluation/nodes/normalization/aispeech_norm/manifest.yaml",
        },
        {
            "node_id": "scoring/wenet_cer",
            "stage": "scoring",
            "version": "v1",
            "manifest_path": "src/sure_eval/evaluation/nodes/scoring/wenet_wer/manifest.yaml",
        },
    ]
    assert description_payload["required_roles"] == ["hyp", "ref"]


def test_classification_script_run_writes_label_spec_report(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts.classification import run

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    output_dir = tmp_path / "classification_out"
    _write_key_text(ref_file, [("utt1", "hap"), ("utt2", "ang")])
    _write_key_text(hyp_file, [("utt1", "happy"), ("utt2", "2")])

    run(
        ref_file=str(ref_file),
        hyp_file=str(hyp_file),
        task="SER",
        output_dir=str(output_dir),
    )

    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    description_payload = json.loads((output_dir / "pipeline_description.json").read_text(encoding="utf-8"))
    assert report_payload["task"] == "SER"
    assert report_payload["metric"] == "accuracy"
    assert report_payload["details"]["label_spec"]["id"] == "ser_default"
    assert description_payload["pipeline_id"] == "ser.accuracy.classify"
    assert description_payload["required_roles"] == ["hyp", "ref"]
    assert description_payload["optional_roles"] == ["label_spec"]


def test_kws_script_run_writes_sure_json_route_report(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts.kws import run

    reference_jsonl = tmp_path / "reference.jsonl"
    sample_output = tmp_path / "sample_output.json"
    output_dir = tmp_path / "kws_out"
    _write_jsonl(
        reference_jsonl,
        [
            {"key": "utt1", "audio": "utt1.wav", "expected": True, "expected_keyword": "小爱同学"},
            {"key": "utt2", "audio": "utt2.wav", "expected": False},
        ],
    )
    sample_output.write_text(
        json.dumps(
            [
                {"key": "utt1", "result": {"detected": True, "keyword": "小爱同学", "score": 0.9}},
                {"key": "utt2", "result": {"detected": False, "score": 0.1}},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = run(
        reference_jsonl=str(reference_jsonl),
        sample_output=str(sample_output),
        output_dir=str(output_dir),
    )

    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    description_payload = json.loads((output_dir / "pipeline_description.json").read_text(encoding="utf-8"))
    assert report.pipeline_id == "kws.sure_json.accuracy.wekws_det"
    assert report_payload["pipeline_id"] == "kws.sure_json.accuracy.wekws_det"
    assert description_payload["pipeline_id"] == "kws.sure_json.accuracy.wekws_det"
    assert report_payload["pipeline_trace"][0]["node_id"] == "scoring/wekws_det"


def test_sd_script_run_writes_meeteval_params_and_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_meeteval(monkeypatch)
    from sure_eval.evaluation.scripts.sd import run

    ref_file = tmp_path / "ref.rttm"
    hyp_file = tmp_path / "hyp.rttm"
    output_dir = tmp_path / "sd_out"
    _write_annotation(ref_file, ["SPEAKER rec1 1 0.00 1.00 <NA> <NA> spk1 <NA> <NA>"])
    _write_annotation(hyp_file, ["SPEAKER rec1 1 0.00 1.00 <NA> <NA> hyp1 <NA> <NA>"])

    report = run(ref_file=str(ref_file), hyp_file=str(hyp_file), output_dir=str(output_dir))

    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report.task == "SD"
    assert report.score == pytest.approx(0.2)
    assert report_payload["pipeline_id"] == "sd.der.meeteval"
    assert report_payload["details"]["scoring_result"]["der"] == pytest.approx(0.2)
    assert report_payload["pipeline_trace"][0]["node_id"] == "scoring/meeteval"
    assert report_payload["pipeline_trace"][0]["details"]["params"]["collar"] == 0.25
    assert report_payload["pipeline_trace"][0]["details"]["input_loader"] == "meeteval.io.load"


def test_sa_asr_script_run_reports_cpwer_and_der_companion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _install_fake_meeteval(monkeypatch)
    from sure_eval.evaluation.scripts.sa_asr import run

    ref_file = tmp_path / "ref.stm"
    hyp_file = tmp_path / "hyp.stm"
    output_dir = tmp_path / "sa_asr_out"
    _write_annotation(ref_file, ["rec1 1 spk1 0.00 1.00 hello world"])
    _write_annotation(hyp_file, ["rec1 1 hyp1 0.00 1.00 hello there"])

    report = run(ref_file=str(ref_file), hyp_file=str(hyp_file), output_dir=str(output_dir))

    report_payload = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report.task == "SA-ASR"
    assert report.metric == "cpwer"
    assert report.score == pytest.approx(0.375)
    assert report_payload["pipeline_id"] == "sa_asr.cpwer.gstar_norm.meeteval"
    assert report_payload["details"]["scoring_result"]["cpwer"] == pytest.approx(0.375)
    assert report_payload["details"]["scoring_result"]["der"] == pytest.approx(0.2)
    assert report_payload["pipeline_trace"][0]["node_id"] == "normalization/gstar_norm"
    assert report_payload["pipeline_trace"][0]["details"]["input_schema"] == "key_text_files"
    assert report_payload["pipeline_trace"][1]["details"]["params"]["companion_metrics"] == ["der"]
    assert report_payload["details"]["conversion_trace"][0]["id"] == "sa_asr__cpwer"
    assert report_payload["details"]["conversion_trace"][0]["source_format"] == "stm"
    assert any(item["target_format"] == "stm" for item in report_payload["details"]["conversion_trace"])
    assert (output_dir / "conversion" / "sa_asr__cpwer" / "ref.txt").exists()
    assert (output_dir / "conversion" / "sa_asr__cpwer" / "ref.normalized.stm").exists()
    assert calls["dscore"][0][2] == 0.5


def test_unified_script_entrypoint_dispatches_describe_and_run(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts.run import describe_pipeline, run_task

    description = describe_pipeline("asr", language="zh", metric="cer")
    assert description.pipeline_id == "asr.zh.cer.aispeech_norm.wenet_cer"

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    output_dir = tmp_path / "entrypoint_out"
    _write_key_text(ref_file, [("utt1", "你好世界")])
    _write_key_text(hyp_file, [("utt1", "你好世界")])

    report = run_task(
        "asr",
        ref_file=str(ref_file),
        hyp_file=str(hyp_file),
        language="zh",
        metric="cer",
        output_dir=str(output_dir),
    )

    assert report.score == 0.0
    assert (output_dir / "report.json").exists()
    assert (output_dir / "pipeline_description.json").exists()


def test_unified_script_entrypoint_dispatches_sd_and_sa_asr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_meeteval(monkeypatch)
    from sure_eval.evaluation.scripts.run import describe_pipeline, run_task

    sd_description = describe_pipeline("sd", metric="der")
    sa_asr_description = describe_pipeline("sa-asr", metric="cpwer")

    assert sd_description.pipeline_id == "sd.der.meeteval"
    assert sa_asr_description.pipeline_id == "sa_asr.cpwer.gstar_norm.meeteval"

    ref_file = tmp_path / "ref.stm"
    hyp_file = tmp_path / "hyp.stm"
    _write_annotation(ref_file, ["rec1 1 spk1 0.00 1.00 hello world"])
    _write_annotation(hyp_file, ["rec1 1 hyp1 0.00 1.00 hello there"])

    report = run_task(
        "sa-asr",
        ref_file=str(ref_file),
        hyp_file=str(hyp_file),
        output_dir=str(tmp_path / "sa_asr_entrypoint_out"),
    )

    assert report.pipeline_id == "sa_asr.cpwer.gstar_norm.meeteval"
    assert report.details["scoring_result"]["der"] == pytest.approx(0.2)


def test_asr_script_run_uses_executor_declared_by_route(monkeypatch, tmp_path: Path) -> None:
    import sure_eval.evaluation.tasks.asr.pipeline as asr_pipeline
    from sure_eval.evaluation.core.types import EvaluationReport
    from sure_eval.evaluation.scripts.asr import run

    calls: dict[str, dict[str, str]] = {}

    def fake_executor(**kwargs):
        calls["kwargs"] = kwargs
        return EvaluationReport(
            task="ASR",
            language=kwargs["language"],
            metric=kwargs["metric"],
            score=0.0,
            pipeline_id="asr.zh.cer.aispeech_norm.wenet_cer",
            pipeline_trace=(),
        )

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好世界")])
    _write_key_text(hyp_file, [("utt1", "你好世界")])
    monkeypatch.setattr(asr_pipeline, "evaluate_asr_files", fake_executor)

    run(
        ref_file=str(ref_file),
        hyp_file=str(hyp_file),
        language="zh",
        metric="cer",
        output_dir=str(tmp_path / "out"),
    )

    assert calls["kwargs"]["ref_file"] == str(ref_file)
    assert calls["kwargs"]["hyp_file"] == str(hyp_file)
    assert calls["kwargs"]["metric"] == "cer"


def test_asr_script_run_rejects_route_pipeline_id_mismatch(monkeypatch, tmp_path: Path) -> None:
    import sure_eval.evaluation.tasks.asr.pipeline as asr_pipeline
    from sure_eval.evaluation.core.types import EvaluationReport
    from sure_eval.evaluation.scripts.asr import run

    def fake_executor(**kwargs):
        return EvaluationReport(
            task="ASR",
            language=kwargs["language"],
            metric=kwargs["metric"],
            score=0.0,
            pipeline_id="asr.zh.cer.unexpected",
            pipeline_trace=(),
        )

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_text(ref_file, [("utt1", "你好世界")])
    _write_key_text(hyp_file, [("utt1", "你好世界")])
    monkeypatch.setattr(asr_pipeline, "evaluate_asr_files", fake_executor)

    with pytest.raises(ValueError, match="pipeline_id mismatch"):
        run(
            ref_file=str(ref_file),
            hyp_file=str(hyp_file),
            language="zh",
            metric="cer",
            output_dir=str(tmp_path / "out"),
        )


def test_tts_and_vc_nonsemantic_descriptions_match_runtime_aggregate_pipeline_ids() -> None:
    from sure_eval.evaluation.scripts.tts import describe_pipeline as describe_tts
    from sure_eval.evaluation.scripts.vc import describe_pipeline as describe_vc

    tts_description = describe_tts(language="zh", metrics=["dnsmos"])
    vc_description = describe_vc(language="en", metrics=["dnsmos"])

    assert tts_description.pipeline_id == "tts.zh.multi.audio_metric_nodes"
    assert vc_description.pipeline_id == "vc.en.multi.audio_metric_nodes"


def test_tts_script_run_calls_task_level_executor_with_normalized_metrics(monkeypatch, tmp_path: Path) -> None:
    import sure_eval.evaluation.tasks.tts.pipeline as tts_pipeline
    from sure_eval.evaluation.core.types import EvaluationReport
    from sure_eval.evaluation.scripts.tts import run
    from sure_eval.evaluation.tasks.tts.types import TTSSample

    calls: dict[str, object] = {}

    def fake_executor(**kwargs):
        calls["kwargs"] = kwargs
        return EvaluationReport(
            task="TTS",
            language="zh",
            metric="multi",
            score=0.0,
            pipeline_id="tts.zh.multi.audio_metric_nodes",
            pipeline_trace=(),
        )

    monkeypatch.setattr(tts_pipeline, "evaluate_tts_samples", fake_executor)

    run(
        [
            TTSSample(
                prediction_audio="hyp.wav",
                reference_text="你好世界",
                reference_audio="ref.wav",
                language="zh",
                sample_id="utt1",
            )
        ],
        metrics=["TTS_CER", "SIM/WAVLM-LARGE", "DNSMOS"],
        output_dir=str(tmp_path / "tts_out"),
    )

    assert calls["kwargs"]["metrics"] == ("tts_cer", "sim/wavlm-large", "dnsmos")
    assert len(calls["kwargs"]["samples"]) == 1


def test_tts_script_run_injects_shared_transcriber_for_semantic_metric(monkeypatch, tmp_path: Path) -> None:
    import sure_eval.evaluation.tasks.tts.pipeline as tts_pipeline
    from sure_eval.evaluation.core.types import EvaluationReport
    from sure_eval.evaluation.scripts.tts import run
    from sure_eval.evaluation.tasks.tts.types import TTSSample

    calls: dict[str, object] = {}

    class FakeWhisperTranscriber:
        def __init__(self, *, device: str, cache_dir: object) -> None:
            self.device = device
            self.cache_dir = cache_dir

    def fake_executor(**kwargs):
        calls["kwargs"] = kwargs
        return EvaluationReport(
            task="TTS",
            language="en",
            metric="tts_wer",
            score=0.0,
            pipeline_id="tts.en.tts_wer.whisper_large_v3.whisper_norm.wenet_wer",
            pipeline_trace=(),
            details={"results": {"tts_wer": {"score": 0.0}}},
        )

    monkeypatch.setattr(tts_pipeline, "evaluate_tts_samples", fake_executor)
    monkeypatch.setattr(
        "sure_eval.evaluation.nodes.transcription.common.providers.WhisperLargeV3Transcriber",
        FakeWhisperTranscriber,
    )

    run(
        [
            TTSSample(
                prediction_audio="hyp1.wav",
                reference_text="hello",
                reference_audio="ref1.wav",
                language="en",
                sample_id="utt1",
            ),
            TTSSample(
                prediction_audio="hyp2.wav",
                reference_text="world",
                reference_audio="ref2.wav",
                language="en",
                sample_id="utt2",
            ),
        ],
        metrics=["tts_wer"],
        output_dir=str(tmp_path / "tts_out"),
    )

    transcribers = calls["kwargs"]["transcribers"]
    assert set(transcribers) == {"en"}
    assert isinstance(transcribers["en"], FakeWhisperTranscriber)
    assert transcribers["en"].device == "cuda"
    assert str(transcribers["en"].cache_dir).endswith(
        "src/sure_eval/evaluation/nodes/transcription/whisper_large_v3/checkpoints"
    )


def test_vc_script_run_calls_task_level_executor_with_normalized_metrics(monkeypatch, tmp_path: Path) -> None:
    import sure_eval.evaluation.tasks.vc.pipeline as vc_pipeline
    from sure_eval.evaluation.core.types import EvaluationReport
    from sure_eval.evaluation.scripts.vc import run
    from sure_eval.evaluation.tasks.vc.types import VCSample

    calls: dict[str, object] = {}

    def fake_executor(**kwargs):
        calls["kwargs"] = kwargs
        return EvaluationReport(
            task="VC",
            language="zh",
            metric="multi",
            score=0.0,
            pipeline_id="vc.zh.multi.audio_metric_nodes",
            pipeline_trace=(),
        )

    monkeypatch.setattr(vc_pipeline, "evaluate_vc_samples", fake_executor)

    run(
        [
            VCSample(
                converted_audio="converted.wav",
                reference_audio="target.wav",
                reference_text="你好世界",
                language="zh",
                sample_id="utt1",
            )
        ],
        metrics=["VC_CER", "SIM/ECAPA-TDNN", "UTMOS"],
        output_dir=str(tmp_path / "vc_out"),
    )

    assert calls["kwargs"]["metrics"] == ("vc_cer", "sim/ecapa-tdnn", "utmos")
    assert len(calls["kwargs"]["samples"]) == 1
