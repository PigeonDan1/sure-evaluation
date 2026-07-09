from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from sure_eval.cli import app


def test_node_env_checker_treats_in_process_node_as_ok_without_venv() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    checker = NodeEnvChecker()
    result = checker.check_node("scoring/wenet_wer")

    assert result.node_id == "scoring/wenet_wer"
    assert result.status == "ok"
    assert result.runtime == "in_process"
    assert result.required is False


def test_heavy_audio_nodes_require_node_local_envs() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    checker = NodeEnvChecker()
    node_ids = {
        "scoring/wavlm_large_sim",
        "scoring/ecapa_tdnn_sim",
        "scoring/eres2net_sim",
        "scoring/dnsmos",
        "scoring/wv_mos",
        "scoring/utmos",
        "transcription/paraformer_zh",
        "transcription/whisper_large_v3",
    }

    results = {node_id: checker.check_node(node_id) for node_id in node_ids}

    assert {result.runtime for result in results.values()} == {"node_local_project"}
    assert {result.required for result in results.values()} == {True}
    for node_id, result in results.items():
        assert result.node_id == node_id
        assert "checkpoint_path" in result.details


def test_node_env_checker_accepts_default_bleurt_checkpoint_without_env_var(monkeypatch) -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    monkeypatch.delenv("BLEURT_20_CHECKPOINT", raising=False)

    checker = NodeEnvChecker()
    result = checker.check_node("scoring/bleurt_20")

    assert result.node_id == "scoring/bleurt_20"
    assert result.runtime == "node_local_project"
    assert result.required is True
    assert result.status == "ok"


def test_node_env_checker_accepts_default_xcomet_checkpoint_without_env_var(monkeypatch) -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    monkeypatch.delenv("XCOMET_XL_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("XCOMET_XL_CHECKPOINT_DIR", raising=False)

    checker = NodeEnvChecker()
    result = checker.check_node("scoring/xcomet_xl")

    assert result.node_id == "scoring/xcomet_xl"
    assert result.runtime == "node_local_project"
    assert result.required is True
    assert result.status == "ok"


def test_transcription_node_local_envs_are_checked() -> None:
    from sure_eval.evaluation.env_check import NodeEnvChecker

    checker = NodeEnvChecker()

    paraformer = checker.check_node("transcription/paraformer_zh")
    whisper = checker.check_node("transcription/whisper_large_v3")

    assert paraformer.runtime == "node_local_project"
    assert whisper.runtime == "node_local_project"
    assert paraformer.status == "ok"
    assert whisper.status == "ok"
    assert "model.pt" in paraformer.details["checkpoint_path"]
    assert "model.safetensors" in whisper.details["checkpoint_path"]


def test_audio_runtime_uses_node_local_transcription_subprocesses() -> None:
    from sure_eval.evaluation.audio_runtime import build_tts_runtime
    from sure_eval.evaluation.nodes.transcription.common.providers import NodeLocalTranscriber

    zh_runtime = build_tts_runtime(metrics=("tts_cer",), language="zh", device="cpu")
    en_runtime = build_tts_runtime(metrics=("tts_wer",), language="en", device="cpu")

    assert isinstance(zh_runtime["transcribers"]["zh"], NodeLocalTranscriber)
    assert zh_runtime["transcribers"]["zh"].node_id == "transcription/paraformer_zh"
    assert isinstance(en_runtime["transcribers"]["en"], NodeLocalTranscriber)
    assert en_runtime["transcribers"]["en"].node_id == "transcription/whisper_large_v3"


def test_audio_runtime_uses_node_local_scoring_subprocesses() -> None:
    from sure_eval.evaluation.audio_runtime import build_tts_runtime
    from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalMOSProvider, NodeLocalSpeakerProvider

    runtime = build_tts_runtime(
        metrics=("sim/wavlm-large", "sim/ecapa-tdnn", "sim/eres2net", "dnsmos", "wv-mos", "utmos"),
        language="en",
        device="cpu",
    )

    assert isinstance(runtime["speaker_providers"]["wavlm-large"], NodeLocalSpeakerProvider)
    assert runtime["speaker_providers"]["wavlm-large"].node_id == "scoring/wavlm_large_sim"
    assert isinstance(runtime["speaker_providers"]["ecapa-tdnn"], NodeLocalSpeakerProvider)
    assert runtime["speaker_providers"]["ecapa-tdnn"].node_id == "scoring/ecapa_tdnn_sim"
    assert isinstance(runtime["speaker_providers"]["eres2net"], NodeLocalSpeakerProvider)
    assert runtime["speaker_providers"]["eres2net"].node_id == "scoring/eres2net_sim"
    assert isinstance(runtime["mos_providers"]["dnsmos"], NodeLocalMOSProvider)
    assert runtime["mos_providers"]["dnsmos"].node_id == "scoring/dnsmos"
    assert isinstance(runtime["mos_providers"]["wv-mos"], NodeLocalMOSProvider)
    assert runtime["mos_providers"]["wv-mos"].node_id == "scoring/wv_mos"
    assert isinstance(runtime["mos_providers"]["utmos"], NodeLocalMOSProvider)
    assert runtime["mos_providers"]["utmos"].node_id == "scoring/utmos"


def test_cache_dir_uses_sure_eval_cache_dir(monkeypatch, tmp_path: Path) -> None:
    from sure_eval.evaluation.cache import get_cache_dir

    root = tmp_path / "cache"
    monkeypatch.setenv("SURE_EVAL_CACHE_DIR", str(root))

    cache_dir = get_cache_dir("tts-metrics")

    assert cache_dir == root / "tts-metrics"
    assert cache_dir.exists()


def test_doctor_treats_optional_node_failures_as_warning(monkeypatch) -> None:
    from sure_eval.evaluation import env_check

    def _fake_check_node(self, node_id: str):
        return env_check.EnvCheckResult(
            name=node_id,
            node_id=node_id,
            runtime="node_local_project",
            required=True,
            status="failed",
            message=".venv is missing",
            fix="uv sync",
        )

    monkeypatch.setattr(env_check.NodeEnvChecker, "check_node", _fake_check_node)

    payload = env_check.doctor_payload()

    assert payload["status"] in {"ok", "warning"}
    node_checks = [item for item in payload["checks"] if item.get("node_id")]
    assert node_checks
    assert {item["status"] for item in node_checks} == {"warning"}
    assert all("optional node is not prepared" in item["message"] for item in node_checks)


def test_env_setup_dry_run_reads_node_env_metadata() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["env", "setup", "--node", "scoring/dnsmos", "--dry-run", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    action = payload["actions"][0]
    assert payload["status"] == "planned"
    assert action["node_id"] == "scoring/dnsmos"
    assert action["python"] == "3.11"
    assert action["project"] == "pyproject.toml"
    assert action["node_env"].endswith("node_env.yaml")
    assert action["downloads"][0]["env"] == "DNSMOS_CHECKPOINT"


def test_env_setup_dry_run_resolves_task_metrics_and_group() -> None:
    runner = CliRunner()

    task_result = runner.invoke(
        app,
        [
            "env",
            "setup",
            "--task",
            "tts",
            "--language",
            "zh",
            "--metrics",
            "tts_cer,dnsmos",
            "--dry-run",
            "--json",
        ],
    )
    group_result = runner.invoke(
        app,
        ["env", "setup", "--group", "tts-vc-mos", "--dry-run", "--json"],
    )

    assert task_result.exit_code == 0, task_result.stdout
    task_payload = json.loads(task_result.stdout)
    assert [item["node_id"] for item in task_payload["actions"]] == [
        "transcription/paraformer_zh",
        "scoring/dnsmos",
    ]

    assert group_result.exit_code == 0, group_result.stdout
    group_payload = json.loads(group_result.stdout)
    assert {item["node_id"] for item in group_payload["actions"]} == {
        "scoring/dnsmos",
        "scoring/wv_mos",
        "scoring/utmos",
    }


def test_env_download_dry_run_reports_declared_assets() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["env", "download", "--node", "scoring/dnsmos", "--dry-run", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "planned"
    assert payload["dry_run"] is True
    plan = payload["downloads"][0]
    assert plan["node_id"] == "scoring/dnsmos"
    assert plan["assets"][0]["id"] == "DNSMOS/model_v8.onnx"
    assert plan["assets"][0]["provider"] == "manual"
    assert plan["assets"][0]["env_override"] == "DNSMOS_CHECKPOINT"


def test_node_local_transcriber_uses_repo_src_pythonpath(monkeypatch, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.transcription.common.providers import NodeLocalTranscriber

    repo_root = tmp_path / "repo"
    node_dir = repo_root / "src" / "sure_eval" / "evaluation" / "nodes" / "transcription" / "paraformer_zh"
    python_bin = node_dir / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    python_bin.chmod(0o755)
    captured = {}

    class _Completed:
        returncode = 0
        stdout = '{"transcript": "ok"}'
        stderr = ""

    def _fake_run(command, cwd, env, **kwargs):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["pythonpath"] = env["PYTHONPATH"]
        return _Completed()

    monkeypatch.setattr("subprocess.run", _fake_run)

    transcript = NodeLocalTranscriber(
        node_id="transcription/paraformer_zh",
        node_dir=node_dir,
        device="cpu",
    ).transcribe("audio.wav", language="zh")

    assert transcript == "ok"
    assert captured["cwd"] == repo_root
    assert str(repo_root / "src") in captured["pythonpath"].split(":")
    assert captured["command"][0] == str(python_bin)


def test_node_local_transcriber_falls_back_when_venv_python_symlink_is_broken(
    monkeypatch, tmp_path: Path
) -> None:
    import sys

    from sure_eval.evaluation.nodes.transcription.common.providers import NodeLocalTranscriber

    repo_root = tmp_path / "repo"
    node_dir = repo_root / "src" / "sure_eval" / "evaluation" / "nodes" / "transcription" / "whisper_large_v3"
    python_bin = node_dir / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.symlink_to("/missing/container/python3.11")
    site_packages = node_dir / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    site_packages.mkdir(parents=True)
    captured = {}

    class _Completed:
        returncode = 0
        stdout = '{"transcript": "ok"}'
        stderr = ""

    def _fake_run(command, cwd, env, **kwargs):
        captured["command"] = command
        captured["pythonpath"] = env["PYTHONPATH"]
        return _Completed()

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setenv("PYTHONPATH", "/external/should-not-leak")

    transcript = NodeLocalTranscriber(
        node_id="transcription/whisper_large_v3",
        node_dir=node_dir,
        device="cpu",
    ).transcribe("audio.wav", language="en")

    assert transcript == "ok"
    assert captured["command"][:2] == [sys.executable, "-S"]
    assert str(site_packages) in captured["pythonpath"].split(":")
    assert "/external/should-not-leak" not in captured["pythonpath"].split(":")


def test_node_local_scoring_provider_uses_repo_root_cwd(monkeypatch, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalMOSProvider

    repo_root = tmp_path / "repo"
    node_dir = repo_root / "src" / "sure_eval" / "evaluation" / "nodes" / "scoring" / "dnsmos"
    python_bin = node_dir / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    python_bin.chmod(0o755)
    captured = {}

    class _Completed:
        returncode = 0
        stdout = '{"result": {"OVRL": 3.2, "score": 3.2}}'
        stderr = ""

    def _fake_run(command, cwd, env, **kwargs):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["pythonpath"] = env["PYTHONPATH"]
        return _Completed()

    monkeypatch.setattr("subprocess.run", _fake_run)

    row = NodeLocalMOSProvider(
        node_id="scoring/dnsmos",
        node_dir=node_dir,
        device="cpu",
    )("relative.wav")

    assert row["OVRL"] == 3.2
    assert captured["cwd"] == repo_root
    assert str(repo_root / "src") in captured["pythonpath"].split(":")
    assert captured["command"][0] == str(python_bin)
    assert "--prediction-audio" in captured["command"]


def test_node_local_scoring_provider_falls_back_when_venv_python_symlink_is_broken(
    monkeypatch, tmp_path: Path
) -> None:
    import sys

    from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalMOSProvider

    repo_root = tmp_path / "repo"
    node_dir = repo_root / "src" / "sure_eval" / "evaluation" / "nodes" / "scoring" / "dnsmos"
    python_bin = node_dir / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.symlink_to("/missing/container/python3.11")
    site_packages = node_dir / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    site_packages.mkdir(parents=True)
    captured = {}

    class _Completed:
        returncode = 0
        stdout = '{"result": {"OVRL": 3.2, "score": 3.2}}'
        stderr = ""

    def _fake_run(command, cwd, env, **kwargs):
        captured["command"] = command
        captured["pythonpath"] = env["PYTHONPATH"]
        return _Completed()

    monkeypatch.setattr("subprocess.run", _fake_run)
    monkeypatch.setenv("PYTHONPATH", "/external/should-not-leak")

    row = NodeLocalMOSProvider(
        node_id="scoring/dnsmos",
        node_dir=node_dir,
        device="cpu",
    )("relative.wav")

    assert row["OVRL"] == 3.2
    assert captured["command"][:2] == [sys.executable, "-S"]
    assert str(site_packages) in captured["pythonpath"].split(":")
    assert "/external/should-not-leak" not in captured["pythonpath"].split(":")


def test_metric_run_validate_env_passes_for_provider_injected_audio_node(monkeypatch, tmp_path: Path) -> None:
    import sure_eval.evaluation.cli as metric_cli

    runner = CliRunner()
    pipeline_path = tmp_path / "pipeline.json"
    samples_jsonl = tmp_path / "samples.jsonl"
    (tmp_path / "generated.wav").write_bytes(b"fake")
    samples_jsonl.write_text(
        json.dumps(
            {
                "sample_id": "utt1",
                "prediction_audio": "generated.wav",
                "reference_text": "你好世界",
                "language": "zh",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
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
            "tts_cer",
            "--output",
            str(pipeline_path),
            "--json",
        ],
    )
    assert describe_result.exit_code == 0, describe_result.stdout

    def _fake_run_pipeline_spec(payload, **kwargs):
        return {
            "status": "ok",
            "task": "TTS",
            "metric": "tts_cer",
            "score": 0.0,
            "pipeline_id": payload["pipeline_id"],
            "report_path": str(tmp_path / "out" / "report.json"),
            "pipeline_description_path": str(tmp_path / "out" / "pipeline_description.json"),
            "environment_note": "",
            "node_config_paths": [],
        }

    monkeypatch.setattr(metric_cli, "run_pipeline_spec", _fake_run_pipeline_spec)

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--samples-jsonl",
            str(samples_jsonl),
            "--output-dir",
            str(tmp_path / "out"),
            "--validate-env",
            "--json",
        ],
    )

    assert run_result.exit_code == 0, run_result.stdout
    payload = json.loads(run_result.stdout)
    assert payload["status"] == "ok"


def test_sure_eval_doctor_outputs_json_status() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code in {0, 1}
    payload = json.loads(result.stdout)
    assert payload["status"] in {"ok", "warning", "failed"}
    assert any(item["name"] == "python" for item in payload["checks"])
    assert any(item["name"] == "uv" for item in payload["checks"])
