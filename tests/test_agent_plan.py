from __future__ import annotations

import json

from typer.testing import CliRunner

from sure_eval.cli import app
from sure_eval.evaluation.env_check import EnvCheckResult


def test_agent_plan_cli_outputs_route_and_env_contract() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["agent", "plan", "asr", "--language", "zh", "--metric", "cer", "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["schema"] == "sure.eval.agent_plan.v1"
    assert payload["task"] == "asr"
    assert payload["metrics"] == ["cer"]
    route = payload["selected_routes"][0]
    assert route["pipeline_id"] == "asr.zh.cer.wetext_norm_zh_itn_v1.wenet_cer_v1"
    assert [node["node_id"] for node in route["nodes"]] == [
        "normalization/wetext_norm",
        "scoring/wenet_cer",
    ]
    wetext_check = next(
        check for check in route["env_checks"] if check["node_id"] == "normalization/wetext_norm"
    )
    assert wetext_check["runtime"] == "node_local_project"
    assert "required_for_selected_route" in wetext_check


def test_agent_plan_accepts_task_option_for_non_positional_callers() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["agent", "plan", "--task", "tts", "--language", "zh", "--metrics", "tts_cer", "--json"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    route = payload["selected_routes"][0]
    assert route["pipeline_id"] == (
        "tts.zh.cer.funasr_loader_16k_mono_v1.paraformer_zh_v1."
        "punctuation_strip_norm_v1.wenet_cer_v1"
    )
    assert route["metric"] == "tts_cer"
    assert route["resolved_metric"] == "cer"
    assert "route_id" not in route
    assert "normalization/punctuation_strip_norm" in [
        node["node_id"] for node in route["nodes"]
    ]


def test_agent_plan_reports_blocking_setup_hints(monkeypatch) -> None:
    from sure_eval.evaluation import agent_plan

    def fake_check_node(self, node_id: str) -> EnvCheckResult:
        if node_id == "normalization/wetext_norm":
            return EnvCheckResult(
                name=node_id,
                node_id=node_id,
                runtime="node_local_project",
                required=True,
                status="failed",
                message=".venv is missing",
                fix="cd node && uv sync",
            )
        return EnvCheckResult(
            name=node_id,
            node_id=node_id,
            runtime="in_process",
            required=False,
            status="ok",
            message="in-process node",
        )

    monkeypatch.setattr(agent_plan.NodeEnvChecker, "check_node", fake_check_node)

    payload = agent_plan.build_agent_plan("asr", language="zh", metric="cer")

    assert payload["status"] == "blocked"
    assert payload["can_run_now"] is False
    assert payload["blocking_issues"] == [
        "cer:normalization/wetext_norm: .venv is missing"
    ]
    setup = payload["selected_routes"][0]["env_checks"][0]["setup"]
    assert setup["runtime"] == "uv"
    assert setup["packages"] == ["WeTextProcessing==1.2.0"]
    assert setup["command"].endswith("wetext_norm && uv sync")
