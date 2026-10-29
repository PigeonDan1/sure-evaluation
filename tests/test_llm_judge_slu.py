"""Offline node and CLI integration tests; never contact a real API."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock
import pytest
from typer.testing import CliRunner

from sure_eval.cli import app
from sure_eval.evaluation.cli_adapters import build_pipeline_spec, run_pipeline_spec
from sure_eval.evaluation.nodes.scoring.llm_judge import node
from sure_eval.evaluation.scripts import describe_pipeline, run_task


@pytest.fixture
def inputs(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '\n'.join(json.dumps(row) for row in [
            {"key": "a", "prompt": "What city is mentioned?", "context": "The speaker says Paris.",
             "choices": {"A": "Paris", "B": "London"}, "reference": "must not reach judge"},
            {"key": "b", "question": "What time is the meeting?", "context": "Meeting at nine."},
        ]) + '\n', encoding="utf-8",
    )
    hyp = tmp_path / "hyp.txt"
    hyp.write_text("b\tAt nine.\na\tA\n", encoding="utf-8")
    config = tmp_path / "judge.json"
    config.write_text(json.dumps({"base_url": "https://example.invalid/v1", "model": "test-judge"}),
                      encoding="utf-8")
    return prompts, hyp, config


def test_slu_judge_routes_and_default_accuracy():
    default = describe_pipeline("slu")
    judge = describe_pipeline("slu", metric="llm_judge")
    assert default.metric == "accuracy"
    assert default.node_ids == ("normalization/prompt_norm", "scoring/classify")
    assert judge.pipeline_id == "slu.any.llm_judge.llm_judge_v1"
    assert judge.required_roles == ("hyp", "prompt_jsonl")
    assert judge.node_ids == ("scoring/llm_judge",)


def test_node_aligns_answers_and_passes_only_explicit_context(inputs):
    prompts, hyp, _ = inputs
    seen = []

    def caller(config, messages):
        data = json.loads(messages[1]["content"])
        seen.append(data)
        return {"score": 4 if data["answer"] == "A" else 2, "reason": "test reason"}

    result = node.score_slu_judge(
        hyp_file=str(hyp), prompt_jsonl=str(prompts),
        config=node.JudgeConfig("https://example.invalid/v1", "fake"), caller=caller,
    )
    assert result.details["result"]["score"] == 3
    assert seen[0]["answer"] == "A"
    assert seen[0]["choices"]["A"] == "Paris"
    assert "reference" not in seen[0]
    assert result.details["input_files"] == {"hyp": str(hyp), "prompt_jsonl": str(prompts)}


@pytest.mark.parametrize("value", [True, 0, 6, 2.5, "5", float("nan")])
def test_answer_scores_are_validated(value):
    with pytest.raises(node.JudgeError):
        node.validate_answer_judgment({"score": value, "reason": "test"})


def test_key_mismatch_prevents_api_call(inputs, monkeypatch):
    prompts, hyp, _ = inputs
    hyp.write_text("different\tanswer\n", encoding="utf-8")
    monkeypatch.setattr(node, "call_judge", lambda *args: pytest.fail("must not call API"))
    with pytest.raises(ValueError, match="same non-empty key set"):
        node.score_slu_judge(hyp_file=str(hyp), prompt_jsonl=str(prompts),
                             config=node.JudgeConfig("https://example.invalid/v1", "fake"))


@pytest.mark.parametrize("rows", [
    [{"key": "a", "prompt": "question"}, {"key": "a", "prompt": "duplicate"}],
    [{"key": "a", "prompt": ""}],
    [{"key": "a", "prompt": "question", "context": 42}],
])
def test_invalid_questions_prevent_api_call(inputs, monkeypatch, rows):
    prompts, hyp, _ = inputs
    prompts.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(node, "call_judge", lambda *args: pytest.fail("must not call API"))
    with pytest.raises(ValueError):
        node.score_slu_judge(hyp_file=str(hyp), prompt_jsonl=str(prompts),
                            config=node.JudgeConfig("https://example.invalid/v1", "fake"))


def test_cli_routes_describe_run_and_standard_outputs(inputs, tmp_path, monkeypatch):
    prompts, hyp, config = inputs
    runner = CliRunner()
    seen = []

    def caller(config, messages):
        seen.append(config.model)
        return {"score": 4, "reason": "answers the question"}

    monkeypatch.setattr(node, "call_judge", caller)
    routes = runner.invoke(app, ["metric", "routes", "slu", "--metric", "llm_judge", "--json"])
    assert routes.exit_code == 0, routes.output
    assert json.loads(routes.stdout)["count"] == 1
    pipeline = tmp_path / "pipeline.json"
    describe = runner.invoke(app, ["metric", "describe", "slu", "--metric", "llm_judge",
                                   "--output", str(pipeline), "--json"])
    assert describe.exit_code == 0, describe.output
    spec = json.loads(pipeline.read_text(encoding="utf-8"))
    spec["run_args"]["llm_config"] = str(config)
    pipeline.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "output"
    run = runner.invoke(app, ["metric", "run", "--pipeline", str(pipeline),
                             "--hyp-file", str(hyp), "--prompt-jsonl", str(prompts),
                             "--output-dir", str(output), "--json"])
    assert run.exit_code == 0, run.output
    assert seen == ["test-judge", "test-judge"]
    summary = json.loads(run.stdout)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    description = json.loads((output / "pipeline_description.json").read_text(encoding="utf-8"))
    assert summary["score"] == report["score"] == 4
    assert report["pipeline_id"] == description["pipeline_id"]
    assert report["computation_node_ids"] == ["scoring/llm_judge"]
    assert report["input_files"] == {"hyp": str(hyp), "prompt_jsonl": str(prompts)}


def test_python_api_and_pipeline_config_parameter(inputs, tmp_path, monkeypatch):
    prompts, hyp, config = inputs
    monkeypatch.setattr(node, "call_judge", lambda *args: {"score": 5, "reason": "good"})
    report = run_task("slu", metric="llm_judge", hyp_file=str(hyp), prompt_jsonl=str(prompts),
                      llm_config=str(config), output_dir=str(tmp_path / "api"))
    assert report.score == 5
    pipeline = build_pipeline_spec("slu", metric="llm_judge")
    pipeline["run_args"]["llm_config"] = str(config)
    summary = run_pipeline_spec(pipeline, hyp_file=str(hyp), prompt_jsonl=str(prompts),
                                output_dir=str(tmp_path / "adapter"))
    assert summary["score"] == 5


def test_pipeline_validates_llm_config_before_api_call(inputs, monkeypatch):
    from sure_eval.evaluation.tasks.slu.pipeline import evaluate_slu_files

    prompts, hyp, config = inputs
    config.write_text(json.dumps({"base_url": "http://example.invalid/v1", "model": "test"}),
                      encoding="utf-8")
    monkeypatch.setattr(node, "call_judge", lambda *args: pytest.fail("must not call API"))
    with pytest.raises(ValueError, match="HTTPS"):
        evaluate_slu_files(hyp_file=str(hyp), prompt_jsonl=str(prompts),
                           metric="llm_judge", llm_config=str(config))


def test_cli_passes_aligned_references_to_judge(inputs, tmp_path, monkeypatch):
    prompts, hyp, config = inputs
    ref = tmp_path / "ref.txt"
    ref.write_text("b\tAt nine.\na\tParis\n", encoding="utf-8")
    spec = build_pipeline_spec("slu", metric="llm_judge")
    spec["run_args"]["llm_config"] = str(config)
    assert spec["executor"].endswith(".evaluate_slu_files")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(json.dumps(spec), encoding="utf-8")
    seen = []

    def caller(config, messages):
        seen.append(json.loads(messages[1]["content"]))
        return {"score": 5, "reason": "matches reference"}

    monkeypatch.setattr(node, "call_judge", caller)
    output = tmp_path / "with-ref"
    result = CliRunner().invoke(app, [
        "metric", "run", "--pipeline", str(pipeline), "--hyp-file", str(hyp),
        "--ref-file", str(ref), "--prompt-jsonl", str(prompts), "--output-dir", str(output),
    ])
    assert result.exit_code == 0, result.output
    assert [row["reference_answer"] for row in seen] == ["Paris", "At nine."]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["input_files"]["ref"] == str(ref)
    assert report["pipeline_trace"][0]["details"]["input_files"]["ref"] == str(ref)


def test_reference_key_mismatch_prevents_api_call(inputs, tmp_path, monkeypatch):
    prompts, hyp, _ = inputs
    ref = tmp_path / "ref.txt"
    ref.write_text("different\tanswer\n", encoding="utf-8")
    monkeypatch.setattr(node, "call_judge", lambda *args: pytest.fail("must not call API"))
    with pytest.raises(ValueError, match="ref and hyp"):
        node.score_slu_judge(
            hyp_file=str(hyp), prompt_jsonl=str(prompts), ref_file=str(ref),
            config=node.JudgeConfig("https://example.invalid/v1", "fake"),
        )


def test_missing_config_and_accuracy_reference_checks(inputs, tmp_path):
    prompts, hyp, config = inputs
    with pytest.raises(ValueError, match="llm_config"):
        run_task("slu", metric="llm_judge", hyp_file=str(hyp), prompt_jsonl=str(prompts),
                  output_dir=str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="ref_file"):
        run_task("slu", hyp_file=str(hyp), prompt_jsonl=str(prompts), output_dir=str(tmp_path / "acc"))
    with pytest.raises(ValueError, match="only accepted"):
        pipeline = build_pipeline_spec("slu")
        pipeline["run_args"]["llm_config"] = str(config)
        run_pipeline_spec(pipeline,
                          hyp_file=str(hyp), prompt_jsonl=str(prompts), output_dir=str(tmp_path / "bad"))


def test_failed_batch_preserves_evidence_and_cli_fails(inputs, tmp_path, monkeypatch):
    prompts, hyp, config = inputs

    def caller(config, messages):
        if json.loads(messages[1]["content"])["answer"] == "A":
            return {"score": 4, "reason": "good"}
        raise node.JudgeError("http_error_429")

    monkeypatch.setattr(node, "call_judge", caller)
    pipeline = tmp_path / "pipeline.json"
    spec = build_pipeline_spec("slu", metric="llm_judge")
    spec["run_args"]["llm_config"] = str(config)
    pipeline.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "failed"
    result = CliRunner().invoke(app, ["metric", "run", "--pipeline", str(pipeline),
                                      "--hyp-file", str(hyp), "--prompt-jsonl", str(prompts),
                                      "--output-dir", str(output), "--json"])
    assert result.exit_code == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["score"] is None
    assert report["details"]["scoring_result"]["num_failed"] == 1
    assert report["pipeline_trace"][0]["details"]["sample_results"][1]["error_code"] == "http_error_429"


@pytest.mark.parametrize("credential_source", ["environment", "config"])
def test_api_protocol_and_returned_model(monkeypatch, credential_source):
    opener = MagicMock()
    opener.open.return_value.__enter__.return_value.read.return_value = json.dumps({
        "model": "actual-model", "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"score": 4, "reason": "good"})}}],
    }).encode()
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "fake-test-key")
    monkeypatch.setattr(node.urllib.request, "build_opener", lambda *args: opener)
    config = node.JudgeConfig(
        "https://example.invalid/v1", "test-model",
        api_key="local-test-key" if credential_source == "config" else None,
    )
    result = node.call_judge(config, [{"role": "user", "content": "test"}])
    request = opener.open.call_args.args[0]
    assert request.full_url == "https://example.invalid/v1/chat/completions"
    assert json.loads(request.data)["model"] == "test-model"
    expected_key = "local-test-key" if credential_source == "config" else "fake-test-key"
    assert request.get_header("Authorization") == f"Bearer {expected_key}"
    assert result["response_model"] == "actual-model"
    assert "fake-test-key" not in json.dumps(result)


@pytest.mark.parametrize("failure", ["http", "truncated"])
def test_api_errors_are_sanitized_and_truncation_rejected(monkeypatch, failure):
    opener = MagicMock()
    if failure == "http":
        opener.open.side_effect = urllib.error.HTTPError(
            "https://example.invalid", 401, "sensitive server message", {}, None,
        )
        expected = "^http_error_401$"
    else:
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"finish_reason": "length", "message": {
                "content": '{"score": 5, "reason": "good"}'}}],
        }).encode()
        expected = "incomplete_or_refused_response"
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "fake-test-key")
    monkeypatch.setattr(node.urllib.request, "build_opener", lambda *args: opener)
    with pytest.raises(node.JudgeError, match=expected):
        node.call_judge(node.JudgeConfig("https://example.invalid/v1", "test"), [])


@pytest.mark.parametrize("credential_source", ["environment", "config"])
def test_standalone_cli_runs_slu(inputs, tmp_path, monkeypatch, credential_source):
    prompts, hyp, config = inputs
    output = tmp_path / "node-result.json"
    monkeypatch.setenv("LLM_JUDGE_API_KEY", "fake-test-key")
    if credential_source == "config":
        payload = json.loads(config.read_text(encoding="utf-8"))
        payload["api_key"] = "local-test-key"
        config.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.delenv("LLM_JUDGE_API_KEY")
    monkeypatch.setattr(node, "call_judge", lambda *args: {"score": 4, "reason": "good"})
    code = node.main([
        "--prompt-jsonl", str(prompts), "--hyp-file", str(hyp),
        "--config", str(config), "--output", str(output),
    ])
    assert code == 0
    content = output.read_text(encoding="utf-8")
    assert json.loads(content)["details"]["result"]["score"] == 4
    assert "fake-test-key" not in content
    assert "local-test-key" not in content
    assert "api_key" not in json.loads(content)["details"]["llm_config"]
    assert "local-test-key" not in repr(node.JudgeConfig("https://example.invalid/v1", "test", api_key="local-test-key"))
