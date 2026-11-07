"""Failure matrix and report compatibility checks; no real credentials or API."""

from __future__ import annotations

import json
import ssl
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from sure_eval.cli import app
from sure_eval.evaluation.cli_adapters import build_pipeline_spec
from sure_eval.evaluation.nodes.scoring.llm_judge import node
from sure_eval.evaluation.tasks.slu.pipeline import evaluate_slu_files


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("This audit must never access the network")

    monkeypatch.setattr(node.urllib.request.OpenerDirector, "open", forbidden)
    monkeypatch.delenv("LLM_JUDGE_API_KEY", raising=False)


@pytest.fixture
def case(tmp_path):
    paths = {name: tmp_path / name for name in (
        "questions.jsonl", "hyp.txt", "ref.txt", "config.json", "pipeline.json",
    )}
    paths["questions.jsonl"].write_text(json.dumps({
        "key": "a", "prompt": "Which city?", "choices": {"A": "Paris", "B": "London"},
    }) + "\n", encoding="utf-8")
    paths["hyp.txt"].write_text("a\tA\n", encoding="utf-8")
    paths["ref.txt"].write_text("a\tA\n", encoding="utf-8")
    paths["config.json"].write_text(json.dumps({
        "base_url": "https://example.invalid/v1", "model": "fake",
        "api_key": "audit-fake-secret",
    }), encoding="utf-8")
    spec = build_pipeline_spec("slu", metric="llm_judge")
    spec["run_args"]["llm_config"] = str(paths["config.json"])
    paths["pipeline.json"].write_text(json.dumps(spec), encoding="utf-8")
    return paths


def run_cli(case, output, *, json_output=True, extra=()):
    args = ["metric", "run", "--pipeline", str(case["pipeline.json"]),
            "--hyp-file", str(case["hyp.txt"]), "--prompt-jsonl", str(case["questions.jsonl"]),
            "--ref-file", str(case["ref.txt"]), "--output-dir", str(output), *extra]
    if json_output:
        args.append("--json")
    return CliRunner().invoke(app, args)


def record_failure(name, result):
    assert result.exit_code == 1, result.output
    body = json.loads(result.stdout)
    assert set(body) == {"status", "message"}
    assert body["status"] == "error" and body["message"]
    assert "Traceback" not in result.output
    assert "audit-fake-secret" not in result.output
    print(json.dumps({"case": name, "exit_code": result.exit_code, "cli": body}))
    return body["message"]


BAD_CONFIG = [
    ("base_url", "", "base_url"), ("base_url", None, "base_url"),
    ("base_url", 1, "base_url"), ("base_url", "ftp://example.invalid", "HTTPS"),
    ("base_url", "http://example.invalid", "HTTPS"),
    ("base_url", "https:///path", "HTTPS"),
    ("base_url", "https://user:password@example.invalid", "credentials"),
    ("base_url", "https://example.invalid?token=x", "query"),
    ("base_url", "https://example.invalid#fragment", "fragment"),
    ("model", "", "model"), ("model", None, "model"), ("model", 1, "model"),
    ("api_key_env", "", "api_key_env"), ("api_key_env", None, "api_key_env"),
    ("api_key", 1, "api_key"), ("api_key", [], "api_key"),
    ("timeout", 0, "positive"), ("timeout", -1, "positive"),
    ("timeout", True, "finite"), ("timeout", "60", "finite"),
    ("timeout", float("nan"), "finite"), ("timeout", float("inf"), "finite"),
    ("temperature", -1, "non-negative"), ("temperature", True, "finite"),
    ("temperature", "0", "finite"), ("temperature", float("inf"), "finite"),
    ("max_tokens", 0, "positive integer"), ("max_tokens", -1, "positive integer"),
    ("max_tokens", True, "positive integer"), ("max_tokens", 1.5, "positive integer"),
    ("instruction", "", "instruction"), ("instruction", 1, "instruction"),
]


@pytest.mark.parametrize("field,value,message", BAD_CONFIG)
def test_bad_config_has_cli_error_before_network(case, tmp_path, field, value, message):
    config = json.loads(case["config.json"].read_text(encoding="utf-8"))
    config[field] = value
    case["config.json"].write_text(json.dumps(config), encoding="utf-8")
    result = run_cli(case, tmp_path / "out")
    assert message in record_failure(f"config:{field}={value!r}", result)
    assert not (tmp_path / "out" / "report.json").exists()


@pytest.mark.parametrize("text,message", [
    ("{", "Expecting"), ("[]", "JSON object"), ("null", "JSON object"),
    ('{"model":"fake"}', "base_url"),
    ('{"base_url":"https://example.invalid","model":"fake","typo":1}', "typo"),
])
def test_bad_config_file_structure(case, tmp_path, text, message):
    case["config.json"].write_text(text, encoding="utf-8")
    assert message in record_failure("config-file", run_cli(case, tmp_path / "out"))


@pytest.mark.parametrize("filename", ["config.json", "hyp.txt", "questions.jsonl", "ref.txt"])
def test_missing_files_have_cli_errors(case, tmp_path, filename):
    case[filename].unlink()
    message = record_failure("missing:" + filename, run_cli(case, tmp_path / "out"))
    assert filename in message


BAD_QUESTIONS = [
    ("", "same non-empty key set"), ("{\n", "Expecting"), ("[]\n", "object"),
    ('{"key":"a"}\n', "prompt/question"),
    ('{"key":"a","prompt":42}\n', "prompt/question"),
    ('{"key":"a","prompt":" "}\n', "prompt/question"),
    ('{"prompt":"q"}\n', "empty or duplicate key"),
    ('{"key":1,"prompt":"q"}\n', "empty or duplicate key"),
    ('{"key":" ","prompt":"q"}\n', "empty or duplicate key"),
    ('{"key":"a","prompt":"q"}\n' * 2, "empty or duplicate key"),
    ('{"key":"a","prompt":"q","context":1}\n', "context must be text"),
    ('{"key":"a","prompt":"q","choices":"A"}\n', "choices must be"),
    ('{"key":"b","prompt":"q"}\n', "same non-empty key set"),
]


@pytest.mark.parametrize("text,message", BAD_QUESTIONS)
def test_bad_questions_have_cli_errors(case, tmp_path, text, message):
    case["questions.jsonl"].write_text(text, encoding="utf-8")
    assert message in record_failure("question-input", run_cli(case, tmp_path / "out"))


@pytest.mark.parametrize("filename", ["hyp.txt", "ref.txt"])
@pytest.mark.parametrize("text,message", [
    ("", "no samples"), ("a answer\n", "key<TAB>text"),
    ("\tanswer\n", "empty or duplicate key"),
    ("a\tanswer\na\tduplicate\n", "empty or duplicate key"),
    ("other\tanswer\n", "same non-empty key set"),
])
def test_bad_answer_files_have_cli_errors(case, tmp_path, filename, text, message):
    case[filename].write_text(text, encoding="utf-8")
    assert message in record_failure("answer-input:" + filename, run_cli(case, tmp_path / "out"))


def test_empty_reference_rejected_but_empty_hypothesis_preserved(case, monkeypatch):
    case["ref.txt"].write_text("a\t\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty text"):
        evaluate_slu_files(
            str(case["ref.txt"]), str(case["hyp.txt"]), prompt_jsonl=str(case["questions.jsonl"]),
            metric="llm_judge", llm_config=str(case["config.json"]),
        )
    case["ref.txt"].write_text("a\tA\n", encoding="utf-8")
    case["hyp.txt"].write_text("a\t\n", encoding="utf-8")
    seen = []

    def judge(config, messages):
        seen.append(json.loads(messages[1]["content"])["answer"])
        return {"score": 1, "reason": "empty answer"}

    monkeypatch.setattr(node, "call_judge", judge)
    report = evaluate_slu_files(
        str(case["ref.txt"]), str(case["hyp.txt"]), prompt_jsonl=str(case["questions.jsonl"]),
        metric="llm_judge", llm_config=str(case["config.json"]),
    )
    assert report.score == 1 and seen == [""]


API_BODIES = [
    (None, "invalid_api_response"), ([], "invalid_api_response"),
    ({}, "invalid_api_response"), ({"choices": []}, "invalid_api_response"),
    ({"choices": [None]}, "invalid_api_response"),
    ({"choices": [{"finish_reason": "stop", "message": None}]}, "invalid_api_response"),
]


def completion(content, *, finish="stop", refusal=None):
    return {"choices": [{"finish_reason": finish, "message": {
        "content": content, "refusal": refusal,
    }}]}


API_BODIES += [
    (completion('{}', finish="length"), "incomplete_or_refused_response"),
    (completion('{}', finish="content_filter"), "incomplete_or_refused_response"),
    (completion('{}', refusal="refused"), "incomplete_or_refused_response"),
    (completion(None), "invalid_response_content"),
    (completion([]), "invalid_response_content"),
    (completion("not JSON"), "invalid_api_response"),
    (completion("```json\n{}\n```"), "invalid_api_response"),
    (completion("[]"), "invalid_judgment_fields"),
    (completion("null"), "invalid_judgment_fields"),
    (completion('{}'), "invalid_judgment_fields"),
    (completion('{"score":4}'), "invalid_judgment_fields"),
    (completion('{"score":4,"reason":"ok","extra":1}'), "invalid_judgment_fields"),
    *[(completion(json.dumps({"score": score, "reason": "ok"})), "invalid_judgment_score")
      for score in [True, 0, 6, 2.5, "5", None]],
    *[(completion(json.dumps({"score": 4, "reason": reason})), "invalid_judgment_reason")
      for reason in ["", "  ", None, 1]],
]


@pytest.mark.parametrize("body,expected", API_BODIES)
def test_malformed_api_response_preserves_report_and_fails_cli(case, tmp_path, monkeypatch, body, expected):
    opener = MagicMock()
    opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(body).encode()
    monkeypatch.setattr(node.urllib.request, "build_opener", lambda *args: opener)
    out = tmp_path / "out"
    result = run_cli(case, out)
    assert "judge_batch_failed" in record_failure("api-response:" + expected, result)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["score"] is None
    sample = report["pipeline_trace"][0]["details"]["sample_results"][0]
    assert sample == {"key": "a", "status": "failed", "error_code": expected}
    assert "audit-fake-secret" not in (out / "report.json").read_text(encoding="utf-8")
    print(json.dumps({"sample_error": sample}))


@pytest.mark.parametrize("code", [301, 302, 400, 401, 403, 404, 408, 429, 500, 503])
def test_http_error_codes_are_preserved_without_response_body(case, tmp_path, monkeypatch, code):
    opener = MagicMock()
    opener.open.side_effect = urllib.error.HTTPError(
        "https://example.invalid", code, "server secret audit-fake-secret", {}, None,
    )
    monkeypatch.setattr(node.urllib.request, "build_opener", lambda *args: opener)
    out = tmp_path / "out"
    record_failure(f"http:{code}", run_cli(case, out))
    content = (out / "report.json").read_text(encoding="utf-8")
    sample = json.loads(content)["pipeline_trace"][0]["details"]["sample_results"][0]
    assert sample["error_code"] == f"http_error_{code}"
    assert "server secret" not in content and "audit-fake-secret" not in content
    print(json.dumps({"sample_error": sample}))


@pytest.mark.parametrize("failure", [
    TimeoutError("private"), urllib.error.URLError("private"),
    ssl.SSLError("private"), ConnectionResetError("private"),
    b"not-json", b"\xff",
])
def test_transport_and_invalid_body_failures(case, tmp_path, monkeypatch, failure):
    opener = MagicMock()
    if isinstance(failure, bytes):
        opener.open.return_value.__enter__.return_value.read.return_value = failure
    else:
        opener.open.side_effect = failure
    monkeypatch.setattr(node.urllib.request, "build_opener", lambda *args: opener)
    out = tmp_path / "out"
    record_failure("transport-or-body", run_cli(case, out))
    content = (out / "report.json").read_text(encoding="utf-8")
    sample = json.loads(content)["pipeline_trace"][0]["details"]["sample_results"][0]
    assert sample["error_code"] == "api_transport_or_json_error"
    assert "private" not in content
    print(json.dumps({"sample_error": sample}))


def test_missing_key_records_failure_without_network(case, tmp_path):
    config = json.loads(case["config.json"].read_text(encoding="utf-8"))
    config.pop("api_key")
    case["config.json"].write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "out"
    record_failure("missing-key", run_cli(case, out))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["pipeline_trace"][0]["details"]["sample_results"][0]["error_code"] == "missing_api_key"


@pytest.mark.parametrize("metric", ["accuracy", "llm_judge"])
def test_pipeline_and_parameter_errors(case, tmp_path, metric):
    spec = build_pipeline_spec("slu", metric=metric)
    spec["run_args"]["llm_config"] = str(case["config.json"])
    if metric == "llm_judge":
        spec["nodes"][0]["version"] = "invalid-version"
    case["pipeline.json"].write_text(json.dumps(spec), encoding="utf-8")
    message = record_failure("pipeline:" + metric, run_cli(case, tmp_path / "out"))
    if metric == "accuracy":
        assert "only accepted" in message


def test_human_cli_errors_are_readable(case, tmp_path):
    case["config.json"].write_text("{}", encoding="utf-8")
    result = run_cli(case, tmp_path / "out", json_output=False)
    assert result.exit_code == 1
    assert "Error:" in result.output and "base_url" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("field,value", [("version", "v999"), ("stage", "frontend")])
@pytest.mark.parametrize("metric", ["accuracy", "llm_judge"])
def test_declared_node_metadata_must_match_current_route(case, tmp_path, metric, field, value):
    spec = build_pipeline_spec("slu", metric=metric)
    spec["nodes"][0][field] = value
    case["pipeline.json"].write_text(json.dumps(spec), encoding="utf-8")
    message = record_failure("node-metadata:" + field, run_cli(case, tmp_path / "out"))
    assert "identity mismatch" in message and field in message


@pytest.mark.parametrize("field", ["hyp_file", "prompt_jsonl"])
def test_missing_required_input_role(case, tmp_path, field):
    args = ["metric", "run", "--pipeline", str(case["pipeline.json"]),
            "--output-dir", str(tmp_path / "out"), "--json"]
    if field != "hyp_file":
        args += ["--hyp-file", str(case["hyp.txt"])]
    if field != "prompt_jsonl":
        args += ["--prompt-jsonl", str(case["questions.jsonl"])]
    message = record_failure("missing-role:" + field, CliRunner().invoke(app, args))
    assert field in message


def test_missing_llm_config_path(case, tmp_path):
    spec = json.loads(case["pipeline.json"].read_text(encoding="utf-8"))
    spec["run_args"]["llm_config"] = None
    case["pipeline.json"].write_text(json.dumps(spec), encoding="utf-8")
    assert "llm_config" in record_failure("missing-config-path", run_cli(case, tmp_path / "out"))


@pytest.mark.parametrize("text", ["{", "[]", "null", '{}'])
def test_invalid_pipeline_json_has_cli_error(case, tmp_path, text):
    case["pipeline.json"].write_text(text, encoding="utf-8")
    record_failure("invalid-pipeline-json", run_cli(case, tmp_path / "out"))


def test_output_path_is_file_reports_error(case, tmp_path, monkeypatch):
    monkeypatch.setattr(node, "call_judge", lambda *args: {"score": 4, "reason": "ok"})
    out = tmp_path / "output-file"
    out.write_text("keep this", encoding="utf-8")
    record_failure("output-path-is-file", run_cli(case, out))
    assert out.read_text(encoding="utf-8") == "keep this"


def test_partial_success_keeps_counts_null_score_and_same_report_envelope(case, tmp_path, monkeypatch):
    case["questions.jsonl"].write_text(
        '\n'.join(json.dumps({"key": key, "question": "q"}) for key in ["a", "b"]) + '\n',
        encoding="utf-8",
    )
    for name in ("hyp.txt", "ref.txt"):
        case[name].write_text("a\tanswer\nb\tanswer\n", encoding="utf-8")

    def judge(config, messages):
        if not hasattr(judge, "called"):
            judge.called = True
            return {"score": 5, "reason": "ok"}
        raise node.JudgeError("http_error_429")

    monkeypatch.setattr(node, "call_judge", judge)
    out = tmp_path / "out"
    record_failure("partial-success", run_cli(case, out))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    scores = report["details"]["scoring_result"]
    assert scores == {"score": None, "llm_judge": None, "status": "failed",
                      "num_samples": 2, "num_scored": 1, "num_failed": 1}
    assert len(report["pipeline_trace"][0]["details"]["sample_results"]) == 2
    assert (out / "pipeline_description.json").is_file()


@pytest.mark.parametrize("output_mode", ["choice_id", "choice_text"])
def test_judge_report_matches_accuracy_envelope(case, tmp_path, monkeypatch, output_mode):
    monkeypatch.setattr(node, "call_judge", lambda *args: {"score": 4, "reason": "ok"})
    reports, descriptions, summaries = {}, {}, {}
    for metric in ("accuracy", "llm_judge"):
        pipeline_id = None
        if metric == "accuracy":
            pipeline_id = f"slu.any.accuracy.prompt_norm_{output_mode}_v1.classify_v1"
        spec = (build_pipeline_spec("slu", pipeline_id=pipeline_id) if pipeline_id
                else build_pipeline_spec("slu", metric=metric))
        if metric == "llm_judge":
            spec["run_args"]["llm_config"] = str(case["config.json"])
        case["pipeline.json"].write_text(json.dumps(spec), encoding="utf-8")
        out = tmp_path / metric
        result = run_cli(case, out)
        assert result.exit_code == 0, result.output
        summaries[metric] = json.loads(result.stdout)
        reports[metric] = json.loads((out / "report.json").read_text(encoding="utf-8"))
        descriptions[metric] = json.loads((out / "pipeline_description.json").read_text(encoding="utf-8"))
        assert Path(summaries[metric]["report_path"]).is_file()
        assert set(path.name for path in out.iterdir()) == {"report.json", "pipeline_description.json"}
        assert reports[metric]["pipeline_id"] == descriptions[metric]["pipeline_id"]
        assert reports[metric]["score"] == summaries[metric]["score"]
        assert set(reports[metric]["input_files"]) == {"ref", "hyp", "prompt_jsonl"}
        assert descriptions[metric]["computation_node_ids"] == reports[metric]["computation_node_ids"]
    assert set(reports["accuracy"]) == set(reports["llm_judge"])
    assert set(descriptions["accuracy"]) == set(descriptions["llm_judge"])
    assert set(summaries["accuracy"]) == set(summaries["llm_judge"])
    for key in reports["accuracy"]:
        assert type(reports["accuracy"][key]) is type(reports["llm_judge"][key])
    accuracy_trace = reports["accuracy"]["pipeline_trace"][0]
    judge_trace = reports["llm_judge"]["pipeline_trace"][0]
    assert set(accuracy_trace) == set(judge_trace)
    assert reports["accuracy"]["score"] == 1.0
    assert reports["llm_judge"]["score"] == 4.0
    print(json.dumps({"compatibility": output_mode, "report_fields": sorted(reports["llm_judge"]),
                      "accuracy_score": 1.0, "judge_score": 4.0}))
