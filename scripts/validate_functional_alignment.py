#!/usr/bin/env python
"""Compare demo evaluation results between a baseline and candidate checkout.

The script intentionally compares score-bearing outputs instead of public
pipeline IDs. Identity fields may change during route/catalog refactors, but
the functional metric results and score-affecting node chain must stay aligned.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


METRIC_ALIASES = {
    "cer_canonical": "cer",
    "wer_canonical": "wer",
    "mer_canonical": "mer",
    "tts_cer": "cer",
    "tts_wer": "wer",
    "vc_cer": "cer",
    "vc_wer": "wer",
    "tse_cer": "cer",
    "tse_wer": "wer",
    "sim": "spk_sim",
    "sim/wavlm-large": "spk_sim",
    "sim/ecapa-tdnn": "spk_sim",
    "sim/eres2net": "spk_sim",
    "macro-recall": "macro_recall",
    "si-sdr": "si_sdr",
    "wv-mos": "wv_mos",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, help="Checkout for the pre-change baseline")
    parser.add_argument("--candidate-repo", type=Path, default=Path.cwd(), help="Checkout to validate")
    parser.add_argument("--work-dir", type=Path, help="Directory for generated demo inputs and outputs")
    parser.add_argument("--output", type=Path, help="Write comparison summary JSON")
    parser.add_argument("--emit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    parser.add_argument("--emit-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.emit:
        if not args.variant or not args.emit_output:
            parser.error("--emit requires --variant and --emit-output")
        payload = run_demo_suite(args.variant, args.emit_output.parent)
        _write_json(args.emit_output, payload)
        return 0

    if not args.baseline_repo:
        parser.error("--baseline-repo is required")

    baseline_repo = args.baseline_repo.resolve()
    candidate_repo = args.candidate_repo.resolve()
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="sure-eval-alignment-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    baseline_payload = _run_emit(baseline_repo, "baseline", work_dir / "baseline")
    candidate_payload = _run_emit(candidate_repo, "candidate", work_dir / "candidate")
    comparison = compare_payloads(baseline_payload, candidate_payload)
    comparison["work_dir"] = str(work_dir)
    comparison["baseline_repo"] = str(baseline_repo)
    comparison["candidate_repo"] = str(candidate_repo)

    output = args.output or work_dir / "functional_alignment_summary.json"
    _write_json(output, comparison)
    print(f"Wrote comparison summary to {output}")
    for case in comparison["cases"]:
        print(f"OK {case}")
    return 0


def _run_emit(repo: Path, variant: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "results.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    env.setdefault("UV_LINK_MODE", "copy")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--emit",
            "--variant",
            variant,
            "--emit-output",
            str(output),
        ],
        cwd=repo,
        env=env,
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def run_demo_suite(variant: str, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "asr_zh_cer_aispeech": _case_asr_zh_cer_aispeech(work_dir / "asr_zh_cer"),
        "asr_zh_cer_canonical": _case_asr_zh_cer_canonical(work_dir / "asr_zh_canon"),
        "asr_cs_mer_canonical": _case_asr_cs_mer_canonical(work_dir / "asr_cs_mer"),
        "asr_canonical_description": _case_asr_canonical_description(variant),
        "s2tt_zh_bleu": _case_s2tt_zh_bleu(work_dir / "s2tt"),
        "classification_accuracy": _case_classification(work_dir / "classification"),
        "slu_choice_accuracy": _case_slu(work_dir / "slu"),
        "kws_macro_recall": _case_kws_macro_recall(),
        "se_stub_bundle": _case_se_stub(),
        "tse_stub_bundle": _case_tse_stub(),
        "tts_stub_bundle": _case_tts_stub(),
        "vc_stub_bundle": _case_vc_stub(),
    }
    payload: dict[str, Any] = {"variant": variant, "cases": cases}
    if variant == "candidate":
        payload["identity_checks"] = _candidate_identity_checks()
    return payload


def _case_asr_zh_cer_aispeech(work_dir: Path) -> dict[str, Any]:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref, hyp = _write_key_text_pair(
        work_dir,
        [("utt1", "你好世界"), ("utt2", "今天 天气 很好")],
        [("utt1", "你好世界"), ("utt2", "今天 天气 一般")],
    )
    report = evaluate_asr_files(str(ref), str(hyp), language="zh", metric="cer", normalizer="aispeech")
    return _report_summary(report)


def _case_asr_zh_cer_canonical(work_dir: Path) -> dict[str, Any]:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref, hyp = _write_key_text_pair(
        work_dir,
        [("utt1", "今天 二零二四 年"), ("utt2", "百分之五十 完成")],
        [("utt1", "今天 2024 年"), ("utt2", "50% 完成")],
    )
    report = evaluate_asr_files(str(ref), str(hyp), language="zh", metric="cer_canonical")
    return _report_summary(report)


def _case_asr_cs_mer_canonical(work_dir: Path) -> dict[str, Any]:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref, hyp = _write_key_text_pair(
        work_dir,
        [("utt1", "hello 世界 twenty four"), ("utt2", "colour 百分之五十")],
        [("utt1", "hello 世界 24"), ("utt2", "color 50%")],
    )
    report = evaluate_asr_files(str(ref), str(hyp), language="cs", metric="mer_canonical")
    return _report_summary(report)


def _case_asr_canonical_description(variant: str) -> dict[str, Any]:
    from sure_eval.evaluation.scripts import describe_pipeline

    if variant == "candidate":
        description = describe_pipeline("asr", pipeline_id="asr.zh.cer.canonical_itn_zh_v1.token_cer_v1")
    else:
        description = describe_pipeline("asr", language="zh", metric="cer_canonical")
    return _description_summary(description)


def _case_s2tt_zh_bleu(work_dir: Path) -> dict[str, Any]:
    from sure_eval.evaluation.tasks.s2tt.pipeline import evaluate_s2tt_files

    ref, hyp = _write_key_text_pair(
        work_dir,
        [("utt1", "你好世界。"), ("utt2", "今天天气很好。")],
        [("utt1", "你好世界。"), ("utt2", "今天的天气很好。")],
    )
    report = evaluate_s2tt_files(str(ref), str(hyp), language="zh", metric="bleu")
    return _report_summary(report)


def _case_classification(work_dir: Path) -> dict[str, Any]:
    from sure_eval.evaluation.tasks.classification.pipeline import evaluate_classification_files

    ref, hyp = _write_key_text_pair(
        work_dir,
        [("utt1", "cat"), ("utt2", "dog"), ("utt3", "cat")],
        [("utt1", "cat"), ("utt2", "cat"), ("utt3", "cat")],
    )
    report = evaluate_classification_files(str(ref), str(hyp))
    return _report_summary(report)


def _case_slu(work_dir: Path) -> dict[str, Any]:
    from sure_eval.evaluation.tasks.slu.pipeline import evaluate_slu_files

    ref, hyp = _write_key_text_pair(
        work_dir,
        [("utt1", "A"), ("utt2", "B")],
        [("utt1", "A. book a flight"), ("utt2", "C. cancel")],
    )
    prompt_jsonl = work_dir / "prompt.jsonl"
    _write_jsonl(
        prompt_jsonl,
        [
            {"key": "utt1", "prompt": "A. book a flight\nB. play music\nC. cancel"},
            {"key": "utt2", "prompt": "A. book a flight\nB. play music\nC. cancel"},
        ],
    )
    report = evaluate_slu_files(str(ref), str(hyp), prompt_jsonl=str(prompt_jsonl))
    return _report_summary(report)


def _case_kws_macro_recall() -> dict[str, Any]:
    from sure_eval.evaluation.tasks.kws import KWSSample
    from sure_eval.evaluation.tasks.kws.pipeline import evaluate_kws_samples

    samples = [
        KWSSample(
            key="pos",
            expected_detected=True,
            expected_keyword="嗨小问",
            duration=1.2,
            detected=True,
            predicted_keyword="嗨小问",
            score=0.9,
        ),
        KWSSample(
            key="neg",
            expected_detected=False,
            duration=2.4,
            detected=False,
            score=None,
        ),
    ]
    report = evaluate_kws_samples(
        samples,
        metric="macro-recall",
        threshold=0.5,
        thresholds=[0.0, 0.5, 0.95],
        macro_recall_false_alarms=0,
    )
    return _report_summary(report)


def _case_se_stub() -> dict[str, Any]:
    from sure_eval.evaluation.tasks.se.pipeline import evaluate_se_samples
    from sure_eval.evaluation.tasks.se.types import SESample

    report = evaluate_se_samples(
        [SESample(enhanced_audio="enhanced.wav", reference_audio="clean.wav", sample_id="utt1")],
        metrics=("si-sdr", "dnsmos"),
        reference_providers={"si-sdr": lambda prediction, reference, **kwargs: {"si_sdr": 7.5}},
        mos_providers={"dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.0}},
    )
    return _report_summary(report)


def _case_tse_stub() -> dict[str, Any]:
    from sure_eval.evaluation.tasks.tse.pipeline import evaluate_tse_samples
    from sure_eval.evaluation.tasks.tse.types import TSESample

    report = evaluate_tse_samples(
        [TSESample(prediction_audio="pred.wav", reference_audio="ref.wav", language="en", sample_id="utt1")],
        metrics=("sim/wavlm-large", "dnsmos"),
        speaker_providers={"wavlm-large": lambda prediction, reference, **kwargs: {"ASV": 0.86}},
        mos_providers={"dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.6}},
    )
    return _report_summary(report)


def _case_tts_stub() -> dict[str, Any]:
    from sure_eval.evaluation.tasks.tts.compat import TTSSample
    from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples

    class StaticTranscriber:
        def transcribe(self, audio_path: str, *, language: str = "zh") -> str:
            return "你好世界"

    report = evaluate_tts_samples(
        [
            TTSSample(
                prediction_audio="tts.wav",
                reference_text="你好世界",
                reference_audio="speaker.wav",
                language="zh",
                sample_id="utt1",
            )
        ],
        metrics=("tts_cer", "sim/wavlm-large", "dnsmos"),
        transcribers={"zh": StaticTranscriber()},
        speaker_providers={"wavlm-large": lambda prediction, reference, **kwargs: {"ASV": 0.88}},
        mos_providers={"dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.4}},
    )
    return _report_summary(report)


def _case_vc_stub() -> dict[str, Any]:
    from sure_eval.evaluation.tasks.vc.compat import VCSample
    from sure_eval.evaluation.tasks.vc.pipeline import evaluate_vc_samples

    class StaticTranscriber:
        def transcribe(self, audio_path: str, *, language: str = "zh") -> str:
            return "你好世界"

    report = evaluate_vc_samples(
        [
            VCSample(
                converted_audio="converted.wav",
                reference_audio="target.wav",
                reference_text="你好世界",
                source_audio="source.wav",
                language="zh",
                sample_id="utt1",
            )
        ],
        metrics=("vc_cer", "sim/ecapa-tdnn", "utmos"),
        transcribers={"zh": StaticTranscriber()},
        speaker_providers={"ecapa-tdnn": lambda prediction, reference, **kwargs: {"ASV": 0.81}},
        mos_providers={"utmos": lambda prediction, reference="", **kwargs: {"utmos": 3.7}},
    )
    return _report_summary(report)


def _candidate_identity_checks() -> dict[str, Any]:
    from sure_eval.evaluation.cli_adapters import build_pipeline_spec
    from sure_eval.evaluation.scripts import describe_pipeline

    pipeline_id = "asr.cs.mer.canonical_itn_cs_v1.token_mer_v1"
    description = describe_pipeline("asr", pipeline_id=pipeline_id)
    payload = build_pipeline_spec("asr", pipeline_id=pipeline_id)
    old_selector_rejected = False
    try:
        describe_pipeline("asr", language="cs", metric="mer_canonical")
    except ValueError:
        old_selector_rejected = True
    return {
        "pipeline_id": description.pipeline_id,
        "metric": description.metric,
        "execution_metrics": list(description.execution_metrics),
        "route_config_path": payload["route_config_path"],
        "script_entrypoint": payload["script_entrypoint"],
        "executor": payload["executor"],
        "old_selector_rejected": old_selector_rejected,
        "internal_executor_metric_leaked": "internal_executor_metric" in json.dumps(payload, ensure_ascii=False),
    }


def _report_summary(report: Any) -> dict[str, Any]:
    trace_node_ids = [node.node_id for node in report.pipeline_trace]
    computation_node_ids = list(getattr(report, "computation_node_ids", ()) or ()) or trace_node_ids
    return {
        "task": str(report.task),
        "language": str(report.language),
        "metric": _metric_key(str(report.metric)),
        "score": _json_number(report.score),
        "result_scores": _extract_result_scores(report),
        "trace_node_ids": trace_node_ids,
        "computation_node_ids": computation_node_ids,
    }


def _description_summary(description: Any) -> dict[str, Any]:
    return {
        "task": str(description.task),
        "language": str(description.language),
        "metric": _metric_key(str(description.metric)),
        "node_ids": list(description.node_ids),
        "required_roles": list(description.required_roles),
    }


def _extract_result_scores(report: Any) -> dict[str, float | str]:
    details = report.details or {}
    scores: dict[str, float | str] = {}
    results = details.get("results") if isinstance(details, dict) else None
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, dict) and "score" in value:
                scores[_metric_key(str(key))] = _json_number(value["score"])
    scoring = details.get("scoring_result") if isinstance(details, dict) else None
    if isinstance(scoring, dict):
        if "score" in scoring:
            scores.setdefault(_metric_key(str(report.metric)), _json_number(scoring["score"]))
        for key in ("cer", "wer", "mer", "bleu", "bleu_char", "chrf", "accuracy", "macro-recall", "macro_recall"):
            if key in scoring and isinstance(scoring[key], (int, float)):
                scores[_metric_key(key)] = _json_number(scoring[key])
    if not scores:
        scores[_metric_key(str(report.metric))] = _json_number(report.score)
    return dict(sorted(scores.items()))


def compare_payloads(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_cases = baseline["cases"]
    candidate_cases = candidate["cases"]
    if set(baseline_cases) != set(candidate_cases):
        raise AssertionError(f"Case mismatch: {set(baseline_cases) ^ set(candidate_cases)}")
    for case_name in sorted(baseline_cases):
        _assert_aligned(case_name, baseline_cases[case_name], candidate_cases[case_name])
    checks = candidate.get("identity_checks") or {}
    if checks.get("metric") != "mer":
        raise AssertionError(f"candidate ASR pipeline_id metric check failed: {checks}")
    if checks.get("execution_metrics") != ["mer"]:
        raise AssertionError(f"candidate ASR execution_metrics check failed: {checks}")
    if checks.get("route_config_path") != "src/sure_eval/evaluation/tasks/asr/routes.yaml":
        raise AssertionError(f"candidate route_config_path check failed: {checks}")
    if checks.get("internal_executor_metric_leaked"):
        raise AssertionError("internal_executor_metric leaked into public pipeline spec")
    if not checks.get("old_selector_rejected"):
        raise AssertionError("old mer_canonical public selector was not rejected")
    return {
        "status": "ok",
        "cases": sorted(baseline_cases),
        "identity_checks": checks,
    }


def _assert_aligned(case_name: str, baseline: Any, candidate: Any, path: str = "") -> None:
    if isinstance(baseline, dict) and isinstance(candidate, dict):
        if set(baseline) != set(candidate):
            raise AssertionError(
                f"{case_name}{path}: key mismatch baseline={sorted(baseline)} candidate={sorted(candidate)}"
            )
        for key in sorted(baseline):
            _assert_aligned(case_name, baseline[key], candidate[key], f"{path}.{key}")
        return
    if isinstance(baseline, list) and isinstance(candidate, list):
        if len(baseline) != len(candidate):
            raise AssertionError(f"{case_name}{path}: length mismatch {len(baseline)} != {len(candidate)}")
        for index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
            _assert_aligned(case_name, left, right, f"{path}[{index}]")
        return
    if isinstance(baseline, (int, float)) and isinstance(candidate, (int, float)):
        if math.isinf(float(baseline)) or math.isinf(float(candidate)):
            if baseline != candidate:
                raise AssertionError(f"{case_name}{path}: {baseline!r} != {candidate!r}")
            return
        if not math.isclose(float(baseline), float(candidate), rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f"{case_name}{path}: {baseline!r} != {candidate!r}")
        return
    if baseline != candidate:
        raise AssertionError(f"{case_name}{path}: {baseline!r} != {candidate!r}")


def _metric_key(metric: str) -> str:
    raw = metric.lower()
    if raw in METRIC_ALIASES:
        return METRIC_ALIASES[raw]
    normalized = raw.replace("_", "-")
    return METRIC_ALIASES.get(normalized, normalized.replace("-", "_"))


def _json_number(value: Any) -> float | str:
    if isinstance(value, (int, float)):
        if math.isinf(float(value)):
            return "inf" if value > 0 else "-inf"
        if math.isnan(float(value)):
            return "nan"
        return float(value)
    return value


def _write_key_text_pair(
    work_dir: Path,
    ref_rows: list[tuple[str, str]],
    hyp_rows: list[tuple[str, str]],
) -> tuple[Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    ref = work_dir / "ref.txt"
    hyp = work_dir / "hyp.txt"
    ref.write_text("".join(f"{key}\t{text}\n" for key, text in ref_rows), encoding="utf-8")
    hyp.write_text("".join(f"{key}\t{text}\n" for key, text in hyp_rows), encoding="utf-8")
    return ref, hyp


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
