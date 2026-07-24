#!/usr/bin/env python3
"""Generate docs/pipeline_catalog.jsonl from describe_pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

from sure_eval.evaluation.scripts import describe_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "pipeline_catalog.jsonl"

COMBINATIONS = [
    ("asr", {"language": "zh", "metric": "cer"}),
    ("asr", {"pipeline_id": "asr.zh.cer.canonical_itn_zh_v1.token_cer_v1"}),
    ("asr", {"language": "en", "metric": "wer"}),
    ("asr", {"pipeline_id": "asr.en.wer.canonical_itn_en_v1.token_mer_v1"}),
    ("asr", {"language": "cs", "metric": "mer"}),
    ("asr", {"pipeline_id": "asr.cs.mer.canonical_itn_cs_v1.token_mer_v1"}),
    ("s2tt", {"language": "zh", "metric": "bleu"}),
    ("s2tt", {"language": "zh", "metric": "bleu_char"}),
    ("s2tt", {"language": "zh", "metric": "chrf"}),
    ("s2tt", {"language": "zh", "metric": "xcomet_xl"}),
    ("s2tt", {"language": "zh", "metric": "bleurt_20"}),
    ("s2tt", {"language": "en", "metric": "bleu"}),
    ("s2tt", {"language": "en", "metric": "xcomet_xl"}),
    ("s2tt", {"language": "en", "metric": "bleurt_20"}),
    ("sd", {"metric": "der"}),
    ("sa_asr", {"metric": "cpwer"}),
    ("tts", {"language": "zh", "metrics": ("tts_cer",)}),
    ("tts", {"language": "en", "metrics": ("tts_wer",)}),
    ("tts", {"language": "zh", "metrics": ("sim/wavlm-large",)}),
    ("tts", {"language": "zh", "metrics": ("sim/ecapa-tdnn",)}),
    ("tts", {"language": "zh", "metrics": ("sim/eres2net",)}),
    ("tts", {"language": "zh", "metrics": ("dnsmos",)}),
    ("tts", {"language": "zh", "metrics": ("wv-mos",)}),
    ("tts", {"language": "zh", "metrics": ("utmos",)}),
    ("tts", {"language": "zh", "metrics": ("tts_cer", "sim/wavlm-large", "dnsmos")}),
    ("vc", {"language": "zh", "metrics": ("vc_cer",)}),
    ("vc", {"language": "en", "metrics": ("vc_wer",)}),
    ("vc", {"language": "zh", "metrics": ("sim/wavlm-large",)}),
    ("vc", {"language": "zh", "metrics": ("dnsmos",)}),
    ("vc", {"language": "zh", "metrics": ("vc_cer", "sim/wavlm-large", "dnsmos")}),
    ("se", {"metrics": ("si-sdr",)}),
    ("se", {"metrics": ("stoi",)}),
    ("se", {"metrics": ("pesq",)}),
    ("se", {"metrics": ("dnsmos",)}),
    ("se", {"metrics": ("wv-mos",)}),
    ("se", {"metrics": ("utmos",)}),
    ("se", {"metrics": ("si-sdr", "stoi", "pesq", "dnsmos", "wv-mos", "utmos")}),
    ("tse", {"language": "zh", "metrics": ("si_sdr",)}),
    ("tse", {"language": "en", "metrics": ("si_sdr",)}),
    ("tse", {"language": "zh", "metrics": ("sim/wavlm-large",)}),
    ("tse", {"language": "zh", "metrics": ("sim/ecapa-tdnn",)}),
    ("tse", {"language": "zh", "metrics": ("sim/eres2net",)}),
    ("tse", {"language": "zh", "metrics": ("dnsmos",)}),
    ("tse", {"language": "zh", "metrics": ("wv-mos",)}),
    ("tse", {"language": "zh", "metrics": ("utmos",)}),
    ("tse", {"language": "zh", "metrics": ("tse_cer",)}),
    ("tse", {"language": "en", "metrics": ("tse_wer",)}),
    ("tse", {"language": "zh", "metrics": ("si_sdr", "sim/wavlm-large", "dnsmos")}),
    ("classification", {}),
    ("ser", {}),
    ("gr", {}),
    ("slu", {}),
    ("kws", {"metric": "accuracy", "input_mode": "sure_json"}),
    ("kws", {"metric": "macro-recall", "input_mode": "sure_json"}),
    ("kws", {"metric": "accuracy", "input_mode": "wekws_score_ctc"}),
    ("kws", {"metric": "macro-recall", "input_mode": "wekws_score_ctc"}),
    ("kws", {"metric": "accuracy", "input_mode": "wekws_frame_score"}),
    ("kws", {"metric": "macro-recall", "input_mode": "wekws_frame_score"}),
]


def main() -> None:
    rows: list[dict] = []
    for task, kwargs in COMBINATIONS:
        desc = describe_pipeline(task, **kwargs)
        rows.append(
            {
                "task": desc.task,
                "task_alias": task,
                "language": desc.language,
                "metric": desc.metric,
                "pipeline_id": desc.pipeline_id,
                "pipeline_kind": desc.pipeline_kind,
                "member_pipeline_ids": list(desc.member_pipeline_ids),
                "execution_metrics": list(desc.execution_metrics),
                "nodes": list(desc.node_ids),
                "computation_node_ids": list(desc.computation_node_ids),
                "task_config_path": desc.task_config_path,
                "route_config_path": desc.route_config_path,
                "describe_entrypoint": desc.describe_entrypoint,
                "script_entrypoint": desc.script_entrypoint,
                "executor": desc.executor,
                "required_roles": list(desc.required_roles),
                "optional_roles": list(desc.optional_roles),
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
