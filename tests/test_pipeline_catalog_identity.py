from __future__ import annotations

import json
from pathlib import Path


CATALOG = Path("docs/pipeline_catalog.jsonl")


def _catalog_rows() -> list[dict]:
    return [json.loads(line) for line in CATALOG.read_text(encoding="utf-8").splitlines() if line]


def test_pipeline_catalog_rows_expose_pipeline_identity() -> None:
    rows = _catalog_rows()
    assert rows
    for row in rows:
        assert row["pipeline_id"]
        assert row["pipeline_kind"] in {"atomic", "bundle"}
        assert row["computation_node_ids"]
        assert row["task_config_path"].startswith("src/sure_eval/evaluation/tasks/")
        assert row["route_config_path"].startswith("src/sure_eval/evaluation/tasks/")
        assert not Path(row["task_config_path"]).is_absolute()
        assert not Path(row["route_config_path"]).is_absolute()
        assert row["script_entrypoint"].startswith("sure_eval.evaluation.scripts.")
        assert row["executor"].startswith("sure_eval.evaluation.tasks.")
        assert "route_id" not in row
        assert "route_ids" not in row
        assert "legacy_pipeline_id" not in row
        assert "metric_family" not in row
        if row["pipeline_kind"] == "atomic":
            assert row["member_pipeline_ids"] == []
            assert len(row["execution_metrics"]) == 1
        else:
            assert row["member_pipeline_ids"]
            assert row["metric"] == "multi"
            assert len(row["execution_metrics"]) == len(row["member_pipeline_ids"])


def test_pipeline_catalog_has_new_kws_and_sa_asr_identities() -> None:
    rows = _catalog_rows()
    kws_macro_rows = [
        row for row in rows if row["task_alias"] == "kws" and row["metric"] == "macro_recall"
    ]

    assert {row["pipeline_id"] for row in kws_macro_rows} == {
        "kws.any.macro_recall.conversion_kws_sure_json_to_samples_v1.wekws_det_v1",
        "kws.any.macro_recall.conversion_kws_wekws_score_ctc_to_samples_v1.wekws_det_v1",
        "kws.any.macro_recall.conversion_kws_wekws_frame_score_to_samples_v1.wekws_det_v1",
    }
    assert {row["execution_metrics"][0] for row in kws_macro_rows} == {"macro-recall"}

    sa_asr = next(row for row in rows if row["task_alias"] == "sa_asr")
    assert sa_asr["pipeline_id"] == (
        "sa_asr.en.cpwer.conversion_sa_asr_cpwer_v1.gstar_norm_v1.meeteval_v1"
    )
    assert sa_asr["computation_node_ids"][0] == "conversion/sa_asr__cpwer"


def test_pipeline_catalog_asr_canonical_rows_use_canonical_public_metrics() -> None:
    rows = _catalog_rows()
    asr_rows = [row for row in rows if row["task_alias"] == "asr"]

    assert not {
        metric
        for row in asr_rows
        for metric in row["execution_metrics"]
        if metric.endswith("_canonical")
    }
    assert {
        row["pipeline_id"]
        for row in asr_rows
        if row["pipeline_id"].endswith("canonical_itn_cs_v1.token_mer_v1")
    } == {"asr.cs.mer.canonical_itn_cs_v1.token_mer_v1"}
