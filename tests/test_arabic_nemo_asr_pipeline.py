from __future__ import annotations

from pathlib import Path


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

    try:
        normalized = normalize_nemo_text("21", cache_dir=None)
    except RuntimeError as exc:
        if "node-local environment" in str(exc):
            return
        raise

    assert normalized == "واحد وعشرون"


def test_asr_ar_route_scores_raw_key_text_with_nemo_normalization(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    ref_file.write_text("utt1\t21 كتابا\n", encoding="utf-8")
    hyp_file.write_text("utt1\tواحد وعشرون كتابا\n", encoding="utf-8")

    try:
        report = evaluate_asr_files(
            str(ref_file),
            str(hyp_file),
            language="ar",
            metric="cer",
        )
    except RuntimeError as exc:
        if "node-local environment" in str(exc):
            return
        raise

    assert report.score == 0.0
    assert report.pipeline_id == "asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1"
    assert report.pipeline_trace[0].node_id == "normalization/nemo_norm"
