from __future__ import annotations

from pathlib import Path


def test_asr_ar_route_describes_nemo_itn_and_wenet_wer() -> None:
    from sure_eval.evaluation.scripts.asr import describe_pipeline

    description = describe_pipeline(language="ar", metric="wer")

    assert description.pipeline_id == "asr.ar.wer.nemo_norm_ar_itn_v1.wenet_wer_v1"
    assert description.language == "ar"
    assert description.node_ids == (
        "normalization/nemo_norm",
        "scoring/wenet_wer",
    )


def test_asr_ar_nemo_normalizer_converts_spoken_number_to_written_form() -> None:
    from sure_eval.evaluation.nodes.normalization.nemo_norm import normalize_nemo_text

    try:
        normalized = normalize_nemo_text("واحد وعشرون", cache_dir=None)
    except RuntimeError as exc:
        if "node-local environment" in str(exc):
            return
        raise

    assert normalized == "21"


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
            metric="wer",
        )
    except RuntimeError as exc:
        if "node-local environment" in str(exc):
            return
        raise

    assert report.score == 0.0
    assert report.pipeline_id == "asr.ar.wer.nemo_norm_ar_itn_v1.wenet_wer_v1"
    assert report.pipeline_trace[0].node_id == "normalization/nemo_norm"
