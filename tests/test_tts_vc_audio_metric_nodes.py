from __future__ import annotations


def test_wavlm_large_sim_node_scores_named_backend() -> None:
    from sure_eval.evaluation.nodes.scoring.wavlm_large_sim import score_wavlm_large_sim

    result = score_wavlm_large_sim(
        [
            ("utt1", "hyp.wav", "ref.wav"),
            ("utt2", "hyp2.wav", "ref2.wav"),
        ],
        provider=lambda prediction, reference, **kwargs: {
            "ASV": 0.7 if prediction == "hyp.wav" else 0.9,
            "backend": "unit-test",
            "prediction_audio": prediction,
            "reference_audio": reference,
        },
    )

    assert result.stage == "scoring"
    assert result.node_id == "scoring/wavlm_large_sim"
    assert result.details["metric"] == "sim/wavlm-large"
    assert result.details["result"]["score"] == 0.8
    assert result.details["result"]["per_sample"][0]["ASV"] == 0.7
    assert "prediction_audio" not in result.details["result"]["per_sample"][0]
    assert result.internal_stages == ("embedding_or_score_provider", "score_normalization", "mean_aggregation")


def test_dnsmos_node_scores_with_metric_aggregation() -> None:
    from sure_eval.evaluation.nodes.scoring.dnsmos import score_dnsmos

    result = score_dnsmos(
        [
            ("utt1", "hyp.wav"),
            ("utt2", "hyp2.wav"),
        ],
        provider=lambda prediction, reference="", **kwargs: {
            "OVRL": 3.0 if prediction == "hyp.wav" else 4.0,
            "SIG": 3.2,
            "filename": prediction,
        },
    )

    assert result.stage == "scoring"
    assert result.node_id == "scoring/dnsmos"
    assert result.details["metric"] == "dnsmos"
    assert result.details["result"]["score"] == 3.5
    assert result.details["result"]["mean_SIG"] == 3.2
    assert "filename" not in result.details["result"]["per_sample"][0]
    assert result.internal_stages == ("audio_score_provider", "score_normalization", "mean_aggregation")


def test_audio_quality_dispatch_imports_without_task_cycle() -> None:
    from sure_eval.evaluation.nodes.scoring._audio_quality_dispatch import score_speaker_metric

    result = score_speaker_metric(
        [("utt1", "hyp.wav", "ref.wav")],
        backend_name="wavlm-large",
        provider=lambda prediction, reference, **kwargs: {"ASV": 0.5},
    )

    assert result.node_id == "scoring/wavlm_large_sim"
