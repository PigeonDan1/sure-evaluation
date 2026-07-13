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


def test_wavlm_large_default_provider_uses_seed_checkpoint(monkeypatch) -> None:
    from sure_eval.evaluation.nodes.scoring.common import speaker_providers
    from sure_eval.evaluation.nodes.scoring.wavlm_large_sim import node

    captured: dict[str, object] = {}

    class FakeSeedWavLMProvider:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def __call__(self, audio_path: str) -> list[float]:
            return [1.0, 0.0]

    monkeypatch.setattr(
        speaker_providers,
        "SeedWavLMSpeakerEmbeddingProvider",
        FakeSeedWavLMProvider,
        raising=False,
    )

    provider = node.build_default_provider(device="cpu")

    assert node.DEFAULT_CHECKPOINT_PATH.name == "wavlm_large_finetune.pth"
    assert captured["checkpoint_path"] == node.DEFAULT_CHECKPOINT_PATH
    assert captured["device"] == "cpu"
    assert provider.backend == "seed-tts-wavlm-large-cosine"


def test_wavlm_large_default_provider_accepts_base_config_override(monkeypatch) -> None:
    from sure_eval.evaluation.nodes.scoring.common import speaker_providers
    from sure_eval.evaluation.nodes.scoring.wavlm_large_sim import node

    captured: dict[str, object] = {}

    class FakeSeedWavLMProvider:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def __call__(self, audio_path: str) -> list[float]:
            return [1.0, 0.0]

    monkeypatch.setenv("WAVLM_LARGE_BASE_CONFIG", "/tmp/wavlm-config.json")
    monkeypatch.setattr(
        speaker_providers,
        "SeedWavLMSpeakerEmbeddingProvider",
        FakeSeedWavLMProvider,
        raising=False,
    )

    node.build_default_provider(device="cpu")

    assert captured["model_id"] == "/tmp/wavlm-config.json"


def test_seed_wavlm_state_dict_converter_maps_backbone_and_ecapa_keys() -> None:
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import convert_seed_wavlm_state_dict

    class FakeTensor:
        def __init__(self, *shape: int) -> None:
            self.shape = shape

    seed_state = {
        "feature_weight": FakeTensor(25),
        "feature_extract.model.mask_emb": FakeTensor(1024),
        "feature_extract.model.feature_extractor.conv_layers.0.0.weight": FakeTensor(512, 1, 10),
        "feature_extract.model.feature_extractor.conv_layers.0.2.1.weight": FakeTensor(512),
        "feature_extract.model.feature_extractor.conv_layers.0.2.1.bias": FakeTensor(512),
        "feature_extract.model.post_extract_proj.weight": FakeTensor(1024, 512),
        "feature_extract.model.post_extract_proj.bias": FakeTensor(1024),
        "feature_extract.model.encoder.pos_conv.0.weight_g": FakeTensor(1, 1, 128),
        "feature_extract.model.encoder.pos_conv.0.weight_v": FakeTensor(1024, 64, 128),
        "feature_extract.model.encoder.layer_norm.weight": FakeTensor(1024),
        "feature_extract.model.encoder.layers.0.self_attn.k_proj.weight": FakeTensor(1024, 1024),
        "feature_extract.model.encoder.layers.0.self_attn.grep_a": FakeTensor(1, 16, 1, 1),
        "feature_extract.model.encoder.layers.0.self_attn.grep_linear.weight": FakeTensor(8, 64),
        "feature_extract.model.encoder.layers.0.self_attn.relative_attention_bias.weight": FakeTensor(320, 16),
        "feature_extract.model.encoder.layers.0.self_attn_layer_norm.weight": FakeTensor(1024),
        "feature_extract.model.encoder.layers.0.fc1.weight": FakeTensor(4096, 1024),
        "feature_extract.model.encoder.layers.0.fc2.bias": FakeTensor(1024),
        "feature_extract.model.encoder.layers.0.final_layer_norm.bias": FakeTensor(1024),
        "layer1.conv.weight": FakeTensor(512, 1024, 5),
        "linear.bias": FakeTensor(256),
        "loss_calculator.projection.weight": FakeTensor(5994, 256),
    }

    converted = convert_seed_wavlm_state_dict(seed_state)

    assert converted["feature_weight"].shape == (25,)
    assert converted["wavlm.masked_spec_embed"].shape == (1024,)
    assert converted["wavlm.feature_extractor.conv_layers.0.conv.weight"].shape == (512, 1, 10)
    assert converted["wavlm.feature_extractor.conv_layers.0.layer_norm.weight"].shape == (512,)
    assert converted["wavlm.feature_projection.projection.weight"].shape == (1024, 512)
    assert converted["wavlm.encoder.pos_conv_embed.conv.parametrizations.weight.original0"].shape == (1, 1, 128)
    assert converted["wavlm.encoder.layers.0.attention.gru_rel_pos_const"].shape == (1, 16, 1, 1)
    assert converted["wavlm.encoder.layers.0.attention.rel_attn_embed.weight"].shape == (320, 16)
    assert converted["wavlm.encoder.layers.0.layer_norm.weight"].shape == (1024,)
    assert converted["wavlm.encoder.layers.0.feed_forward.intermediate_dense.weight"].shape == (4096, 1024)
    assert converted["wavlm.encoder.layers.0.feed_forward.output_dense.bias"].shape == (1024,)
    assert converted["layer1.conv.weight"].shape == (512, 1024, 5)
    assert "loss_calculator.projection.weight" not in converted


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
