from __future__ import annotations
from sure_eval.evaluation.tasks.tta.pipeline import evaluate_tta_samples
from sure_eval.evaluation.tasks.tta.types import TTASample


def _samples():
    return [
        TTASample("gen1.wav", "ref1.wav", "dog barking", "one"),
        TTASample("gen2.wav", "ref2.wav", "rain falling", "two"),
    ]


def test_tta_all_metrics_bundle():
    report = evaluate_tta_samples(
        _samples(),
        metrics=(
            "fad",
            "fd",
            "kl",
            "is",
            "clap_score",
            "audiobox_ce",
            "audiobox_cu",
            "audiobox_pc",
            "audiobox_pq",
        ),
        providers={
            "audioldm_eval": lambda rows, metrics: {
                "fad": 1.0,
                "fd": 2.0,
                "kl": 3.0,
                "inception_score_mean": 4.0,
                "inception_score_std": 0.2,
            },
            "clap_score": lambda rows: [0.5, 0.7],
            "audiobox_aesthetics": lambda rows: [
                {"CE": 5, "CU": 6, "PC": 7, "PQ": 8},
                {"CE": 7, "CU": 8, "PC": 9, "PQ": 10},
            ],
        },
    )
    assert report.task == "TTA"
    assert report.metric == "multi"
    assert report.pipeline_kind == "bundle"
    assert report.details["results"]["is"]["score"] == 4.0
    assert report.details["results"]["is"]["score_std"] == 0.2
    assert report.details["results"]["clap_score"]["score"] == 0.6
    assert report.details["results"]["audiobox_ce"]["score"] == 6.0
    assert len(report.pipeline_trace) == 3


def test_tta_atomic_fad_identity():
    report = evaluate_tta_samples(
        _samples(),
        metrics=("fad",),
        providers={"audioldm_eval": lambda rows, metrics: {"frechet_audio_distance": 1.25}},
    )
    assert report.metric == "fad"
    assert report.pipeline_id == "tta.any.fad.audioldm_eval_v1"
    assert report.score == 1.25


def test_tta_runtime_provider_contract_is_flat(monkeypatch):
    from sure_eval.evaluation import audio_runtime

    provider = object()
    monkeypatch.setattr(audio_runtime, "_NodeLocalTTAProvider", lambda *args: provider)

    runtime = audio_runtime.build_tta_runtime(metrics=("fad",), device="cpu")

    assert runtime == {"audioldm_eval": provider}


def test_tta_script_uses_default_runtime_provider_contract(monkeypatch, tmp_path):
    from sure_eval.evaluation import audio_runtime
    from sure_eval.evaluation.scripts import tta

    def provider(rows, metrics):
        return {"frechet_audio_distance": 1.5}
    monkeypatch.setattr(
        audio_runtime,
        "build_tta_runtime",
        lambda **kwargs: {"audioldm_eval": provider},
    )

    report = tta.run(
        _samples(),
        metrics=("fad",),
        output_dir=str(tmp_path / "tta-out"),
        device="cpu",
    )

    assert report.score == 1.5
    assert (tmp_path / "tta-out" / "report.json").is_file()


def test_tta_requires_metric_specific_inputs():
    try:
        evaluate_tta_samples(
            [TTASample("gen.wav", sample_id="one")],
            metrics=("clap_score",),
            providers={"clap_score": lambda rows: [0.5]},
        )
        assert False
    except ValueError as exc:
        assert "caption" in str(exc)


def test_audioldm_runtime_home_exposes_managed_checkpoints(monkeypatch, tmp_path):
    import os

    from sure_eval.evaluation.nodes.scoring.audioldm_eval import node

    checkpoint = tmp_path / "managed" / "Cnn14_16k_mAP=0.438.pth"
    companion = tmp_path / "managed" / "Cnn14_mAP=0.431.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"16k")
    companion.write_bytes(b"32k")
    monkeypatch.setattr(node, "NODE_DIR", tmp_path / "node")
    monkeypatch.setenv("HOME", "/original/home")

    with node._audioldm_runtime_home(checkpoint, companion):
        runtime_home = tmp_path / "node" / "checkpoints" / "runtime_home"
        assert os.environ["HOME"] == str(runtime_home)
        legacy_dir = runtime_home / ".cache" / "audioldm_eval" / "ckpt"
        assert (legacy_dir / checkpoint.name).resolve() == checkpoint.resolve()
        assert (legacy_dir / companion.name).resolve() == companion.resolve()

    assert os.environ["HOME"] == "/original/home"


def test_torchvggish_cache_is_materialized_from_pinned_archive(monkeypatch, tmp_path):
    import zipfile

    from sure_eval.evaluation.nodes.scoring.audioldm_eval import node

    archive_path = tmp_path / "torchvggish.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("torchvggish-pinned/hubconf.py", "# test\n")
    hub_dir = tmp_path / "hub"
    checkpoints = hub_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    weights = (
        checkpoints / "vggish-10086976.pth",
        checkpoints / "vggish_pca_params-970ea276.pth",
    )
    for path in weights:
        path.write_bytes(b"test")

    monkeypatch.setattr(node, "_TORCHVGGISH_ARCHIVE", archive_path)
    monkeypatch.setattr(node, "_TORCH_HUB_DIR", hub_dir)
    monkeypatch.setattr(node, "_TORCHVGGISH_REPO", hub_dir / "harritaylor_torchvggish_master")
    monkeypatch.setattr(node, "_VGGISH_CHECKPOINTS", weights)

    node._prepare_torchvggish_cache()

    assert (node._TORCHVGGISH_REPO / "hubconf.py").is_file()


def test_torchvggish_hub_request_is_redirected_locally(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from sure_eval.evaluation.nodes.scoring.audioldm_eval import node

    calls = []

    def fake_load(repo_or_dir, model, *args, **kwargs):
        calls.append((repo_or_dir, model, args, kwargs))
        return "model"

    fake_torch = SimpleNamespace(hub=SimpleNamespace(load=fake_load))
    repo = tmp_path / "harritaylor_torchvggish_master"
    monkeypatch.setattr(node, "_TORCHVGGISH_REPO", repo)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with node._use_local_torchvggish():
        assert fake_torch.hub.load("harritaylor/torchvggish", "vggish") == "model"

    assert calls == [(str(repo), "vggish", (), {"source": "local"})]
    assert fake_torch.hub.load is fake_load


def test_trusted_legacy_torch_load_is_scoped(monkeypatch):
    import sys
    from types import SimpleNamespace

    from sure_eval.evaluation.nodes.scoring.audioldm_eval import node

    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return "checkpoint"

    fake_torch = SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with node._use_trusted_legacy_torch_load("cpu"):
        assert fake_torch.load("checkpoint.pth") == "checkpoint"

    assert calls == [(("checkpoint.pth",), {"weights_only": False, "map_location": "cpu"})]
    assert fake_torch.load is fake_load


def test_soundfile_torchaudio_load_is_scoped(monkeypatch):
    import sys
    from types import SimpleNamespace

    import numpy as np

    from sure_eval.evaluation.nodes.scoring.audioldm_eval import node

    original_load = object()
    fake_torchaudio = SimpleNamespace(load=original_load)
    calls = []

    def fake_read(uri, **kwargs):
        calls.append((uri, kwargs))
        return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32), 44100

    fake_soundfile = SimpleNamespace(read=fake_read)
    fake_torch = SimpleNamespace(from_numpy=lambda value: value)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    with node._use_soundfile_torchaudio_load():
        waveform, sample_rate = fake_torchaudio.load("audio.wav")

    assert calls == [("audio.wav", {"dtype": "float32", "always_2d": True})]
    assert sample_rate == 44100
    assert waveform.shape == (2, 3)
    assert np.allclose(waveform, [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    assert fake_torchaudio.load is original_load


def test_tta_node_provider_reports_progress(monkeypatch, capsys):
    from sure_eval.evaluation.audio_runtime import _NodeLocalTTAProvider
    from sure_eval.evaluation.nodes.scoring.common import node_local

    call = {}

    def fake_run_node_json(**kwargs):
        call.update(kwargs)
        return {"metrics": {"fad": 1.0, "is": 2.0}}

    monkeypatch.setattr(node_local, "_run_node_json", fake_run_node_json)
    provider = _NodeLocalTTAProvider("scoring/audioldm_eval", "cpu")
    result = provider([("one", "prediction.wav", "reference.wav")], metrics=("fad", "is"))

    output = capsys.readouterr()
    assert result == {"fad": 1.0, "is": 2.0}
    assert call["forward_stderr"] is True
    assert "[TTA] starting scoring/audioldm_eval: 1 samples, metrics=fad,is" in output.err
    assert "[TTA] result scoring/audioldm_eval: fad=1, is=2" in output.err
    assert "[TTA] completed scoring/audioldm_eval in" in output.err


def test_clap_provider_scores_in_batches(monkeypatch, capsys):
    import numpy as np

    from sure_eval.evaluation.nodes.scoring.clap_score.node import CLAPProvider

    class FakeModel:
        def __init__(self):
            self.text_batch_sizes = []
            self.audio_batch_sizes = []

        def get_text_embedding(self, texts, *, use_tensor):
            self.text_batch_sizes.append(len(texts))
            return np.ones((len(texts), 2), dtype=np.float32)

        def get_audio_embedding_from_filelist(self, *, x, use_tensor):
            self.audio_batch_sizes.append(len(x))
            return np.ones((len(x), 2), dtype=np.float32)

    provider = object.__new__(CLAPProvider)
    provider.batch_size = 2
    provider.model = FakeModel()
    rows = [(str(index), f"{index}.wav", f"caption {index}") for index in range(5)]

    scores = provider(rows)

    assert np.allclose(scores, np.ones(5))
    assert provider.model.text_batch_sizes == [2, 2, 1]
    assert provider.model.audio_batch_sizes == [2, 2, 1]
    log = capsys.readouterr().err
    assert "CLAP scoring" in log
    assert "3/3" in log


def test_tta_result_summary_includes_clap_and_audiobox_scores():
    from sure_eval.evaluation.audio_runtime import _tta_result_summary

    assert (
        _tta_result_summary(
            "scoring/clap_score",
            {"result": {"score": 0.61234567}},
        )
        == "clap_score=0.612346"
    )
    assert (
        _tta_result_summary(
            "scoring/audiobox_aesthetics",
            {
                "axes": {
                    "CE": {"score": 6.1},
                    "CU": {"score": 7.2},
                    "PC": {"score": 8.3},
                    "PQ": {"score": 9.4},
                }
            },
        )
        == "audiobox_ce=6.1, audiobox_cu=7.2, audiobox_pc=8.3, audiobox_pq=9.4"
    )


def test_audiobox_provider_prefers_managed_local_checkpoint(monkeypatch, tmp_path):
    import sys
    from types import ModuleType

    from sure_eval.evaluation.nodes.scoring.audiobox_aesthetics import node

    checkpoint = tmp_path / "checkpoints" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    calls = []
    predictor = object()

    infer = ModuleType("audiobox_aesthetics.infer")

    def initialize_predictor(path=None):
        calls.append(path)
        return predictor

    infer.initialize_predictor = initialize_predictor
    package = ModuleType("audiobox_aesthetics")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "audiobox_aesthetics", package)
    monkeypatch.setitem(sys.modules, "audiobox_aesthetics.infer", infer)
    monkeypatch.setattr(node, "NODE_DIR", tmp_path)
    monkeypatch.delenv("AUDIOBOX_AESTHETICS_CHECKPOINT", raising=False)

    provider = node.AudioBoxProvider()

    assert provider.predictor is predictor
    assert calls == [str(checkpoint)]


def test_audiobox_provider_keeps_online_fallback(monkeypatch, tmp_path):
    import sys
    from types import ModuleType

    from sure_eval.evaluation.nodes.scoring.audiobox_aesthetics import node

    calls = []
    infer = ModuleType("audiobox_aesthetics.infer")
    infer.initialize_predictor = lambda path=None: calls.append(path) or object()
    package = ModuleType("audiobox_aesthetics")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "audiobox_aesthetics", package)
    monkeypatch.setitem(sys.modules, "audiobox_aesthetics.infer", infer)
    monkeypatch.setattr(node, "NODE_DIR", tmp_path)
    monkeypatch.delenv("AUDIOBOX_AESTHETICS_CHECKPOINT", raising=False)

    node.AudioBoxProvider()

    assert calls == [None]


def test_audiobox_soundfile_loader_avoids_torchcodec(monkeypatch):
    import sys
    from types import SimpleNamespace

    import numpy as np

    from sure_eval.evaluation.nodes.scoring.audiobox_aesthetics import node

    original_load = object()
    fake_torchaudio = SimpleNamespace(load=original_load)
    fake_soundfile = SimpleNamespace(
        read=lambda uri, **kwargs: (
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            16000,
        )
    )
    fake_torch = SimpleNamespace(from_numpy=lambda value: value)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    with node._use_soundfile_torchaudio_load():
        waveform, sample_rate = fake_torchaudio.load("audio.wav")

    assert sample_rate == 16000
    assert waveform.shape == (2, 2)
    assert np.allclose(waveform, [[0.1, 0.3], [0.2, 0.4]])
    assert fake_torchaudio.load is original_load
