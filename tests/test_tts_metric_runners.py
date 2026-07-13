from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest


def test_semantic_provider_scores_transcribed_text_with_sure_metrics() -> None:
    from sure_eval.evaluation.nodes.transcription import StaticTranscriber, TTSSemanticErrorRateProvider

    provider = TTSSemanticErrorRateProvider(transcriber=StaticTranscriber("hello brave world"))
    row = provider("hyp.wav", "hello world", language="en", metric="wer")

    assert row["transcript"] == "hello brave world"
    assert row["reference_text"] == "hello world"
    assert row["wer"] > 0
    assert row["score"] == row["wer"]


def test_paraformer_transcriber_disables_online_update_check(monkeypatch) -> None:
    from sure_eval.evaluation.nodes.transcription import ParaformerZHTranscriber

    seen_kwargs: dict[str, object] = {}

    class FakeAutoModel:
        def __init__(self, **kwargs: object) -> None:
            seen_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=FakeAutoModel))

    transcriber = ParaformerZHTranscriber(model_id="paraformer-test", device="cpu")

    assert transcriber._load() is transcriber._model
    assert seen_kwargs["model"] == "paraformer-test"
    assert seen_kwargs["device"] == "cpu"
    assert seen_kwargs["disable_update"] is True


def test_paraformer_transcriber_falls_back_when_disable_update_is_unsupported(monkeypatch) -> None:
    from sure_eval.evaluation.nodes.transcription import ParaformerZHTranscriber

    calls: list[dict[str, object]] = []

    class FakeAutoModel:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)
            if "disable_update" in kwargs:
                raise TypeError("unexpected keyword argument 'disable_update'")

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=FakeAutoModel))

    transcriber = ParaformerZHTranscriber(model_id="paraformer-test", device="cpu")

    assert transcriber._load() is transcriber._model
    assert calls == [
        {"model": "paraformer-test", "device": "cpu", "disable_update": True},
        {"model": "paraformer-test", "device": "cpu"},
    ]


def test_paraformer_transcriber_prefers_local_modelscope_checkpoint(monkeypatch, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.transcription import ParaformerZHTranscriber

    model_dir = (
        tmp_path
        / "modelscope"
        / "models"
        / "iic"
        / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    )
    model_dir.mkdir(parents=True)
    (model_dir / "configuration.json").write_text("{}", encoding="utf-8")
    seen_kwargs: dict[str, object] = {}

    class FakeAutoModel:
        def __init__(self, **kwargs: object) -> None:
            seen_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=FakeAutoModel))

    transcriber = ParaformerZHTranscriber(cache_dir=tmp_path, device="cpu")

    assert transcriber._load() is transcriber._model
    assert seen_kwargs["model"] == str(model_dir)
    assert seen_kwargs["device"] == "cpu"
    assert seen_kwargs["disable_update"] is True


def test_node_local_transcriber_batch_invokes_node_once(monkeypatch) -> None:
    from sure_eval.evaluation.core.types import PipelineNodeResult
    import sure_eval.evaluation.nodes.transcription.common.providers as providers
    from sure_eval.evaluation.nodes.transcription.common.providers import NodeLocalTranscriber

    class FakeRuntime:
        command_prefix = [sys.executable]
        extra_pythonpath: tuple[str, ...] = ()
        inherit_pythonpath = True

    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        input_jsonl = Path(command[command.index("--input-jsonl") + 1])
        rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines()]
        stdout = "\n".join(
            json.dumps(
                {
                    "audio_path": row["audio_path"],
                    "language": row["language"],
                    "transcript": f"transcript-{index}",
                    "trace": {
                        "audio_path": row["audio_path"],
                        "language": row["language"],
                        "role": row["role"],
                        "transcript": f"transcript-{index}",
                    },
                },
                ensure_ascii=False,
            )
            for index, row in enumerate(rows)
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout + "\n", stderr="")

    monkeypatch.setattr(providers, "resolve_node_local_python", lambda *_args: FakeRuntime())
    monkeypatch.setattr(providers.subprocess, "run", fake_run)

    node_dir = Path("src/sure_eval/evaluation/nodes/transcription/paraformer_zh").resolve()
    transcriber = NodeLocalTranscriber(
        node_id="transcription/paraformer_zh",
        node_dir=node_dir,
        device="cuda:0",
    )

    results = transcriber.transcribe_batch(["a.wav", "b.wav"], language="zh", role="prediction_audio")

    assert len(calls) == 1
    assert "--input-jsonl" in calls[0]
    assert "--audio-path" not in calls[0]
    assert [transcript for transcript, _trace in results] == ["transcript-0", "transcript-1"]
    assert all(isinstance(trace, PipelineNodeResult) for _transcript, trace in results)
    assert [trace.details["role"] for _transcript, trace in results] == ["prediction_audio", "prediction_audio"]


def test_node_local_transcriber_batch_can_chunk_invocations(monkeypatch) -> None:
    import sure_eval.evaluation.nodes.transcription.common.providers as providers
    from sure_eval.evaluation.nodes.transcription.common.providers import NodeLocalTranscriber

    class FakeRuntime:
        command_prefix = [sys.executable]
        extra_pythonpath: tuple[str, ...] = ()
        inherit_pythonpath = True

    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        input_jsonl = Path(command[command.index("--input-jsonl") + 1])
        rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines()]
        stdout = "\n".join(
            json.dumps(
                {
                    "audio_path": row["audio_path"],
                    "language": row["language"],
                    "transcript": f"transcript-{row['audio_path']}",
                    "trace": {
                        "audio_path": row["audio_path"],
                        "language": row["language"],
                        "role": row["role"],
                        "transcript": f"transcript-{row['audio_path']}",
                    },
                },
                ensure_ascii=False,
            )
            for row in rows
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout + "\n", stderr="")

    monkeypatch.setenv("SURE_EVAL_NODE_LOCAL_TRANSCRIBE_BATCH_SIZE", "2")
    monkeypatch.setattr(providers, "resolve_node_local_python", lambda *_args: FakeRuntime())
    monkeypatch.setattr(providers.subprocess, "run", fake_run)

    node_dir = Path("src/sure_eval/evaluation/nodes/transcription/whisper_large_v3").resolve()
    transcriber = NodeLocalTranscriber(
        node_id="transcription/whisper_large_v3",
        node_dir=node_dir,
        device="cuda:0",
    )

    results = transcriber.transcribe_batch(["a.wav", "b.wav", "c.wav"], language="en", role="prediction_audio")

    assert len(calls) == 2
    assert [transcript for transcript, _trace in results] == [
        "transcript-a.wav",
        "transcript-b.wav",
        "transcript-c.wav",
    ]


def test_node_local_transcriber_batch_prefers_node_specific_chunk_size(monkeypatch) -> None:
    import sure_eval.evaluation.nodes.transcription.common.providers as providers
    from sure_eval.evaluation.nodes.transcription.common.providers import NodeLocalTranscriber

    class FakeRuntime:
        command_prefix = [sys.executable]
        extra_pythonpath: tuple[str, ...] = ()
        inherit_pythonpath = True

    calls: list[int] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        input_jsonl = Path(command[command.index("--input-jsonl") + 1])
        rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines()]
        calls.append(len(rows))
        stdout = "\n".join(
            json.dumps({"audio_path": row["audio_path"], "language": row["language"], "transcript": row["audio_path"]})
            for row in rows
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout + "\n", stderr="")

    monkeypatch.setenv("SURE_EVAL_NODE_LOCAL_TRANSCRIBE_BATCH_SIZE", "4")
    monkeypatch.setenv("SURE_EVAL_TRANSCRIPTION_BATCH_SIZE_TRANSCRIPTION_WHISPER_LARGE_V3", "2")
    monkeypatch.setattr(providers, "resolve_node_local_python", lambda *_args: FakeRuntime())
    monkeypatch.setattr(providers.subprocess, "run", fake_run)

    transcriber = NodeLocalTranscriber(
        node_id="transcription/whisper_large_v3",
        node_dir=Path("src/sure_eval/evaluation/nodes/transcription/whisper_large_v3").resolve(),
        device="cuda:0",
    )

    results = transcriber.transcribe_batch(["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"], language="en")

    assert calls == [2, 2, 1]
    assert [transcript for transcript, _trace in results] == ["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"]


def test_ecapa_speechbrain_link_conflict_cleanup_only_removes_same_target_symlink(tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import (
        _clear_speechbrain_snapshot_link_conflicts,
    )

    snapshot = tmp_path / "snapshot"
    savedir = tmp_path / "speechbrain"
    snapshot.mkdir()
    savedir.mkdir()
    source = snapshot / "custom.py"
    source.write_text("source", encoding="utf-8")
    same_target = savedir / "custom.py"
    same_target.symlink_to(source)
    ordinary = savedir / "hyperparams.yaml"
    ordinary.write_text("keep", encoding="utf-8")

    _clear_speechbrain_snapshot_link_conflicts(snapshot, savedir)

    assert not same_target.exists()
    assert ordinary.read_text(encoding="utf-8") == "keep"


def test_tts_semantic_pipeline_uses_batch_transcriber_when_available() -> None:
    from sure_eval.evaluation.core.types import PipelineNodeResult
    from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples
    from sure_eval.evaluation.tasks.tts.types import TTSSample

    class BatchOnlyTranscriber:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], str, str]] = []

        def transcribe(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("semantic evaluation should use transcribe_batch")

        def transcribe_batch(
            self,
            audio_paths: list[str],
            *,
            language: str = "en",
            role: str = "prediction_audio",
        ) -> list[tuple[str, PipelineNodeResult]]:
            self.calls.append((list(audio_paths), language, role))
            return [
                (
                    transcript,
                    PipelineNodeResult(
                        stage="transcription",
                        node_id="transcription/paraformer_zh",
                        version="test",
                        details={
                            "audio_path": audio_path,
                            "language": language,
                            "role": role,
                            "transcript": transcript,
                        },
                    ),
                )
                for audio_path, transcript in zip(audio_paths, ["你好世界", "测试文本"], strict=True)
            ]

    transcriber = BatchOnlyTranscriber()
    samples = [
        TTSSample(
            sample_id="utt1",
            prediction_audio="a.wav",
            reference_text="你好世界",
            language="zh",
        ),
        TTSSample(
            sample_id="utt2",
            prediction_audio="b.wav",
            reference_text="测试文本",
            language="zh",
        ),
    ]

    report = evaluate_tts_samples(
        samples,
        metrics=("tts_cer",),
        transcribers={"zh": transcriber},
    )

    assert transcriber.calls == [(["a.wav", "b.wav"], "zh", "prediction_audio")]
    assert report.details["results"]["tts_cer"]["score"] == 0.0


def test_tts_mos_pipeline_runs_when_zip_lacks_strict_keyword(monkeypatch) -> None:
    import builtins
    from sure_eval.evaluation.tasks.tts import pipeline as tts_pipeline
    from sure_eval.evaluation.tasks.tts.pipeline import evaluate_tts_samples
    from sure_eval.evaluation.tasks.tts.types import TTSSample

    def py38_zip(*iterables):
        return builtins.zip(*iterables)

    monkeypatch.setattr(tts_pipeline, "zip", py38_zip, raising=False)
    report = evaluate_tts_samples(
        [
            TTSSample(
                sample_id="utt1",
                prediction_audio="a.wav",
                reference_text="hello",
                language="en",
            )
        ],
        metrics=("utmos",),
        mos_providers={"utmos": lambda prediction, reference="", **kwargs: {"utmos": 4.2}},
    )

    assert report.details["results"]["utmos"]["score"] == 4.2
    assert report.details["rows"][0]["mos"]["utmos"]["utmos"] == 4.2


def test_node_local_speaker_provider_batch_invokes_node_once(monkeypatch) -> None:
    import sure_eval.evaluation.nodes.scoring.common.node_local as node_local
    from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalSpeakerProvider

    calls: list[tuple[str, list[str]]] = []

    def fake_run_node_json(*, node_id: str, node_dir: Path, module_name: str, args: list[str]) -> dict[str, object]:
        calls.append((module_name, args))
        input_jsonl = Path(args[args.index("--input-jsonl") + 1])
        rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines()]
        return {
            "node_id": node_id,
            "result": {
                "per_sample": [
                    {"ASV": 0.75 + index, "backend": "fake-speaker"}
                    for index, _row in enumerate(rows)
                ]
            },
        }

    monkeypatch.setattr(node_local, "_run_node_json", fake_run_node_json)

    provider = NodeLocalSpeakerProvider(
        node_id="scoring/wavlm_large_sim",
        node_dir=Path("src/sure_eval/evaluation/nodes/scoring/wavlm_large_sim"),
        device="cuda:0",
    )

    rows = provider.score_batch(
        [("utt1", "hyp1.wav", "ref1.wav"), ("utt2", "hyp2.wav", "ref2.wav")],
        metric_name="sim/wavlm-large",
    )

    assert len(calls) == 1
    assert "--input-jsonl" in calls[0][1]
    assert "--prediction-audio" not in calls[0][1]
    assert [row["ASV"] for row in rows] == [0.75, 1.75]


def test_node_local_mos_provider_chunks_batches(monkeypatch) -> None:
    import sure_eval.evaluation.nodes.scoring.common.node_local as node_local
    from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalMOSProvider

    calls: list[list[str]] = []

    def fake_run_node_json(*, node_id: str, node_dir: Path, module_name: str, args: list[str]) -> dict[str, object]:
        calls.append(args)
        input_jsonl = Path(args[args.index("--input-jsonl") + 1])
        rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines()]
        return {
            "node_id": node_id,
            "result": {
                "per_sample": [
                    {"OVRL": 3.0 + index, "backend": "fake-dnsmos"}
                    for index, _row in enumerate(rows)
                ]
            },
        }

    monkeypatch.setattr(node_local, "_run_node_json", fake_run_node_json)
    monkeypatch.setenv("SURE_EVAL_NODE_LOCAL_MOS_BATCH_SIZE", "2")

    provider = NodeLocalMOSProvider(
        node_id="scoring/dnsmos",
        node_dir=Path("src/sure_eval/evaluation/nodes/scoring/dnsmos"),
        device="cuda:0",
    )
    rows = provider.score_batch(
        [
            ("utt1", "hyp1.wav"),
            ("utt2", "hyp2.wav"),
            ("utt3", "hyp3.wav"),
            ("utt4", "hyp4.wav"),
            ("utt5", "hyp5.wav"),
        ],
        metric_name="dnsmos",
    )

    assert len(calls) == 3
    assert len(rows) == 5
    assert all("--input-jsonl" in call for call in calls)


def test_node_local_eres2net_provider_chunks_batches(monkeypatch) -> None:
    import sure_eval.evaluation.nodes.scoring.common.node_local as node_local
    from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalSpeakerProvider

    calls: list[list[str]] = []

    def fake_run_node_json(*, node_id: str, node_dir: Path, module_name: str, args: list[str]) -> dict[str, object]:
        calls.append(args)
        input_jsonl = Path(args[args.index("--input-jsonl") + 1])
        rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines()]
        return {
            "node_id": node_id,
            "result": {
                "per_sample": [
                    {"ASV": float(index), "backend": "fake-eres2net"}
                    for index, _row in enumerate(rows)
                ]
            },
        }

    monkeypatch.setattr(node_local, "_run_node_json", fake_run_node_json)
    monkeypatch.setenv("SURE_EVAL_ERES2NET_BATCH_SIZE", "2")

    provider = NodeLocalSpeakerProvider(
        node_id="scoring/eres2net_sim",
        node_dir=Path("src/sure_eval/evaluation/nodes/scoring/eres2net_sim"),
        device="cuda:0",
    )
    rows = provider.score_batch(
        [
            ("utt1", "hyp1.wav", "ref1.wav"),
            ("utt2", "hyp2.wav", "ref2.wav"),
            ("utt3", "hyp3.wav", "ref3.wav"),
            ("utt4", "hyp4.wav", "ref4.wav"),
            ("utt5", "hyp5.wav", "ref5.wav"),
        ],
        metric_name="sim/eres2net",
    )

    assert len(calls) == 3
    assert len(rows) == 5


def test_node_local_eres2net_provider_default_does_not_chunk(monkeypatch) -> None:
    import sure_eval.evaluation.nodes.scoring.common.node_local as node_local
    from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalSpeakerProvider

    calls: list[int] = []

    def fake_run_node_json(*, node_id: str, node_dir: Path, module_name: str, args: list[str]) -> dict[str, object]:
        input_jsonl = Path(args[args.index("--input-jsonl") + 1])
        rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines()]
        calls.append(len(rows))
        return {
            "node_id": node_id,
            "result": {
                "per_sample": [
                    {"ASV": float(index), "backend": "fake-eres2net"}
                    for index, _row in enumerate(rows)
                ]
            },
        }

    monkeypatch.setattr(node_local, "_run_node_json", fake_run_node_json)
    monkeypatch.delenv("SURE_EVAL_ERES2NET_BATCH_SIZE", raising=False)

    provider = NodeLocalSpeakerProvider(
        node_id="scoring/eres2net_sim",
        node_dir=Path("src/sure_eval/evaluation/nodes/scoring/eres2net_sim"),
        device="cuda:0",
    )
    rows = provider.score_batch(
        [(f"utt{index}", f"hyp{index}.wav", f"ref{index}.wav") for index in range(10)],
        metric_name="sim/eres2net",
    )

    assert calls == [10]
    assert len(rows) == 10


def test_node_local_eres2net_provider_recovers_cuda_oom_with_cpu_retry(monkeypatch) -> None:
    import sure_eval.evaluation.nodes.scoring.common.node_local as node_local
    from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalSpeakerProvider

    calls: list[tuple[str, int]] = []

    def fake_run_node_json(*, node_id: str, node_dir: Path, module_name: str, args: list[str]) -> dict[str, object]:
        device = args[args.index("--device") + 1]
        input_jsonl = Path(args[args.index("--input-jsonl") + 1])
        rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines()]
        calls.append((device, len(rows)))
        if device.startswith("cuda"):
            raise RuntimeError("scoring/eres2net_sim scoring failed: torch.cuda.OutOfMemoryError")
        return {
            "node_id": node_id,
            "result": {
                "per_sample": [
                    {"ASV": float(index), "backend": "fake-eres2net-cpu"}
                    for index, _row in enumerate(rows)
                ]
            },
        }

    monkeypatch.setattr(node_local, "_run_node_json", fake_run_node_json)
    monkeypatch.setenv("SURE_EVAL_ERES2NET_BATCH_SIZE", "2")
    monkeypatch.setenv("SURE_EVAL_ERES2NET_ALLOW_CPU_FALLBACK", "1")

    provider = NodeLocalSpeakerProvider(
        node_id="scoring/eres2net_sim",
        node_dir=Path("src/sure_eval/evaluation/nodes/scoring/eres2net_sim"),
        device="cuda:0",
    )
    rows = provider.score_batch(
        [("utt1", "hyp1.wav", "ref1.wav"), ("utt2", "hyp2.wav", "ref2.wav")],
        metric_name="sim/eres2net",
    )

    assert rows == [
        {"ASV": 0.0, "backend": "fake-eres2net-cpu"},
        {"ASV": 0.0, "backend": "fake-eres2net-cpu"},
    ]
    assert calls == [("cuda:0", 2), ("cuda:0", 1), ("cpu", 1), ("cuda:0", 1), ("cpu", 1)]


def test_node_local_eres2net_provider_rejects_cpu_retry_by_default(monkeypatch) -> None:
    import sure_eval.evaluation.nodes.scoring.common.node_local as node_local
    from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalSpeakerProvider

    calls: list[str] = []

    def fake_run_node_json(*, node_id: str, node_dir: Path, module_name: str, args: list[str]) -> dict[str, object]:
        device = args[args.index("--device") + 1]
        calls.append(device)
        raise RuntimeError("scoring/eres2net_sim scoring failed: torch.cuda.OutOfMemoryError")

    monkeypatch.setattr(node_local, "_run_node_json", fake_run_node_json)
    monkeypatch.delenv("SURE_EVAL_ERES2NET_ALLOW_CPU_FALLBACK", raising=False)

    provider = NodeLocalSpeakerProvider(
        node_id="scoring/eres2net_sim",
        node_dir=Path("src/sure_eval/evaluation/nodes/scoring/eres2net_sim"),
        device="cuda:0",
    )

    with pytest.raises(RuntimeError, match="OutOfMemoryError"):
        provider.score_batch(
            [("utt1", "hyp1.wav", "ref1.wav")],
            metric_name="sim/eres2net",
        )

    assert calls == ["cuda:0"]


def test_node_local_mos_provider_batch_invokes_node_once(monkeypatch) -> None:
    import sure_eval.evaluation.nodes.scoring.common.node_local as node_local
    from sure_eval.evaluation.nodes.scoring.common.node_local import NodeLocalMOSProvider

    calls: list[tuple[str, list[str]]] = []

    def fake_run_node_json(*, node_id: str, node_dir: Path, module_name: str, args: list[str]) -> dict[str, object]:
        calls.append((module_name, args))
        input_jsonl = Path(args[args.index("--input-jsonl") + 1])
        rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines()]
        return {
            "node_id": node_id,
            "result": {
                "per_sample": [
                    {"mos": 3.0 + index, "backend": "fake-mos"}
                    for index, _row in enumerate(rows)
                ]
            },
        }

    monkeypatch.setattr(node_local, "_run_node_json", fake_run_node_json)

    provider = NodeLocalMOSProvider(
        node_id="scoring/wv_mos",
        node_dir=Path("src/sure_eval/evaluation/nodes/scoring/wv_mos"),
        device="cuda:0",
    )

    rows = provider.score_batch(
        [("utt1", "hyp1.wav"), ("utt2", "hyp2.wav")],
        metric_name="wv-mos",
    )

    assert len(calls) == 1
    assert "--input-jsonl" in calls[0][1]
    assert "--prediction-audio" not in calls[0][1]
    assert [row["mos"] for row in rows] == [3.0, 4.0]


def test_node_local_python_allows_node_specific_interpreter_override(monkeypatch, tmp_path: Path) -> None:
    from sure_eval.evaluation.nodes.common.node_local_python import resolve_node_local_python

    node_dir = tmp_path / "utmos"
    python_bin = node_dir / ".venv" / "bin" / "python"
    override = tmp_path / "python3.8"
    python_bin.parent.mkdir(parents=True)
    python_bin.symlink_to("/missing/container/python3.8")
    override.write_text("#!/bin/sh\n", encoding="utf-8")
    override.chmod(0o755)
    monkeypatch.setenv("SURE_EVAL_NODE_LOCAL_PYTHON_SCORING_UTMOS", str(override))

    runtime = resolve_node_local_python(node_dir, "scoring/utmos")

    assert runtime.command_prefix == (str(override),)
    assert runtime.inherit_pythonpath is False
    assert runtime.isolated is False


def test_audio_scoring_backends_use_batch_provider_when_available() -> None:
    from sure_eval.evaluation.nodes.scoring._audio_quality import score_mos_backend, score_speaker_backend

    class BatchSpeakerProvider:
        def __init__(self) -> None:
            self.calls: list[list[tuple[str, str, str]]] = []

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, float]:
            raise AssertionError("speaker backend should use score_batch")

        def score_batch(
            self,
            rows: list[tuple[str, str, str]],
            *,
            metric_name: str,
        ) -> list[dict[str, float]]:
            self.calls.append(list(rows))
            return [{"ASV": 0.2}, {"ASV": 0.6}]

    class BatchMOSProvider:
        def __init__(self) -> None:
            self.calls: list[list[tuple[str, str]]] = []

        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, float]:
            raise AssertionError("MOS backend should use score_batch")

        def score_batch(
            self,
            rows: list[tuple[str, str]],
            *,
            metric_name: str,
        ) -> list[dict[str, float]]:
            self.calls.append(list(rows))
            return [{"mos": 2.0}, {"mos": 4.0}]

    speaker_provider = BatchSpeakerProvider()
    speaker_result = score_speaker_backend(
        [("utt1", "hyp1.wav", "ref1.wav"), ("utt2", "hyp2.wav", "ref2.wav")],
        backend_name="wavlm-large",
        metric_name="sim/wavlm-large",
        node_id="scoring/wavlm_large_sim",
        provider=speaker_provider,
    )
    mos_provider = BatchMOSProvider()
    mos_result = score_mos_backend(
        [("utt1", "hyp1.wav"), ("utt2", "hyp2.wav")],
        metric_name="wv-mos",
        node_id="scoring/wv_mos",
        provider=mos_provider,
    )

    assert speaker_provider.calls == [[("utt1", "hyp1.wav", "ref1.wav"), ("utt2", "hyp2.wav", "ref2.wav")]]
    assert speaker_result.details["result"]["score"] == 0.4
    assert mos_provider.calls == [[("utt1", "hyp1.wav"), ("utt2", "hyp2.wav")]]
    assert mos_result.details["result"]["score"] == 3.0


def test_speaker_cosine_provider_scores_embeddings() -> None:
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import EmbeddingSpeakerSimilarityProvider

    provider = EmbeddingSpeakerSimilarityProvider(
        embedder=lambda path: [1.0, 0.0] if "same" in path else [0.0, 1.0]
    )

    same = provider("same_hyp.wav", "same_ref.wav")
    different = provider("diff_hyp.wav", "same_ref.wav")

    assert same["ASV"] == 1.0
    assert math.isclose(different["ASV"], 0.0, abs_tol=1e-7)


def test_eres2net_similarity_extracts_common_score_shapes() -> None:
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import ERes2NetSimilarityProvider

    assert ERes2NetSimilarityProvider._extract_score({"score": 0.76}) == 0.76
    assert ERes2NetSimilarityProvider._extract_score([{"similarity": 0.66}]) == 0.66


def test_eres2net_similarity_falls_back_to_embedding_cosine() -> None:
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import ERes2NetSimilarityProvider

    provider = ERes2NetSimilarityProvider(
        pipeline_factory=lambda: (_ for _ in ()).throw(OSError("libsox.so")),
        embedding_provider=lambda path: [1.0, 0.0] if "ref" in path else [0.0, 1.0],
    )

    row = provider("hyp.wav", "ref.wav")

    assert row["ASV"] == 0.0
    assert row["backend"] == "modelscope-eres2net-embedding-cosine"


def test_eres2net_similarity_uses_loaded_pipeline_embeddings_on_cuda_oom() -> None:
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import ERes2NetSimilarityProvider

    calls: list[tuple[object, bool]] = []

    def fake_pipeline(payload: object, output_emb: bool = False) -> dict[str, object]:
        calls.append((payload, output_emb))
        if not output_emb:
            raise RuntimeError("torch.cuda.OutOfMemoryError")
        return {
            "embs": [1.0, 0.0] if payload == ["ref.wav"] else [0.0, 1.0],
        }

    provider = ERes2NetSimilarityProvider(pipeline_factory=lambda: fake_pipeline)

    row = provider("hyp.wav", "ref.wav")

    assert row["ASV"] == 0.0
    assert row["backend"] == "modelscope-eres2net-same-pipeline-embedding-cosine"
    assert calls == [
        (["ref.wav", "hyp.wav"], False),
        (["hyp.wav"], True),
        (["ref.wav"], True),
    ]


def test_eres2net_similarity_can_force_embedding_cosine(monkeypatch) -> None:
    import sys
    import types

    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import ERes2NetSimilarityProvider

    calls: list[tuple[object, bool]] = []
    state = {"enabled": False, "entered": False}

    class FakeInferenceMode:
        def __enter__(self) -> None:
            state["enabled"] = True
            state["entered"] = True

        def __exit__(self, *_args: object) -> None:
            state["enabled"] = False

    fake_torch = types.SimpleNamespace(inference_mode=lambda: FakeInferenceMode())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def fake_pipeline(payload: object, output_emb: bool = False) -> dict[str, object]:
        assert state["enabled"]
        calls.append((payload, output_emb))
        return {
            "embs": [1.0, 0.0] if payload == ["ref.wav"] else [0.0, 1.0],
        }

    monkeypatch.setenv("SURE_EVAL_ERES2NET_FORCE_EMBEDDING_COSINE", "1")
    provider = ERes2NetSimilarityProvider(pipeline_factory=lambda: fake_pipeline)

    row = provider("hyp.wav", "ref.wav")

    assert row["ASV"] == 0.0
    assert row["backend"] == "modelscope-eres2net-same-pipeline-embedding-cosine"
    assert calls == [
        (["hyp.wav"], True),
        (["ref.wav"], True),
    ]
    assert state["entered"]


def test_eres2net_similarity_segments_embeddings_when_single_audio_oom() -> None:
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import ERes2NetSimilarityProvider

    calls: list[tuple[object, bool]] = []

    class FakeProvider(ERes2NetSimilarityProvider):
        def _load_audio_segments(self, audio_path: str) -> list[object]:
            return [f"{audio_path}:0", f"{audio_path}:1"]

    def fake_pipeline(payload: object, output_emb: bool = False) -> dict[str, object]:
        calls.append((payload, output_emb))
        if not output_emb:
            raise RuntimeError("torch.cuda.OutOfMemoryError")
        if payload in (["hyp.wav"], ["ref.wav"]):
            raise RuntimeError("torch.cuda.OutOfMemoryError")
        return {
            "embs": [0.0, 1.0] if str(payload).startswith("['hyp") else [1.0, 0.0],
        }

    provider = FakeProvider(pipeline_factory=lambda: fake_pipeline)

    row = provider("hyp.wav", "ref.wav")

    assert math.isclose(row["ASV"], 0.0, abs_tol=1e-7)
    assert row["backend"] == "modelscope-eres2net-same-pipeline-embedding-cosine"
    assert calls == [
        (["ref.wav", "hyp.wav"], False),
        (["hyp.wav"], True),
        (["hyp.wav:0"], True),
        (["hyp.wav:1"], True),
        (["ref.wav"], True),
        (["ref.wav:0"], True),
        (["ref.wav:1"], True),
    ]


def test_eres2net_similarity_does_not_load_second_model_after_embedding_oom() -> None:
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import ERes2NetSimilarityProvider

    calls: list[object] = []

    def fake_pipeline(payload: object, output_emb: bool = False) -> dict[str, object]:
        calls.append(payload)
        raise RuntimeError("torch.cuda.OutOfMemoryError")

    def embedding_provider(_path: str) -> list[float]:
        raise AssertionError("second ERes2Net embedding provider must not be loaded after CUDA OOM")

    provider = ERes2NetSimilarityProvider(
        pipeline_factory=lambda: fake_pipeline,
        embedding_provider=embedding_provider,
    )

    with pytest.raises(RuntimeError, match="OOM recovery failed"):
        provider("hyp.wav", "ref.wav")

    assert calls == [["ref.wav", "hyp.wav"], ["hyp.wav"]]


def test_eres2net_similarity_retries_with_preprocessed_audio_on_sox_error(monkeypatch) -> None:
    from sure_eval.evaluation.nodes.scoring.common.speaker_providers import ERes2NetSimilarityProvider

    calls: list[object] = []

    class FakeProvider(ERes2NetSimilarityProvider):
        def _preprocess_audio_for_modelscope(self, audio_path: str) -> str:
            return f"prepared:{audio_path}"

    def fake_pipeline(payload: object) -> dict[str, float]:
        calls.append(payload)
        if payload == ["ref.wav", "hyp.wav"]:
            raise OSError("libsox.so: cannot open shared object file")
        return {"score": 0.72}

    provider = FakeProvider(pipeline_factory=lambda: fake_pipeline)

    row = provider("hyp.wav", "ref.wav")

    assert row["ASV"] == 0.72
    assert row["backend"] == "modelscope-eres2net-cosine-preprocessed"
    assert calls == [["ref.wav", "hyp.wav"], ["prepared:ref.wav", "prepared:hyp.wav"]]


def test_eres2net_provider_stubs_deepspeed_before_modelscope_import(monkeypatch) -> None:
    import builtins

    import sure_eval.evaluation.nodes.scoring.common.speaker_providers as providers

    calls: list[str] = []
    real_import = builtins.__import__
    pipeline_module = types.ModuleType("modelscope.pipelines")
    constant_module = types.ModuleType("modelscope.utils.constant")

    def fake_pipeline(**_: object):
        return object()

    pipeline_module.pipeline = fake_pipeline
    constant_module.Tasks = types.SimpleNamespace(speaker_verification="speaker_verification")

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("modelscope"):
            assert calls == ["stub"]
            if name == "modelscope.pipelines":
                return pipeline_module
            if name == "modelscope.utils.constant":
                return constant_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(providers, "install_deepspeed_stub", lambda: calls.append("stub"))
    monkeypatch.setattr(builtins, "__import__", fake_import)

    provider = providers.ERes2NetSimilarityProvider(device="cpu")

    assert provider._load() is not None


def test_mos_command_provider_normalizes_json_and_scalar_outputs() -> None:
    from sure_eval.evaluation.nodes.scoring.common.mos_providers import CommandMOSProvider

    provider = CommandMOSProvider(command_runner=lambda _cmd: '{"OVRL": 3.2, "SIG": 3.6}')
    assert provider("hyp.wav", "", metric="dnsmos")["OVRL"] == 3.2

    scalar_provider = CommandMOSProvider(command_runner=lambda _cmd: "4.1\n")
    assert scalar_provider("hyp.wav", "", metric="utmos")["utmos"] == 4.1


def test_dnsmos_provider_uses_injected_cv3_scorer() -> None:
    from sure_eval.evaluation.nodes.scoring.common.mos_providers import DNSMOSProvider

    provider = DNSMOSProvider(
        scorer=lambda audio_path: {
            "OVRL": 3.2,
            "SIG": 3.4,
            "BAK": 3.1,
            "P808_MOS": 3.3,
            "filename": audio_path,
        }
    )

    row = provider("hyp.wav")

    assert row["OVRL"] == 3.2
    assert row["SIG"] == 3.4
    assert row["P808_MOS"] == 3.3
    assert row["backend"] == "cv3-eval-dnsmos"


def test_tts_audio_metrics_drop_runtime_performance_fields() -> None:
    from sure_eval.evaluation.tasks.tts.metrics import DNSMOSMetric, SIMMetric

    sim = SIMMetric(
        score_provider=lambda prediction, reference: {
            "ASV": 0.7,
            "rtf": 0.04,
            "latency_ms": 123,
            "duration_seconds": 2.0,
            "prediction_audio": prediction,
            "reference_audio": reference,
            "backend": "unit-test",
        }
    )
    mos = DNSMOSMetric(
        score_provider=lambda prediction, reference: {
            "OVRL": 3.2,
            "SIG": 3.4,
            "BAK": 3.1,
            "filename": prediction,
            "len_in_sec": 2.0,
            "sr": 16000,
            "num_hops": 1,
            "elapsed_seconds": 0.3,
            "throughput": 9.9,
        }
    )

    sim_row = sim.calculate("hyp.wav", "ref.wav").details["per_sample"][0]
    mos_report = mos.calculate("hyp.wav", "")
    mos_row = mos_report.details["per_sample"][0]

    assert sim_row["ASV"] == 0.7
    assert sim_row["backend"] == "unit-test"
    assert "rtf" not in sim_row
    assert "latency_ms" not in sim_row
    assert "duration_seconds" not in sim_row
    assert "prediction_audio" not in sim_row
    assert "reference_audio" not in sim_row
    assert mos_row["OVRL"] == 3.2
    assert mos_report.details["mean_SIG"] == 3.4
    assert "filename" not in mos_row
    assert "len_in_sec" not in mos_row
    assert "sr" not in mos_row
    assert "num_hops" not in mos_row
    assert "elapsed_seconds" not in mos_row
    assert "throughput" not in mos_row


def test_named_mos_providers_build_commands_and_normalize_outputs() -> None:
    from sure_eval.evaluation.nodes.scoring.common.mos_providers import DNSMOSProvider, UTMOSProvider, WVMOSProvider

    seen: list[list[str]] = []

    def runner(command: list[str]) -> str:
        seen.append(command)
        if command[0] == "dnsmos":
            return '{"OVRL": 3.4, "SIG": 3.5, "BAK": 3.6}'
        if command[0] == "wv-mos":
            return "3.7"
        return '{"utmos": 3.8}'

    assert DNSMOSProvider(command_runner=runner)("hyp.wav")["OVRL"] == 3.4
    assert WVMOSProvider(command=["wv-mos", "{audio_path}"], command_runner=runner)("hyp.wav")["mos"] == 3.7
    assert UTMOSProvider(command=["utmos", "{audio_path}"], command_runner=runner)("hyp.wav")["utmos"] == 3.8
    assert seen == [
        ["dnsmos", "hyp.wav"],
        ["wv-mos", "hyp.wav"],
        ["utmos", "hyp.wav"],
    ]


def test_wvmos_provider_disables_cudnn_for_cuda_inference(monkeypatch, tmp_path: Path) -> None:
    import types

    from sure_eval.evaluation.nodes.scoring.common.mos_providers import WVMOSProvider

    class FakeModel:
        def __init__(self) -> None:
            self.devices: list[str] = []
            self.eval_called = False

        def to(self, device: str) -> "FakeModel":
            self.devices.append(device)
            return self

        def eval(self) -> None:
            self.eval_called = True

        def calculate_one(self, prediction: str, device: str) -> float:
            assert prediction == "hyp.wav"
            assert device == "cuda:0"
            assert fake_torch.backends.cudnn.enabled is False
            return 4.2

    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(enabled=True)),
    )

    provider = WVMOSProvider(
        repo_dir=tmp_path,
        checkpoint_path=tmp_path / "wav2vec2.ckpt",
        device="cuda:0",
    )
    model = FakeModel()
    monkeypatch.setattr(provider, "_load_model", lambda: model)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    row = provider("hyp.wav")

    assert row["mos"] == 4.2
    assert model.devices == []
    assert fake_torch.backends.cudnn.enabled is True


def test_tts_metric_pipeline_connects_semantic_speaker_and_mos() -> None:
    from sure_eval.evaluation.tasks.tts.compat import TTSMetricPipeline, TTSSample

    class EchoTranscriber:
        def transcribe(self, audio_path: str, *, language: str = "en") -> str:
            return "hello world" if language == "en" else "你好世界"

    samples = [
        TTSSample(
            prediction_audio="en.wav",
            reference_text="hello world",
            reference_audio="en_ref.wav",
            language="en",
        ),
        TTSSample(
            prediction_audio="zh.wav",
            reference_text="你好世界",
            reference_audio="zh_ref.wav",
            language="zh",
        ),
    ]
    pipeline = TTSMetricPipeline(
        semantic_transcribers={"en": EchoTranscriber(), "zh": EchoTranscriber()},
        speaker_provider=lambda prediction, reference, **kwargs: {"ASV": 0.9},
        mos_providers={
            "dnsmos": lambda prediction, reference="", **kwargs: {"OVRL": 3.1},
            "wv-mos": lambda prediction, reference="", **kwargs: {"mos": 3.2},
            "utmos": lambda prediction, reference="", **kwargs: {"utmos": 3.3},
        },
    )

    report = pipeline.evaluate(samples)

    assert set(report.results) == {"tts_wer", "tts_cer", "sim/wavlm-large", "sim", "dnsmos", "wv-mos", "utmos"}
    assert report.results["tts_wer"].score == 0.0
    assert report.results["tts_cer"].score == 0.0
    assert report.results["sim"].score == 0.9
    assert report.results["dnsmos"].score == 3.1
    assert report.results["wv-mos"].score == 3.2
    assert report.results["utmos"].score == 3.3
    assert (
        report.results["tts_wer"].details["pipeline_id"]
        == "tts.en.tts_wer.whisper_large_v3.whisper_norm.wenet_wer"
    )
    assert (
        report.results["tts_cer"].details["pipeline_id"]
        == "tts.zh.tts_cer.funasr_loader_16k_mono.paraformer_zh.punctuation_strip_norm.wenet_cer"
    )
    assert (
        report.results["tts_cer"].details["pipeline_trace"][2]["node_id"]
        == "normalization/punctuation_strip_norm"
    )
    assert report.results["sim/wavlm-large"].details["pipeline_trace"][0]["node_id"] == "scoring/wavlm_large_sim"
    assert report.results["dnsmos"].details["pipeline_trace"][0]["node_id"] == "scoring/dnsmos"
    assert report.rows[0]["semantic"]["metric"] == "tts_wer"
    assert report.rows[0]["semantic"]["asr_metric"] == "wer"
    assert report.rows[1]["semantic"]["metric"] == "tts_cer"
    assert report.rows[1]["semantic"]["asr_metric"] == "cer"


def test_tts_metric_pipeline_forwards_explicit_semantic_normalizer(monkeypatch) -> None:
    from sure_eval.evaluation.core.types import PipelineNodeResult
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline
    from sure_eval.evaluation.tasks.tts.compat import TTSMetricPipeline, TTSSample

    def fake_wetext(files, *, profile: str):
        return (
            files,
            PipelineNodeResult(
                stage="normalization",
                node_id="normalization/wetext_norm",
                version="v1",
                details={"profile": profile},
                internal_stages=("fake_wetext",),
            ),
        )

    class EchoTranscriber:
        def transcribe(self, audio_path: str, *, language: str = "en") -> str:
            return "你好世界"

    monkeypatch.setattr(asr_pipeline, "normalize_wetext_key_text_files", fake_wetext)
    pipeline = TTSMetricPipeline(
        semantic_transcribers={"zh": EchoTranscriber()},
        semantic_normalizer="wetext:zh_tn",
    )

    report = pipeline.evaluate(
        [
            TTSSample(
                prediction_audio="zh.wav",
                reference_text="你好世界",
                reference_audio="zh_ref.wav",
                language="zh",
            )
        ]
    )

    assert report.results["tts_cer"].details["pipeline_id"] == (
        "tts.zh.tts_cer.funasr_loader_16k_mono.paraformer_zh.wetext_zh_tn.wenet_cer"
    )
    assert report.results["tts_cer"].details["pipeline_trace"][2]["node_id"] == "normalization/wetext_norm"
    assert report.results["tts_cer"].details["pipeline_trace"][2]["profile"] == "zh_tn"


def test_tts_metric_pipeline_supports_named_speaker_backends() -> None:
    from sure_eval.evaluation.tasks.tts.compat import TTSMetricPipeline, TTSSample

    sample = TTSSample(
        prediction_audio="hyp.wav",
        reference_text="hello",
        reference_audio="ref.wav",
    )
    pipeline = TTSMetricPipeline(
        speaker_providers={
            "wavlm-large": lambda prediction, reference, **kwargs: {"ASV": 0.7},
            "ecapa-tdnn": lambda prediction, reference, **kwargs: {"ASV": 0.9},
        }
    )

    report = pipeline.evaluate([sample])

    assert report.results["sim"].score == 0.8
    assert report.results["sim/wavlm-large"].score == 0.7
    assert report.results["sim/ecapa-tdnn"].score == 0.9
    assert report.rows[0]["speaker"]["wavlm-large"]["ASV"] == 0.7
    assert report.rows[0]["speaker"]["ecapa-tdnn"]["ASV"] == 0.9


def test_build_default_tts_metric_pipeline_wires_expected_backends() -> None:
    from sure_eval.evaluation.tasks.tts.compat import build_default_tts_metric_pipeline

    pipeline = build_default_tts_metric_pipeline(device="cpu", cache_dir="/tmp/sure-eval-tts")

    assert set(pipeline.semantic_transcribers) == {"en", "zh"}
    assert set(pipeline.speaker_providers) == {"wavlm-large", "ecapa-tdnn", "eres2net"}
    assert set(pipeline.mos_providers) == {"dnsmos", "wv-mos", "utmos"}


def test_tts_pipeline_cli_runs_with_stub_backend(tmp_path) -> None:
    output = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_tts_metric_pipeline.py",
            "--prediction-audio",
            "hyp.wav",
            "--reference-text",
            "你好世界",
            "--reference-audio",
            "ref.wav",
            "--language",
            "zh",
            "--stub",
            "--output",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["sample"]["language"] == "zh"
    assert report["metrics"]["tts_cer"]["score"] == 0.0
    assert report["metrics"]["sim"]["score"] == 0.42
    assert report["metrics"]["dnsmos"]["score"] == 3.0
    assert report["rows"][0]["semantic"]["metric"] == "tts_cer"


def test_tts_pipeline_cli_runner_keeps_successful_backend_results() -> None:
    import importlib.util
    from pathlib import Path

    from sure_eval.evaluation.tasks.tts.compat import TTSMetricPipeline, TTSSample

    spec = importlib.util.spec_from_file_location(
        "run_tts_metric_pipeline",
        Path("scripts/run_tts_metric_pipeline.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    sample = TTSSample(
        prediction_audio="hyp.wav",
        reference_text="hello",
        reference_audio="ref.wav",
    )
    pipeline = TTSMetricPipeline(
        speaker_providers={
            "wavlm-large": lambda prediction, reference, **kwargs: {"ASV": 0.7},
            "ecapa-tdnn": lambda prediction, reference, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        }
    )

    report = module._run_one(pipeline, sample, fail_fast=False)

    assert report["ok"] is False
    assert report["metrics"]["sim/wavlm-large"]["score"] == 0.7
    assert report["errors"][0]["stage"] == "speaker/ecapa-tdnn"


def test_tts_pipeline_cli_runner_forwards_semantic_normalizer(monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    from sure_eval.evaluation.core.types import PipelineNodeResult
    from sure_eval.evaluation.tasks.asr import pipeline as asr_pipeline
    from sure_eval.evaluation.tasks.tts.compat import TTSMetricPipeline, TTSSample

    spec = importlib.util.spec_from_file_location(
        "run_tts_metric_pipeline",
        Path("scripts/run_tts_metric_pipeline.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def fake_wetext(files, *, profile: str):
        return (
            files,
            PipelineNodeResult(
                stage="normalization",
                node_id="normalization/wetext_norm",
                version="v1",
                details={"profile": profile},
                internal_stages=("fake_wetext",),
            ),
        )

    class EchoTranscriber:
        def transcribe(self, audio_path: str, *, language: str = "en") -> str:
            return "你好世界"

    monkeypatch.setattr(asr_pipeline, "normalize_wetext_key_text_files", fake_wetext)
    sample = TTSSample(
        prediction_audio="hyp.wav",
        reference_text="你好世界",
        reference_audio="ref.wav",
        language="zh",
    )
    pipeline = TTSMetricPipeline(semantic_transcribers={"zh": EchoTranscriber()})

    report = module._run_one(
        pipeline,
        sample,
        fail_fast=False,
        semantic_normalizer="wetext:zh_tn",
    )

    assert report["ok"] is True
    assert report["metrics"]["tts_cer"]["details"]["pipeline_id"] == (
        "tts.zh.tts_cer.funasr_loader_16k_mono.paraformer_zh.wetext_zh_tn.wenet_cer"
    )


def test_tts_pipeline_cli_builds_only_requested_real_backends(monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_tts_metric_pipeline",
        Path("scripts/run_tts_metric_pipeline.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class FakeParaformer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_semantic = types.SimpleNamespace(ParaformerZHTranscriber=FakeParaformer)
    monkeypatch.setitem(sys.modules, "sure_eval.evaluation.nodes.transcription", fake_semantic)

    pipeline = module._build_real_pipeline(
        language="zh",
        device="cpu",
        cache_dir=Path("/tmp/sure-eval-tts"),
        semantic=True,
        speaker_backends=set(),
        mos_backends=set(),
    )

    assert set(pipeline.semantic_transcribers) == {"zh"}
    assert pipeline.speaker_providers == {}
    assert pipeline.mos_providers == {}


def test_local_deepspeed_stub_installs_optional_dependency_modules() -> None:
    import sys

    from sure_eval.compat.deepspeed_stub import install_deepspeed_stub

    for name in ("deepspeed.ops.adam", "deepspeed.ops", "deepspeed"):
        sys.modules.pop(name, None)

    install_deepspeed_stub()

    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

    assert deepspeed.__sure_eval_stub__ is True
    assert deepspeed.__version__ == "0.0.0"
    assert hasattr(deepspeed, "DeepSpeedEngine")
    assert DeepSpeedCPUAdam.__name__ == "DeepSpeedCPUAdam"
    assert FusedAdam.__name__ == "FusedAdam"


def test_mos_providers_fallback_to_shared_cache(tmp_path, monkeypatch) -> None:
    import sure_eval.evaluation.nodes.scoring.common.mos_providers as providers
    from sure_eval.evaluation.nodes.scoring.common.mos_providers import DNSMOSProvider, UTMOSProvider, WVMOSProvider

    shared = tmp_path / "shared" / "mos"
    (shared / "dnsmos" / "DNSMOS").mkdir(parents=True)
    (shared / "dnsmos" / "DNSMOS" / "model_v8.onnx").write_text("", encoding="utf-8")
    (shared / "dnsmos" / "DNSMOS" / "sig_bak_ovr.onnx").write_text("", encoding="utf-8")
    (shared / "repos" / "emergenttts-eval-public").mkdir(parents=True)
    (shared / "wv-mos").mkdir(parents=True)
    (shared / "wv-mos" / "wav2vec2.ckpt").write_text("", encoding="utf-8")
    (shared / "utmos22" / "UTMOS-demo").mkdir(parents=True)
    (shared / "utmos22" / "UTMOS-demo" / "epoch=3-step=7459.ckpt").write_text("", encoding="utf-8")
    monkeypatch.setattr(providers, "DEFAULT_SHARED_TTS_METRIC_CACHE", tmp_path / "shared")

    assert DNSMOSProvider(cache_dir=tmp_path / "empty")._resolve_env_root() == shared / "dnsmos"
    assert WVMOSProvider(cache_dir=tmp_path / "empty")._resolve_repo_dir() == shared / "repos" / "emergenttts-eval-public"
    assert WVMOSProvider(cache_dir=tmp_path / "empty")._resolve_checkpoint_path() == shared / "wv-mos" / "wav2vec2.ckpt"
    assert UTMOSProvider(cache_dir=tmp_path / "empty")._resolve_repo_dir() == shared / "utmos22" / "UTMOS-demo"


def test_tts_pipeline_docker_wrapper_plans_required_segments() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_tts_metric_pipeline_docker",
        Path("scripts/run_tts_metric_pipeline_docker.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--prediction-audio",
            "/tmp/hyp.wav",
            "--reference-text",
            "你好世界",
            "--reference-audio",
            "/tmp/ref.wav",
            "--language",
            "zh",
            "--output",
            "/tmp/out.json",
        ]
    )
    segments = module.build_segments(args)

    assert [segment.name for segment in segments] == [
        "semantic",
        "speaker_wavlm_ecapa",
        "speaker_eres2net",
        "mos_dnsmos_wvmos",
        "mos_utmos",
    ]
    eres2net = next(segment for segment in segments if segment.name == "speaker_eres2net")
    assert any("libsox.so" in mount for mount in eres2net.extra_mounts)
    assert any("libltdl.so" in mount for mount in eres2net.extra_mounts)
    assert segments[0].image == module.ASR_FUNASR_IMAGE


def test_tts_pipeline_docker_wrapper_uses_transformers_image_for_english_semantic() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_tts_metric_pipeline_docker",
        Path("scripts/run_tts_metric_pipeline_docker.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--prediction-audio",
            "/tmp/hyp.wav",
            "--reference-text",
            "hello world",
            "--reference-audio",
            "/tmp/ref.wav",
            "--language",
            "en",
            "--output",
            "/tmp/out.json",
        ]
    )
    segments = module.build_segments(args)

    assert segments[0].name == "semantic"
    assert segments[0].image == module.ASR_TTS_IMAGE


def test_tts_pipeline_docker_wrapper_maps_container_paths() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_tts_metric_pipeline_docker",
        Path("scripts/run_tts_metric_pipeline_docker.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    model_root = Path("/mnt/cloudstorfs/sjtu_home/jing.peng/workspace/sure-evaluation/src/sure_eval/models/m")
    args = module.parse_args(
        [
            "--prediction-audio",
            str(model_root / "artifacts" / "outputs" / "hyp.wav"),
            "--reference-text",
            "hello world",
            "--reference-audio",
            str(model_root / "fixture" / "tts" / "en" / "ref.wav"),
            "--cache-dir",
            "/mnt/cloudstorfs/sjtu_home/jing.peng/workspace/sure-evaluation/runtime/cache/tts-metrics",
            "--work-dir",
            str(model_root / "artifacts" / "tts_metric_parts"),
            "--output",
            str(model_root / "artifacts" / "tts_metric_report.json"),
        ]
    )
    segment = module.build_segments(args)[0]
    command = module._segment_command(args, segment, args.work_dir / segment.output_name)

    assert "/mnt/cloudstorfs" not in " ".join(command)
    assert any(
        item
        == "MODELSCOPE_CACHE=/hpc_stor03/sjtu_home/jing.peng/workspace/sure-evaluation/runtime/cache/tts-metrics/semantic/modelscope"
        for item in command
    )
    assert str(command[command.index("--prediction-audio") + 1]).startswith("/hpc_stor03/")
    assert str(command[command.index("--reference-audio") + 1]).startswith("/hpc_stor03/")
    assert str(command[command.index("--cache-dir") + 1]).startswith("/hpc_stor03/")
    assert str(command[command.index("--output") + 1]).startswith("/hpc_stor03/")


def test_tts_pipeline_docker_wrapper_passes_semantic_normalizer_only_to_semantic_segment() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_tts_metric_pipeline_docker",
        Path("scripts/run_tts_metric_pipeline_docker.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--prediction-audio",
            "/tmp/hyp.wav",
            "--reference-text",
            "你好世界",
            "--reference-audio",
            "/tmp/ref.wav",
            "--language",
            "zh",
            "--semantic-normalizer",
            "wetext:zh_tn",
            "--output",
            "/tmp/out.json",
        ]
    )
    segments = module.build_segments(args)
    semantic_command = module._segment_command(args, segments[0], args.work_dir / segments[0].output_name)
    speaker_command = module._segment_command(args, segments[1], args.work_dir / segments[1].output_name)

    assert "--semantic-normalizer" in semantic_command
    assert semantic_command[semantic_command.index("--semantic-normalizer") + 1] == "wetext:zh_tn"
    assert "--semantic-normalizer" not in speaker_command


def test_tts_pipeline_docker_wrapper_allows_metric_image_overrides(monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    monkeypatch.setenv("SURE_TTS_ASR_FUNASR_IMAGE", "registry.local/funasr:structlog")
    monkeypatch.setenv("SURE_TTS_ASR_TTS_IMAGE", "registry.local/asr-tts:structlog")
    monkeypatch.setenv("SURE_TTS_UTMOS_IMAGE", "registry.local/utmos:structlog")
    spec = importlib.util.spec_from_file_location(
        "run_tts_metric_pipeline_docker_override",
        Path("scripts/run_tts_metric_pipeline_docker.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    args = module.parse_args(
        [
            "--prediction-audio",
            "/tmp/hyp.wav",
            "--reference-text",
            "hello world",
            "--reference-audio",
            "/tmp/ref.wav",
            "--language",
            "en",
            "--output",
            "/tmp/out.json",
        ]
    )
    segments = module.build_segments(args)

    assert module.ASR_FUNASR_IMAGE == "registry.local/funasr:structlog"
    assert segments[0].image == "registry.local/asr-tts:structlog"
    assert segments[-1].image == "registry.local/utmos:structlog"


def test_tts_pipeline_docker_wrapper_defaults_to_existing_sensevoice_image(monkeypatch) -> None:
    import importlib.util
    from pathlib import Path

    monkeypatch.delenv("SURE_TTS_ASR_FUNASR_IMAGE", raising=False)
    spec = importlib.util.spec_from_file_location(
        "run_tts_metric_pipeline_docker_default_image",
        Path("scripts/run_tts_metric_pipeline_docker.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert (
        module.ASR_FUNASR_IMAGE
        == "docker.v2.aispeech.com/sjtu/sjtu_yukai-dujunhao-sure_funaudiollm__sensevoicesmall:v1.0"
    )


def test_tts_docker_shell_accepts_skip_semantic_argument() -> None:
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_tts_metric_pipeline_docker.sh",
            "--prediction-audio",
            "hyp.wav",
            "--reference-text",
            "hello",
            "--reference-audio",
            "ref.wav",
            "--output",
            "/tmp/out.json",
            "--skip-semantic",
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "Unknown argument: --skip-semantic" not in completed.stderr


def test_tts_docker_shell_accepts_semantic_normalizer_argument() -> None:
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_tts_metric_pipeline_docker.sh",
            "--prediction-audio",
            "hyp.wav",
            "--reference-text",
            "hello",
            "--reference-audio",
            "ref.wav",
            "--output",
            "/tmp/out.json",
            "--semantic-normalizer",
            "wetext:zh_tn",
            "--help",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "Unknown argument: --semantic-normalizer" not in completed.stderr


def test_tts_pipeline_docker_wrapper_uses_hpc_workdir() -> None:
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_tts_metric_pipeline_docker",
        Path("scripts/run_tts_metric_pipeline_docker.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    mapped = module.to_hpc_path(Path("/mnt/cloudstorfs/sjtu_home/jing.peng/workspace/sure-evaluation"))

    assert str(mapped) == "/hpc_stor03/sjtu_home/jing.peng/workspace/sure-evaluation"


def test_validation_plan_lists_required_models_without_importing_heavy_deps() -> None:
    from sure_eval.evaluation.tasks.tts.validate_metrics import build_validation_plan

    plan = build_validation_plan(suite="all", device="cpu")
    model_ids = {item["model_id"] for item in plan["models"]}
    base_model_ids = {item.get("base_model_id") for item in plan["models"]}

    assert "openai/whisper-large-v3" in model_ids
    assert "paraformer-zh" in model_ids
    assert "wavlm_large_finetune.pth" in model_ids
    assert "microsoft/wavlm-large" in base_model_ids
    assert "speechbrain/spkrec-ecapa-voxceleb" in model_ids
    assert "iic/speech_eres2net_sv_zh-cn_16k-common" in model_ids
