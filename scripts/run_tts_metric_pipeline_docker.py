#!/usr/bin/env python3
"""Run all TTS metric backends through the validated Docker environments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKER_REPO_ROOT = REPO_ROOT
DEFAULT_CACHE_DIR = REPO_ROOT / "runtime" / "cache" / "tts-metrics"
DEFAULT_WORK_DIR = REPO_ROOT / "artifacts" / "tts_metric_pipeline"

ASR_FUNASR_IMAGE = os.environ.get(
    "SURE_TTS_ASR_FUNASR_IMAGE",
    "docker.v2.aispeech.com/sjtu/sjtu_yukai-dujunhao-sure_funaudiollm__sensevoicesmall:v1.0",
)
ASR_TTS_IMAGE = os.environ.get(
    "SURE_TTS_ASR_TTS_IMAGE",
    "docker.v2.aispeech.com/sjtu/sjtu_yukai-wenbinhuang-asr-tts:eval-dnsmos",
)
UTMOS_IMAGE = os.environ.get(
    "SURE_TTS_UTMOS_IMAGE",
    "docker.v2.aispeech.com/sjtu/sjtu_yukai-yiweiguo-utmos:v1.0",
)


@dataclass(frozen=True)
class Segment:
    name: str
    image: str
    output_name: str
    no_semantic: bool = False
    speaker_backends: str = ""
    mos_backends: str = ""
    extra_env: dict[str, str] = field(default_factory=dict)
    extra_mounts: list[str] = field(default_factory=list)


def to_hpc_path(path: Path) -> Path:
    """Map cloudstorfs paths to the /hpc_stor03 path Docker wrapper handles."""
    text = str(path)
    prefix = "/mnt/cloudstorfs/"
    if text.startswith(prefix):
        return Path("/hpc_stor03") / text[len(prefix):]
    return path


def _container_path(path: Path) -> str:
    return str(to_hpc_path(path))


def _container_env_value(value: str) -> str:
    if value.startswith("/mnt/cloudstorfs/"):
        return str(to_hpc_path(Path(value)))
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-audio", required=True, type=Path)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--reference-audio", required=True, type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument(
        "--semantic-normalizer",
        help='Optional semantic ASR normalizer, for example "wetext:zh_tn".',
    )
    parser.add_argument("--speaker-backends", default="wavlm-large,ecapa-tdnn,eres2net")
    parser.add_argument("--mos-backends", default="dnsmos,wv-mos,utmos")
    parser.add_argument("--keep-partials", action="store_true")
    return parser.parse_args(argv)


def _csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _existing_mount(path: str) -> list[str]:
    source = Path(path)
    if source.exists():
        return [f"{source}:{source}:ro"]
    return []


def _eres2net_sox_mounts() -> list[str]:
    mounts: list[str] = []
    for path in (
        "/usr/lib64/libsox.so",
        "/usr/lib64/libsox.so.3",
        "/usr/lib64/libsox.so.3.0.0",
        "/usr/lib64/libltdl.so.7",
        "/usr/lib64/libltdl.so.7.3.1",
    ):
        mounts.extend(_existing_mount(path))
    return mounts


def _uses_chinese_asr(language: str) -> bool:
    return language.lower().startswith(("zh", "cmn", "yue"))


def build_segments(args: argparse.Namespace) -> list[Segment]:
    speaker_backends = _csv(args.speaker_backends)
    mos_backends = _csv(args.mos_backends)
    segments: list[Segment] = []

    if not args.skip_semantic:
        semantic_image = ASR_FUNASR_IMAGE if _uses_chinese_asr(args.language) else ASR_TTS_IMAGE
        segments.append(
            Segment(
                name="semantic",
                image=semantic_image,
                output_name="semantic.json",
                speaker_backends="",
                mos_backends="",
                extra_env={
                    "MODELSCOPE_CACHE": str(args.cache_dir / "semantic" / "modelscope"),
                    "HF_HOME": str(args.cache_dir / "semantic" / "huggingface"),
                    "HF_HUB_CACHE": str(args.cache_dir / "semantic" / "huggingface" / "hub"),
                },
            )
        )

    wavlm_ecapa = ",".join(name for name in ("wavlm-large", "ecapa-tdnn") if name in speaker_backends)
    if wavlm_ecapa:
        segments.append(
            Segment(
                name="speaker_wavlm_ecapa",
                image=ASR_TTS_IMAGE,
                output_name="speaker_wavlm_ecapa.json",
                no_semantic=True,
                speaker_backends=wavlm_ecapa,
                mos_backends="",
                extra_env={
                    "HF_HOME": str(args.cache_dir / "huggingface"),
                    "HF_HUB_CACHE": str(args.cache_dir / "huggingface" / "hub"),
                    "MODELSCOPE_CACHE": str(args.cache_dir / "modelscope"),
                    "TRITON_CACHE_DIR": "/tmp/sure-eval-triton",
                },
            )
        )

    if "eres2net" in speaker_backends:
        segments.append(
            Segment(
                name="speaker_eres2net",
                image=ASR_FUNASR_IMAGE,
                output_name="speaker_eres2net.json",
                no_semantic=True,
                speaker_backends="eres2net",
                mos_backends="",
                extra_env={
                    "MODELSCOPE_CACHE": str(args.cache_dir / "speaker" / "modelscope"),
                    "LD_LIBRARY_PATH": "/usr/lib64:/opt/conda/lib",
                },
                extra_mounts=_eres2net_sox_mounts(),
            )
        )

    dnsmos_wvmos = ",".join(name for name in ("dnsmos", "wv-mos") if name in mos_backends)
    if dnsmos_wvmos:
        segments.append(
            Segment(
                name="mos_dnsmos_wvmos",
                image=ASR_TTS_IMAGE,
                output_name="mos_dnsmos_wvmos.json",
                no_semantic=True,
                speaker_backends="",
                mos_backends=dnsmos_wvmos,
                extra_env={
                    "HF_HOME": str(args.cache_dir / "mos" / "huggingface"),
                    "HF_HUB_CACHE": str(args.cache_dir / "mos" / "huggingface" / "hub"),
                    "TRITON_CACHE_DIR": "/tmp/sure-eval-triton",
                },
            )
        )

    if "utmos" in mos_backends:
        segments.append(
            Segment(
                name="mos_utmos",
                image=UTMOS_IMAGE,
                output_name="mos_utmos.json",
                no_semantic=True,
                speaker_backends="",
                mos_backends="utmos",
            )
        )

    return segments


def _docker_base(args: argparse.Namespace, segment: Segment) -> list[str]:
    command = [
        "env",
        "-u",
        "HTTP_PROXY",
        "-u",
        "HTTPS_PROXY",
        "-u",
        "http_proxy",
        "-u",
        "https_proxy",
        "-u",
        "ALL_PROXY",
        "-u",
        "all_proxy",
        "docker",
        "run",
        "--rm",
        "--gpus",
        f'"device={args.gpu}"',
        "-v",
        "/hpc_stor03:/hpc_stor03",
        "-w",
        str(to_hpc_path(REPO_ROOT)),
        "-e",
        "PYTHONPATH=src",
    ]
    for key, value in segment.extra_env.items():
        command.extend(["-e", f"{key}={_container_env_value(value)}"])
    for mount in segment.extra_mounts:
        command.extend(["-v", mount])
    command.append(segment.image)
    return command


def _segment_command(args: argparse.Namespace, segment: Segment, output_path: Path) -> list[str]:
    command = _docker_base(args, segment)
    command.extend(
        [
            "python",
            "scripts/run_tts_metric_pipeline.py",
            "--prediction-audio",
            _container_path(args.prediction_audio),
            "--reference-text",
            args.reference_text,
            "--reference-audio",
            _container_path(args.reference_audio),
            "--language",
            args.language,
            "--device",
            args.device,
            "--cache-dir",
            _container_path(args.cache_dir),
            "--speaker-backends",
            segment.speaker_backends,
            "--mos-backends",
            segment.mos_backends,
            "--output",
            _container_path(output_path),
        ]
    )
    if args.sample_id:
        command.extend(["--sample-id", args.sample_id])
    if args.semantic_normalizer and not segment.no_semantic:
        command.extend(["--semantic-normalizer", args.semantic_normalizer])
    if segment.no_semantic:
        command.append("--no-semantic")
    return command


def run_segments(args: argparse.Namespace, segments: list[Segment]) -> list[Path]:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for segment in segments:
        output_path = args.work_dir / segment.output_name
        command = _segment_command(args, segment, output_path)
        print(f"[tts-metric] running {segment.name}", flush=True)
        subprocess.run(["/bin/bash", "-lc", shlex.join(command)], check=True)
        outputs.append(output_path)
    return outputs


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_reports(paths: list[Path], output: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "sample": None,
        "ok": True,
        "metrics": {},
        "errors": [],
        "source_reports": [str(path) for path in paths],
    }
    for path in paths:
        report = _read_json(path)
        merged["sample"] = merged["sample"] or report.get("sample")
        merged["metrics"].update(report.get("metrics", {}))
        merged["errors"].extend(report.get("errors", []))

    sim_metrics = {
        name: metric["score"]
        for name, metric in merged["metrics"].items()
        if name.startswith("sim/") and isinstance(metric, dict) and "score" in metric
    }
    if sim_metrics:
        merged["metrics"]["sim"] = {
            "metric_name": "sim",
            "score": sum(float(score) for score in sim_metrics.values()) / len(sim_metrics),
            "details": {
                "num_backends": len(sim_metrics),
                "backend_metrics": sim_metrics,
            },
        }
    merged["ok"] = not merged["errors"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return merged


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    segments = build_segments(args)
    outputs = run_segments(args, segments)
    merged = merge_reports(outputs, args.output)
    print(json.dumps({name: metric["score"] for name, metric in merged["metrics"].items()}, ensure_ascii=False, indent=2))
    if merged["errors"]:
        print(json.dumps(merged["errors"], ensure_ascii=False, indent=2), file=sys.stderr)
    if not args.keep_partials:
        pass
    return 0 if merged["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
