#!/usr/bin/env python3
"""Run the Open ASR Leaderboard recipe with Qwen3-ASR-1.7B.

Consumes the versioned datasets produced by
``scripts/prepare_openasr_leaderboard.py``, transcribes each dataset with
Qwen/Qwen3-ASR-1.7B, scores every dataset through the leaderboard-aligned
pipeline ``asr.en.wer.openasr_norm_english_v1.wenet_wer_v1``, and writes one
report with per-dataset WER, the leaderboard-style arithmetic mean WER, and
informational RTFx (total audio seconds / total transcription seconds).

Prepare the Qwen3-ASR node environment first (see
docs/recipes/openasr_leaderboard_qwen3asr.md):

    sure-eval env setup --node transcription/qwen3_asr_1_7b
    sure-eval env download --node transcription/qwen3_asr_1_7b

Usage:
    python scripts/run_openasr_leaderboard_eval.py \
        --data-root data/datasets/openasr_leaderboard

    # smoke run
    python scripts/run_openasr_leaderboard_eval.py \
        --data-root data/datasets/openasr_leaderboard --limit 20

    # rescore existing hypotheses without running inference
    python scripts/run_openasr_leaderboard_eval.py \
        --data-root data/datasets/openasr_leaderboard --hyp-dir results/hyps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PIPELINE_ID = "asr.en.wer.openasr_norm_english_v1.wenet_wer_v1"
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DATASET_VERSION = "v1.0.0"

ALL_DATASETS = (
    "ami_cleaned_test",
    "earnings22_cleaned_aa_test",
    "gigaspeech_cleaned_test",
    "librispeech_test.clean",
    "librispeech_test.other",
    "spgispeech_test",
    "voxpopuli_cleaned_aa_test",
)


def read_key_text(path: Path) -> list[tuple[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if "\t" not in line:
                continue
            key, text = line.rstrip("\n").split("\t", 1)
            rows.append((key, text))
    return rows


def score_dataset(ref_file: Path, hyp_file: Path) -> dict:
    from sure_eval.evaluation.tasks.asr.pipeline import evaluate_asr_files

    report = evaluate_asr_files(
        str(ref_file),
        str(hyp_file),
        language="en",
        metric="wer",
        normalizer="openasr",
        scorer="wenet",
    )
    if report.pipeline_id != PIPELINE_ID:
        raise ValueError(f"pipeline_id drifted: {report.pipeline_id!r} != {PIPELINE_ID!r}")
    scoring = report.details["scoring_result"]
    return {
        "wer_percent": round(scoring["wer_percent"], 2),
        "num_ref_utts": scoring.get("num_ref_utts"),
        "num_hyp_missing_utts": scoring.get("num_hyp_missing_utts"),
        "pipeline_id": report.pipeline_id,
    }


def transcribe_dataset(
    dataset_dir: Path,
    keys: list[str],
    *,
    device: str,
    max_new_tokens: int,
) -> tuple[Path, float]:
    from sure_eval.evaluation.nodes.transcription.common.providers import (
        Qwen3ASR17BTranscriber,
    )

    transcriber = Qwen3ASR17BTranscriber(device=device, max_new_tokens=max_new_tokens)
    hyp_path = dataset_dir / "hyp.txt"
    total_time = 0.0
    with hyp_path.open("w", encoding="utf-8") as hyp_out:
        for index, key in enumerate(keys, 1):
            wav = dataset_dir / "wavs" / f"{key}.wav"
            started = time.perf_counter()
            text = transcriber.transcribe(str(wav), language="en")
            total_time += time.perf_counter() - started
            hyp_out.write(f"{key}\t{text}\n")
            if index % 100 == 0:
                print(f"  {index}/{len(keys)} transcribed", flush=True)
    return hyp_path, total_time


def aggregate(results: dict[str, dict]) -> dict:
    scored = {name: entry for name, entry in results.items() if entry.get("wer_percent") is not None}
    mean_wer = (
        round(sum(entry["wer_percent"] for entry in scored.values()) / len(scored), 2)
        if scored
        else None
    )
    total_audio = sum(entry.get("audio_s") or 0.0 for entry in scored.values())
    total_time = sum(entry.get("inference_s") or 0.0 for entry in scored.values())
    return {
        "mean_wer_percent": mean_wer,
        "rtfx": round(total_audio / total_time, 2) if total_time else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open ASR Leaderboard recipe with Qwen3-ASR-1.7B"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/datasets/openasr_leaderboard"),
        help="Root written by prepare_openasr_leaderboard.py",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=f"Comma-separated subset of: {', '.join(ALL_DATASETS)} (default: all prepared)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Generation cap per sample (long-form sets like Earnings22 need more than the node default)",
    )
    parser.add_argument(
        "--hyp-dir",
        type=Path,
        default=None,
        help="Reuse <hyp-dir>/<dataset>/hyp.txt instead of running inference",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory (default: results/openasr_leaderboard/<utc timestamp>)",
    )
    args = parser.parse_args()

    if args.datasets:
        selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
        unknown = sorted(set(selected) - set(ALL_DATASETS))
        if unknown:
            print(f"unknown datasets: {unknown}; available: {list(ALL_DATASETS)}", file=sys.stderr)
            return 1
    else:
        selected = [
            name for name in ALL_DATASETS if (args.data_root / name / DATASET_VERSION).is_dir()
        ]
    if not selected:
        print(f"no prepared datasets below {args.data_root}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or (
        Path("results")
        / "openasr_leaderboard"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name in selected:
        dataset_dir = args.data_root / name / DATASET_VERSION
        ref_file = dataset_dir / "ref.txt"
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        ref_rows = read_key_text(ref_file)
        if args.limit is not None:
            ref_rows = ref_rows[: args.limit]
        keys = [key for key, _ in ref_rows]

        print(f"[{name}] {len(keys)} samples")
        inference_s: float | None = None
        if args.hyp_dir is not None:
            hyp_file = args.hyp_dir / name / "hyp.txt"
            if not hyp_file.exists():
                print(f"  missing {hyp_file}, skipped", file=sys.stderr)
                continue
        else:
            hyp_file, inference_s = transcribe_dataset(
                dataset_dir,
                keys,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )

        scores = score_dataset(ref_file, hyp_file)
        audio_s = manifest.get("total_audio_s")
        if args.limit is not None and audio_s is not None:
            audio_s = None  # limit breaks the manifest-level duration; do not guess
        entry = {
            **scores,
            "audio_s": audio_s,
            "inference_s": round(inference_s, 3) if inference_s is not None else None,
        }
        entry["rtfx"] = (
            round(audio_s / inference_s, 2)
            if audio_s is not None and inference_s
            else None
        )
        results[name] = entry
        print(f"  WER = {entry['wer_percent']}%")

    report = {
        "recipe": "openasr_leaderboard_qwen3asr",
        "model": MODEL_ID,
        "pipeline_id": PIPELINE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "datasets": results,
        "aggregate": aggregate(results),
        "notes": [
            "WER uses normalization/openasr_norm + scoring/wenet_wer; the official"
            " leaderboard scores with kaldialign merge_compounds=True.",
            "mean_wer_percent is the arithmetic mean over datasets, matching the"
            " leaderboard composite; rtfx is informational only.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== Open ASR Leaderboard recipe ===")
    for name, entry in results.items():
        print(f"{name}: WER {entry['wer_percent']}%")
    print(f"mean WER: {report['aggregate']['mean_wer_percent']}%")
    if report["aggregate"]["rtfx"] is not None:
        print(f"RTFx: {report['aggregate']['rtfx']}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
