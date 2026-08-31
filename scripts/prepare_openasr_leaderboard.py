#!/usr/bin/env python3
"""Materialize a local Open ASR Leaderboard data copy into versioned SURE-EVAL inputs.

Reads the Hugging Face parquet shards of the Open ASR Leaderboard datasets (and
the Earnings22-Cleaned-AA JSONL variant) from a local directory and writes, per
dataset, a versioned directory:

    <output-root>/<dataset>/v1.0.0/
      wavs/<id>.wav     # 16 kHz mono PCM16 audio
      ref.txt           # key<TAB>text, the engine key-text input contract
      manifest.json     # provenance: source shards + sha256, counts, durations

Dataset names follow the Open ASR Leaderboard scoring keys (for example
``ami_cleaned_test`` or ``librispeech_test.clean``) so reports line up with the
official leaderboard columns.

Optional dependencies (lazy imports): pyarrow for parquet shards, soundfile for
in-memory audio bytes, librosa for the Earnings22 mp3 files.

Usage:
    python scripts/prepare_openasr_leaderboard.py \
        --source-root /path/to/open-asr-leaderboard \
        --output-root data/datasets/openasr_leaderboard

    # smoke subset
    python scripts/prepare_openasr_leaderboard.py \
        --source-root /path/to/open-asr-leaderboard \
        --datasets librispeech_test.clean --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

DATASET_VERSION = "v1.0.0"
MANIFEST_SCHEMA = "openasr_leaderboard.recipe.v1"
TARGET_SAMPLE_RATE = 16000

# Text columns tried in order, mirroring the leaderboard's get_text fallback.
TEXT_KEYS = ("text", "sentence", "normalized_text", "transcript", "transcription")

# Dataset name -> source layout. Names match the leaderboard scoring keys.
PARQUET_DATASETS: dict[str, str] = {
    "ami_cleaned_test": "ami_cleaned/test-*.parquet",
    "gigaspeech_cleaned_test": "gigaspeech_cleaned/test-*.parquet",
    "librispeech_test.clean": "librispeech/test.clean-*.parquet",
    "librispeech_test.other": "librispeech/test.other-*.parquet",
    "spgispeech_test": "spgispeech/test-*.parquet",
    "voxpopuli_cleaned_aa_test": "voxpopuli_cleaned_aa/test-*.parquet",
}
EARNINGS22_DATASET = "earnings22_cleaned_aa_test"
EARNINGS22_JSONL = "earnings22_cleaned_aa/earnings22_cleaned_aa_v1.jsonl"
EARNINGS22_AUDIO_DIR = "earnings22_cleaned_aa/audio"

ALL_DATASETS = tuple(PARQUET_DATASETS) + (EARNINGS22_DATASET,)


def extract_text(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)
        if value is not None:
            return str(value)
    raise ValueError(f"row has none of the known text columns {TEXT_KEYS}: {sorted(row)}")


def sanitize_key(key: str) -> str:
    return str(key).replace("/", "_").replace("\\", "_").strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_wav(path: Path, samples: Any, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    data = np.asarray(samples, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=-1)
    if sample_rate != TARGET_SAMPLE_RATE:
        import librosa

        data = librosa.resample(data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, TARGET_SAMPLE_RATE, subtype="PCM_16")


def write_wav_from_bytes(path: Path, payload: bytes) -> float:
    """Decode encoded audio bytes (wav/flac/...) and write 16 kHz mono wav."""
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")
    write_wav(path, data, sample_rate)
    return float(len(data)) / float(sample_rate)


def transcode_audio_file(src: Path, dst: Path) -> float:
    """Transcode an on-disk audio file (mp3/wav/...) to 16 kHz mono wav."""
    import librosa

    data, _ = librosa.load(str(src), sr=TARGET_SAMPLE_RATE, mono=True)
    write_wav(dst, data, TARGET_SAMPLE_RATE)
    return float(len(data)) / float(TARGET_SAMPLE_RATE)


def iter_parquet_rows(shard: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(shard)
    for batch in parquet.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            yield row


def prepare_parquet_dataset(
    name: str,
    source_root: Path,
    output_root: Path,
    limit: int | None,
) -> dict[str, Any]:
    shards = sorted(source_root.glob(PARQUET_DATASETS[name]))
    if not shards:
        raise FileNotFoundError(
            f"no parquet shards match {PARQUET_DATASETS[name]!r} below {source_root}"
        )

    out_dir = output_root / name / DATASET_VERSION
    wavs_dir = out_dir / "wavs"
    ref_path = out_dir / "ref.txt"

    num_samples = 0
    total_audio_s = 0.0
    with _open_ref(ref_path) as ref_out:
        for shard in shards:
            for row in iter_parquet_rows(shard):
                if limit is not None and num_samples >= limit:
                    break
                key = sanitize_key(row.get("id") or f"sample_{num_samples}")
                audio = row.get("audio") or {}
                payload = audio.get("bytes") if isinstance(audio, dict) else None
                if payload is None:
                    raise ValueError(f"{name} row {key} has no embedded audio bytes")
                duration = write_wav_from_bytes(wavs_dir / f"{key}.wav", bytes(payload))
                declared = row.get("audio_length_s")
                total_audio_s += float(declared) if declared is not None else duration
                ref_out.write(f"{key}\t{extract_text(row)}\n")
                num_samples += 1
            if limit is not None and num_samples >= limit:
                break

    return _finalize(name, source_root, out_dir, shards, num_samples, total_audio_s)


def prepare_earnings22(
    source_root: Path,
    output_root: Path,
    limit: int | None,
) -> dict[str, Any]:
    jsonl_path = source_root / EARNINGS22_JSONL
    audio_dir = source_root / EARNINGS22_AUDIO_DIR
    if not jsonl_path.exists():
        raise FileNotFoundError(f"missing {jsonl_path}")

    out_dir = output_root / EARNINGS22_DATASET / DATASET_VERSION
    wavs_dir = out_dir / "wavs"
    ref_path = out_dir / "ref.txt"

    num_samples = 0
    total_audio_s = 0.0
    with _open_ref(ref_path) as ref_out:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if limit is not None and num_samples >= limit:
                break
            row = json.loads(line)
            key = sanitize_key(row["id"])
            src_audio = audio_dir / str(row["file_name"])
            if not src_audio.exists():
                raise FileNotFoundError(f"missing audio file {src_audio}")
            duration = transcode_audio_file(src_audio, wavs_dir / f"{key}.wav")
            declared = row.get("duration")
            total_audio_s += float(declared) if declared is not None else duration
            ref_out.write(f"{key}\t{extract_text(row)}\n")
            num_samples += 1

    return _finalize(
        EARNINGS22_DATASET, source_root, out_dir, [jsonl_path], num_samples, total_audio_s
    )


def _open_ref(ref_path: Path):
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    return ref_path.open("w", encoding="utf-8")


def _finalize(
    name: str,
    source_root: Path,
    out_dir: Path,
    source_files: Iterable[Path],
    num_samples: int,
    total_audio_s: float,
) -> dict[str, Any]:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dataset": name,
        "version": DATASET_VERSION,
        "generator": Path(__file__).name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "sources": [
            {
                "path": str(path.relative_to(source_root)),
                "sha256": sha256_file(path),
            }
            for path in source_files
        ],
        "num_samples": num_samples,
        "total_audio_s": round(total_audio_s, 3),
        "ref_file": "ref.txt",
        "audio_dir": "wavs",
        "audio_format": f"wav pcm16 {TARGET_SAMPLE_RATE} Hz mono",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Open ASR Leaderboard datasets for SURE-EVAL"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Local copy of the Open ASR Leaderboard data (parquet shards directory)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/openasr_leaderboard"),
        help="Output root (default: data/datasets/openasr_leaderboard)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=f"Comma-separated subset of: {', '.join(ALL_DATASETS)} (default: all)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max samples per dataset")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        print(f"source root does not exist: {source_root}", file=sys.stderr)
        return 1

    if args.datasets:
        selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
        unknown = sorted(set(selected) - set(ALL_DATASETS))
        if unknown:
            print(f"unknown datasets: {unknown}; available: {list(ALL_DATASETS)}", file=sys.stderr)
            return 1
    else:
        selected = list(ALL_DATASETS)

    for name in selected:
        if name == EARNINGS22_DATASET:
            manifest = prepare_earnings22(source_root, args.output_root, args.limit)
        else:
            manifest = prepare_parquet_dataset(name, source_root, args.output_root, args.limit)
        print(
            f"{name}: {manifest['num_samples']} samples, "
            f"{manifest['total_audio_s'] / 3600:.2f} h -> "
            f"{args.output_root / name / DATASET_VERSION}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
