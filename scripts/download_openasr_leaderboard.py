#!/usr/bin/env python3
"""Download the Open ASR Leaderboard datasets to a user-chosen location.

Fetches the exact inputs consumed by ``scripts/prepare_openasr_leaderboard.py``:

- Six parquet configs from the Hugging Face dataset
  ``hf-audio/open-asr-leaderboard`` (per-config shard directories), and
- ``ArtificialAnalysis/Earnings22-Cleaned-AA`` (JSONL + mp3 audio), which the
  leaderboard uses for its cleaned Earnings22 column.

The output layout mirrors the source repos, so the result can be passed
directly as ``--source-root`` to the prepare script:

    <output-dir>/ami_cleaned/test-*.parquet
    <output-dir>/gigaspeech_cleaned/test-*.parquet
    <output-dir>/librispeech/test.clean-*.parquet
    <output-dir>/librispeech/test.other-*.parquet
    <output-dir>/spgispeech/test-*.parquet
    <output-dir>/voxpopuli_cleaned_aa/test-*.parquet
    <output-dir>/earnings22_cleaned_aa/earnings22_cleaned_aa_v1.jsonl
    <output-dir>/earnings22_cleaned_aa/audio/*.mp3

Requires the ``download`` extra (``huggingface-hub``), imported lazily. Network
proxies are honored through the standard ``HTTPS_PROXY``/``HTTP_PROXY``
environment variables; set ``HF_ENDPOINT`` (for example a mirror) or
``HF_TOKEN`` through the environment as usual.

Usage:
    python scripts/download_openasr_leaderboard.py --output-dir /data/open-asr-leaderboard

    # subset + dry run
    python scripts/download_openasr_leaderboard.py \
        --output-dir /data/open-asr-leaderboard \
        --datasets librispeech_test.clean,earnings22_cleaned_aa_test --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

LEADERBOARD_REPO = "hf-audio/open-asr-leaderboard"
EARNINGS22_REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA"
EARNINGS22_DATASET = "earnings22_cleaned_aa_test"
EARNINGS22_PATTERNS = ("earnings22_cleaned_aa_v1.jsonl", "audio/*")

# Recipe dataset name -> shard patterns inside the leaderboard repo.
LEADERBOARD_PATTERNS: dict[str, tuple[str, ...]] = {
    "ami_cleaned_test": ("ami_cleaned/test-*.parquet",),
    "gigaspeech_cleaned_test": ("gigaspeech_cleaned/test-*.parquet",),
    "librispeech_test.clean": ("librispeech/test.clean-*.parquet",),
    "librispeech_test.other": ("librispeech/test.other-*.parquet",),
    "spgispeech_test": ("spgispeech/test-*.parquet",),
    "voxpopuli_cleaned_aa_test": ("voxpopuli_cleaned_aa/test-*.parquet",),
}

ALL_DATASETS = tuple(LEADERBOARD_PATTERNS) + (EARNINGS22_DATASET,)


def planned_files(name: str, api) -> list[str]:
    """List the repo files one dataset would download (dry-run support)."""
    if name == EARNINGS22_DATASET:
        repo, patterns = EARNINGS22_REPO, EARNINGS22_PATTERNS
    else:
        repo, patterns = LEADERBOARD_REPO, LEADERBOARD_PATTERNS[name]
    files = api.list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if any(fnmatch.fnmatch(f, p) for p in patterns))


def download_dataset(name: str, output_dir: Path, api_module) -> Path:
    """Download one dataset and return its local directory."""
    snapshot_download = api_module.snapshot_download
    if name == EARNINGS22_DATASET:
        return Path(
            snapshot_download(
                EARNINGS22_REPO,
                repo_type="dataset",
                allow_patterns=list(EARNINGS22_PATTERNS),
                local_dir=str(output_dir / "earnings22_cleaned_aa"),
            )
        )
    return Path(
        snapshot_download(
            LEADERBOARD_REPO,
            repo_type="dataset",
            allow_patterns=list(LEADERBOARD_PATTERNS[name]),
            local_dir=str(output_dir),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Open ASR Leaderboard datasets from Hugging Face"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Download destination; pass it later as --source-root to prepare_openasr_leaderboard.py",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=f"Comma-separated subset of: {', '.join(ALL_DATASETS)} (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the files that would be downloaded",
    )
    args = parser.parse_args()

    if args.datasets:
        selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
        unknown = sorted(set(selected) - set(ALL_DATASETS))
        if unknown:
            print(f"unknown datasets: {unknown}; available: {list(ALL_DATASETS)}", file=sys.stderr)
            return 1
    else:
        selected = list(ALL_DATASETS)

    try:
        import huggingface_hub
    except ImportError:
        print(
            "huggingface_hub is required: install the 'download' extra "
            "(pip install -e .[download]) or uv run --extra download",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        api = huggingface_hub.HfApi()
        for name in selected:
            files = planned_files(name, api)
            print(f"[{name}] {len(files)} file(s)")
            for path in files:
                print(f"  {path}")
        return 0

    for name in selected:
        print(f"[{name}] downloading ...", flush=True)
        target = download_dataset(name, args.output_dir, huggingface_hub)
        print(f"  -> {target}")

    print("\nNext step:")
    print(
        "  python scripts/prepare_openasr_leaderboard.py "
        f"--source-root {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
