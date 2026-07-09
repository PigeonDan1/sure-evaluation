#!/usr/bin/env python3
"""Materialize local test-only dataset JSONL files for SURE-EVAL.

This script only prepares evaluation splits requested by the local benchmark
setup. It does not touch train/dev splits.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import shutil
import subprocess
import zipfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


DATASETS_ROOT = Path(__file__).resolve().parent.parent / "data" / "datasets"
SURE_ROOT = DATASETS_ROOT / "sure_benchmark"
JSONL_ROOT = SURE_ROOT / "jsonl"
SURE_SUITES_ROOT = SURE_ROOT / "SURE_Test_Suites"

LIBRISPEECH_ROOT = Path("/hpc_stor03/public/shared/data/asr/rawdata/LibriSpeech")
WENETSPEECH_ROOT = Path("/hpc_stor03/public/shared/data/asr/rawdata/WenetSpeech")
GIGASPEECH_ROOT = Path("/hpc_stor03/public/shared/data/asr/am/GigaSpeech")
SLIDESPEECH_ROOT = DATASETS_ROOT / "slidespeech_test"
SLIDESPEECH_INFO_ROOT = SLIDESPEECH_ROOT / "info" / "test"
SLIDESPEECH_AUDIO_ROOT = SLIDESPEECH_ROOT / "audio"
SEEDTTS_ROOT = DATASETS_ROOT / "seedtts_test_eval"
SEEDTTS_SOURCE_ROOT = SEEDTTS_ROOT / "source"
SEEDTTS_HF_MIRROR = "https://hf-mirror.com/datasets/zhaochenyang20/seed-tts-eval/resolve/main"

CV3_GITHUB_API = "https://api.github.com/repos/FunAudioLLM/CV3-Eval/contents"
CV3_HF_MIRROR = "https://hf-mirror.com/datasets/yuekai/CV3-Eval/resolve/main"
CV3_ROOT = DATASETS_ROOT / "cv3_eval"
SOURCES_ROOT = DATASETS_ROOT / "_sources"
CV3_SOURCE_ROOT = SOURCES_ROOT / "CV3-Eval-main"
CV3_SPARSE_ROOT = SOURCES_ROOT / "CV3-Eval-sparse"
CV3_ARCHIVE = SOURCES_ROOT / "CV3-Eval-main.zip"

CV3_PARQUET_SPLITS = [
    "cross_lingual_zeroshot_to_en",
    "cross_lingual_zeroshot_to_hard_en",
    "cross_lingual_zeroshot_to_hard_zh",
    "cross_lingual_zeroshot_to_ja",
    "cross_lingual_zeroshot_to_ko",
    "cross_lingual_zeroshot_to_zh",
    "emotion_zeroshot_en",
    "emotion_zeroshot_zh",
    "subjective_continue_emotion",
    "subjective_continue_rhyme",
    "subjective_continue_speed",
    "subjective_continue_volume",
    "subjective_zeroshot",
    "zero_shot_de",
    "zero_shot_en",
    "zero_shot_es",
    "zero_shot_fr",
    "zero_shot_hard_en",
    "zero_shot_hard_zh",
    "zero_shot_it",
    "zero_shot_ja",
    "zero_shot_ko",
    "zero_shot_ru",
    "zero_shot_zh",
]

CV3_OBJECTIVE_SUBSETS = [
    f"data/{split.replace('_to_', '/to_').replace('zero_shot_', 'zero_shot/').replace('emotion_zeroshot_', 'emotion_zeroshot/')}"
    for split in CV3_PARQUET_SPLITS
    if split.startswith(("zero_shot_", "cross_lingual_zeroshot_", "emotion_zeroshot_"))
]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_key_text(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            records[parts[0]] = parts[1] if len(parts) > 1 else ""
    return records


def read_two_column_text(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            key = parts[0]
            value = parts[1] if len(parts) > 1 else ""
            rows[key] = value
    return rows


def request_json(url: str, retries: int = 5) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, http.client.IncompleteRead) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}")


def github_content_text(path: str) -> str:
    sparse_path = CV3_SPARSE_ROOT / path
    if sparse_path.exists():
        return sparse_path.read_text(encoding="utf-8")
    source_path = CV3_SOURCE_ROOT / path
    if source_path.exists():
        return source_path.read_text(encoding="utf-8")
    local_path = CV3_ROOT / path.removeprefix("data/")
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")
    url = f"{CV3_GITHUB_API}/{path}?ref=main"
    try:
        data = request_json(url)
    except RuntimeError:
        raw_tmp = CV3_ROOT / ".tmp" / path.removeprefix("data/")
        download_file(f"https://raw.githubusercontent.com/FunAudioLLM/CV3-Eval/main/{path}", raw_tmp)
        return raw_tmp.read_text(encoding="utf-8")
    content = data.get("content", "")
    return base64.b64decode(content).decode("utf-8")


def ensure_cv3_source() -> Path:
    if (CV3_SPARSE_ROOT / "data").exists() and any((CV3_SPARSE_ROOT / "data").rglob("prompt_wav.scp")):
        return CV3_SPARSE_ROOT
    if (CV3_SOURCE_ROOT / "data").exists():
        return CV3_SOURCE_ROOT
    SOURCES_ROOT.mkdir(parents=True, exist_ok=True)
    download_file("https://codeload.github.com/FunAudioLLM/CV3-Eval/zip/refs/heads/main", CV3_ARCHIVE)
    with zipfile.ZipFile(CV3_ARCHIVE) as archive:
        archive.extractall(SOURCES_ROOT)
    if not (CV3_SOURCE_ROOT / "data").exists():
        raise FileNotFoundError(f"CV3 source archive did not extract data/: {CV3_SOURCE_ROOT}")
    return CV3_SOURCE_ROOT


def write_text_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def download_file(url: str, dst: Path, expected_size: int | None = None, retries: int = 5) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        if expected_size is None or dst.stat().st_size == expected_size:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    for attempt in range(retries):
        try:
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "-C",
                    "-",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "1800",
                    "--retry",
                    "2",
                    "--retry-delay",
                    "2",
                    "-o",
                    str(tmp),
                    url,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if expected_size is not None and tmp.stat().st_size != expected_size:
                raise RuntimeError(f"size mismatch {tmp.stat().st_size} != {expected_size}")
            tmp.replace(dst)
            return
        except Exception as exc:  # noqa: BLE001 - keep download retries broad.
            last_error = exc
            tmp.unlink(missing_ok=True)
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Failed to download {url} -> {dst}: {last_error}")


def download_hf_mirror_file(repo_path: str, dst: Path) -> None:
    download_file(f"{SEEDTTS_HF_MIRROR}/{repo_path}", dst)


def prepare_librispeech_other() -> int:
    csv_path = SURE_ROOT / "SURE_Test_csv" / "librispeech_test-other_ASR.csv"
    suite_dir = SURE_SUITES_ROOT / "librispeech-test-other"

    def rows() -> Iterable[dict[str, Any]]:
        import csv

        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for record in reader:
                rel = record["Audio:FILE"]
                audio_path = suite_dir / Path(rel).name
                yield {
                    "key": Path(rel).stem,
                    "path": str(audio_path.resolve()),
                    "target": record["Text:LABEL"].strip(),
                    "task": "ASR",
                    "language": "en",
                    "dataset": "librispeech_other",
                    "split": "test-other",
                }

    return write_jsonl(JSONL_ROOT / "librispeech_other.jsonl", rows())


def iter_wenetspeech_audios(json_path: Path) -> Iterable[dict[str, Any]]:
    """Stream WenetSpeech audios from the large top-level JSON file."""
    decoder = json.JSONDecoder()
    with json_path.open("r", encoding="utf-8") as handle:
        buffer = ""
        pos = 0
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise RuntimeError("Could not find WenetSpeech audios array")
            buffer += chunk
            key = buffer.find('"audios"')
            if key >= 0:
                bracket = buffer.find("[", key)
                if bracket >= 0:
                    pos = bracket + 1
                    break
                buffer = buffer[key:]
            elif len(buffer) > 100:
                buffer = buffer[-100:]

        while True:
            while True:
                while pos < len(buffer) and buffer[pos] in " \r\n\t,":
                    pos += 1
                if pos < len(buffer):
                    break
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    return
                buffer = ""
                pos = 0

            if buffer[pos] == "]":
                return

            while True:
                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        raise
                    buffer += chunk
                    continue
                yield obj
                buffer = buffer[end:]
                pos = 0
                break


def extract_audio_segment(src: Path, dst: Path, start: float, end: float) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, end - start)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(src),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def prepare_wenetspeech(split: str, limit: int | None = None) -> int:
    assert split in {"test_net", "test_meeting"}
    split_marker = "TEST_NET" if split == "test_net" else "TEST_MEETING"
    json_path = WENETSPEECH_ROOT / "data" / "WenetSpeech.json"

    def rows() -> Iterable[dict[str, Any]]:
        count = 0
        for audio in iter_wenetspeech_audios(json_path):
            audio_rel = str(audio.get("path", ""))
            segments = audio.get("segments") or []
            if f"audio/{split}/" not in audio_rel and not any(
                split_marker in [str(item).upper() for item in seg.get("subsets", [])]
                for seg in segments
            ):
                continue

            source_audio = WENETSPEECH_ROOT / "data" / audio_rel
            for segment in segments:
                subsets = [str(item).upper() for item in segment.get("subsets", [])]
                if split_marker not in subsets:
                    continue
                key = str(segment["sid"])
                begin = float(segment["begin_time"])
                end = float(segment["end_time"])
                target = str(segment.get("text", "")).strip()
                yield {
                    "key": key,
                    "path": str(source_audio.resolve()),
                    "target": target,
                    "task": "ASR",
                    "language": "zh",
                    "dataset": f"wenetspeech_{split}",
                    "split": split,
                    "source_audio": str(source_audio.resolve()),
                    "begin_time": begin,
                    "end_time": end,
                    "sample_rate": 16000,
                    "duration_ms": int(round((end - begin) * 1000)),
                }
                count += 1
                if limit is not None and count >= limit:
                    return

    return write_jsonl(JSONL_ROOT / f"wenetspeech_{split}.jsonl", rows())


def parse_ark_spec(spec: str) -> tuple[Path, int]:
    ark, offset = spec.rsplit(":", 1)
    ark_path = Path(ark)
    if not ark_path.exists() and str(ark_path).startswith("/mnt/lustre/sjtu/shared/"):
        ark_path = Path(str(ark_path).replace("/mnt/lustre/sjtu/shared", "/hpc_stor03/public/shared", 1))
    return ark_path, int(offset)


def export_kaldi_audio_object(ark_path: Path, offset: int, next_offset: int, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with ark_path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(next_offset - offset)
    flac_pos = payload.find(b"fLaC")
    riff_pos = payload.find(b"RIFF")
    if flac_pos >= 0:
        audio_payload = payload[flac_pos:]
    elif riff_pos >= 0:
        audio_payload = payload[riff_pos:]
    else:
        raise ValueError(f"Could not locate audio payload in {ark_path}:{offset}")
    dst.write_bytes(audio_payload)


def prepare_gigaspeech_test(limit: int | None = None) -> int:
    split_dir = GIGASPEECH_ROOT / "dump" / "raw" / "test"
    texts = read_key_text(split_dir / "text")
    wav_rows: list[tuple[str, Path, int]] = []
    with (split_dir / "wav.scp").open(encoding="utf-8") as handle:
        for line in handle:
            key, spec = line.strip().split(maxsplit=1)
            ark, offset = parse_ark_spec(spec)
            wav_rows.append((key, ark, offset))

    output_audio_root = DATASETS_ROOT / "gigaspeech_test" / "audio"

    def rows() -> Iterable[dict[str, Any]]:
        emitted = 0
        for index, (key, ark, offset) in enumerate(wav_rows):
            if key not in texts:
                continue
            next_offset = ark.stat().st_size
            if index + 1 < len(wav_rows) and wav_rows[index + 1][1] == ark:
                next_offset = wav_rows[index + 1][2]
            dst = output_audio_root / f"{key}.flac"
            export_kaldi_audio_object(ark, offset, next_offset, dst)
            yield {
                "key": key,
                "path": str(dst.resolve()),
                "target": texts[key].strip().lower(),
                "task": "ASR",
                "language": "en",
                "dataset": "gigaspeech_test",
                "split": "test",
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    return write_jsonl(JSONL_ROOT / "gigaspeech_test.jsonl", rows())


def prepare_slidespeech_test(limit: int | None = None) -> int:
    texts = read_key_text(SLIDESPEECH_INFO_ROOT / "text")
    sessions: dict[str, Path] = {}
    with (SLIDESPEECH_INFO_ROOT / "wav.scp").open(encoding="utf-8") as handle:
        for line in handle:
            session, rel_path = line.strip().split(maxsplit=1)
            sessions[session] = (SLIDESPEECH_AUDIO_ROOT / rel_path).resolve()

    segment_rows: list[tuple[str, str, float, float]] = []
    with (SLIDESPEECH_INFO_ROOT / "segments").open(encoding="utf-8") as handle:
        for line in handle:
            key, session, start, end = line.strip().split()
            segment_rows.append((key, session, float(start), float(end)))

    def rows() -> Iterable[dict[str, Any]]:
        emitted = 0
        for key, session, start, end in segment_rows:
            target = texts.get(key)
            source_audio = sessions.get(session)
            if target is None or source_audio is None:
                continue
            yield {
                "key": key,
                "path": str(source_audio),
                "target": target.strip(),
                "task": "ASR",
                "language": "en",
                "dataset": "slidespeech_test",
                "split": "test",
                "source_audio": str(source_audio),
                "begin_time": start,
                "end_time": end,
                "duration_ms": int(round((end - start) * 1000)),
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    return write_jsonl(JSONL_ROOT / "slidespeech_test.jsonl", rows())


def seedtts_dataset_name(split: str) -> str:
    return {
        "en": "seedtts_test_eval_en",
        "zh": "seedtts_test_eval_zh",
        "zh_hard": "seedtts_test_eval_zh_hard",
    }[split]


def seedtts_meta_path(split: str) -> tuple[str, str]:
    return {
        "en": ("en", "meta.lst"),
        "zh": ("zh", "meta.lst"),
        "zh_hard": ("zh", "hardcase.lst"),
    }[split]


def ensure_seedtts_meta(split: str) -> Path:
    lang_dir, filename = seedtts_meta_path(split)
    path = SEEDTTS_SOURCE_ROOT / lang_dir / filename
    if not path.exists() or path.stat().st_size == 0:
        download_hf_mirror_file(f"{lang_dir}/{filename}", path)
    return path


def parse_seedtts_meta(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                raise ValueError(f"Unexpected SeedTTS meta row in {path}: {line}")
            rows.append(
                {
                    "key": parts[0],
                    "prompt_text": parts[1],
                    "prompt_wav": parts[2],
                    "target": parts[3],
                }
            )
    return rows


def prepare_seedtts_split(split: str, limit: int | None = None) -> int:
    lang_dir, _ = seedtts_meta_path(split)
    language = "en" if lang_dir == "en" else "zh"
    dataset_name = seedtts_dataset_name(split)
    meta_path = ensure_seedtts_meta(split)
    meta_rows = parse_seedtts_meta(meta_path)

    def rows() -> Iterable[dict[str, Any]]:
        emitted = 0
        for record in meta_rows:
            prompt_rel = record["prompt_wav"]
            prompt_path = SEEDTTS_SOURCE_ROOT / lang_dir / prompt_rel
            if not prompt_path.exists() or prompt_path.stat().st_size == 0:
                download_hf_mirror_file(f"{lang_dir}/{prompt_rel}", prompt_path)
            sample_id = f"{dataset_name}_{record['key']}"
            yield {
                "key": sample_id,
                "sample_id": sample_id,
                "path": str(prompt_path.resolve()),
                "reference_audio": str(prompt_path.resolve()),
                "target": record["target"],
                "reference_text": record["target"],
                "prompt_text": record["prompt_text"],
                "task": "TTS",
                "language": language,
                "dataset": dataset_name,
                "source_dataset": "seedtts_test_eval",
                "subset": split,
                "split": split,
                "prompt_wav": prompt_rel,
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    return write_jsonl(JSONL_ROOT / f"{dataset_name}.jsonl", rows())


def prepare_seedtts_test_eval(limit: int | None = None) -> int:
    return sum(prepare_seedtts_split(split, limit=limit) for split in ("en", "zh", "zh_hard"))


def cv3_parquet_path(split: str) -> Path:
    return CV3_ROOT / "parquet" / f"{split}-00000-of-00001.parquet"


def download_cv3_parquet(split: str) -> Path:
    parquet_path = cv3_parquet_path(split)
    url = f"{CV3_HF_MIRROR}/data/{split}-00000-of-00001.parquet"
    download_file(url, parquet_path)
    return parquet_path


def cv3_language_from_split(split: str) -> str | None:
    if split.startswith("zero_shot_"):
        language = split.removeprefix("zero_shot_")
    elif split.startswith("cross_lingual_zeroshot_to_"):
        language = split.removeprefix("cross_lingual_zeroshot_to_")
    elif split.startswith("emotion_zeroshot_"):
        language = split.removeprefix("emotion_zeroshot_")
    else:
        return None
    if language.startswith("hard_"):
        language = language.removeprefix("hard_")
    return language


def cv3_language_from_text(text: str) -> str:
    if any("\u3040" <= char <= "\u30ff" for char in text):
        return "ja"
    if any("\uac00" <= char <= "\ud7af" for char in text):
        return "ko"
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "zh"
    return "en"


def cv3_dataset_name_for_split(split: str, language: str | None = None) -> str:
    if cv3_language_from_split(split) is not None or language is None:
        return f"cv3_eval_{split}"
    return f"cv3_eval_{split}_{language}"


def cv3_audio_extension(audio_path: str) -> str:
    suffix = Path(audio_path).suffix.lower()
    return suffix if suffix else ".wav"


def cv3_write_audio(audio: dict[str, Any], dst: Path) -> None:
    audio_bytes = audio.get("bytes")
    if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
        raise ValueError(f"CV3 parquet row does not contain prompt_audio bytes for {dst}")
    if dst.exists() and dst.stat().st_size == len(audio_bytes):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(audio_bytes)


def prepare_cv3_parquet_split(split: str, limit: int | None = None) -> dict[str, int]:
    import pyarrow.parquet as pq

    parquet_path = download_cv3_parquet(split)
    table = pq.read_table(parquet_path)
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    fixed_language = cv3_language_from_split(split)

    emitted = 0
    for record in table.to_pylist():
        target_text = str(record.get("target_text") or "")
        language = fixed_language or cv3_language_from_text(target_text)
        dataset_name = cv3_dataset_name_for_split(split, None if fixed_language else language)
        audio = record.get("prompt_audio")
        if not isinstance(audio, dict):
            raise ValueError(f"Unexpected CV3 prompt_audio value in {parquet_path}: {type(audio).__name__}")
        audio_name = Path(str(audio.get("path") or "")).name
        if not audio_name:
            audio_name = f"{record['id']}{cv3_audio_extension('')}"
        audio_path = CV3_ROOT / split / "audio" / audio_name
        cv3_write_audio(audio, audio_path)

        sample_id = f"{dataset_name}_{record['id']}"
        rows_by_dataset.setdefault(dataset_name, []).append(
            {
                "key": sample_id,
                "sample_id": sample_id,
                "path": str(audio_path.resolve()),
                "reference_audio": str(audio_path.resolve()),
                "target": target_text,
                "reference_text": target_text,
                "prompt_text": str(record.get("prompt_text") or ""),
                "task": "TTS",
                "language": language,
                "dataset": dataset_name,
                "source_dataset": "cv3_eval",
                "subset": split,
                "split": split,
                "prompt_wav": audio_name,
                "source_parquet": str(parquet_path.resolve()),
            }
        )
        emitted += 1
        if limit is not None and emitted >= limit:
            break

    counts: dict[str, int] = {}
    for dataset_name, dataset_rows in rows_by_dataset.items():
        counts[dataset_name] = write_jsonl(JSONL_ROOT / f"{dataset_name}.jsonl", dataset_rows)
    return counts


def cv3_candidate_split_for_dataset(dataset: str) -> tuple[str, str | None] | None:
    if not dataset.startswith("cv3_eval_"):
        return None
    suffix = dataset.removeprefix("cv3_eval_")
    for split in sorted(CV3_PARQUET_SPLITS, key=len, reverse=True):
        fixed_language = cv3_language_from_split(split)
        if fixed_language is not None and suffix == split:
            return split, None
        prefix = f"{split}_"
        if fixed_language is None and suffix.startswith(prefix):
            return split, suffix.removeprefix(prefix)
        if fixed_language is None and suffix == split:
            return split, None
    return None


def prepare_cv3_dataset(dataset: str, limit: int | None = None) -> int:
    candidate = cv3_candidate_split_for_dataset(dataset)
    if candidate is None:
        raise ValueError(f"Unknown CV3 parquet dataset: {dataset}")
    split, language = candidate
    counts = prepare_cv3_parquet_split(split, limit=limit)
    if language is None:
        return sum(counts.values())
    return counts.get(dataset, 0)


def cv3_dataset_name(subset_path: str) -> str:
    suffix = subset_path.removeprefix("data/").replace("/", "_")
    return f"cv3_eval_{suffix}"


def cv3_language(subset_path: str) -> str:
    parts = subset_path.split("/")
    leaf = parts[-1]
    if leaf.startswith("hard_"):
        leaf = leaf.removeprefix("hard_")
    if leaf.startswith("to_hard_"):
        leaf = leaf.removeprefix("to_hard_")
    elif leaf.startswith("to_"):
        leaf = leaf.removeprefix("to_")
    if leaf in {"angry", "happy", "sad"} and len(parts) >= 2:
        leaf = parts[-2]
    return "zh" if leaf.startswith(("zh", "cmn", "yue")) else "en"


def cv3_subset_kind(subset_path: str) -> str:
    return subset_path.split("/")[1]


def cv3_local_prompt_path(remote_path: str, subset_path: str) -> Path:
    remote = remote_path.replace("//", "/")
    if remote.startswith("data/"):
        return CV3_ROOT / remote.removeprefix("data/")
    return CV3_ROOT / subset_path.removeprefix("data/") / Path(remote).name


def cv3_download_subset(subset_path: str, limit: int | None = None) -> int:
    source_root = ensure_cv3_source()
    local_subset = CV3_ROOT / subset_path.removeprefix("data/")
    for filename in ("text", "prompt_text", "prompt_wav.scp"):
        source_file = source_root / subset_path / filename
        if source_file.exists():
            write_text_if_changed(local_subset / filename, source_file.read_text(encoding="utf-8"))
        else:
            write_text_if_changed(local_subset / filename, github_content_text(f"{subset_path}/{filename}"))

    text_rows = read_two_column_text(local_subset / "text")
    prompt_text_rows = read_two_column_text(local_subset / "prompt_text")
    prompt_wav_rows = read_two_column_text(local_subset / "prompt_wav.scp")

    dataset_name = cv3_dataset_name(subset_path)
    language = cv3_language(subset_path)
    split = subset_path.removeprefix("data/")

    def rows() -> Iterable[dict[str, Any]]:
        emitted = 0
        for key in text_rows:
            remote_wav = prompt_wav_rows.get(key)
            if not remote_wav:
                continue
            remote_wav = remote_wav.replace("//", "/")
            local_wav = cv3_local_prompt_path(remote_wav, subset_path)
            source_wav = source_root / remote_wav
            if not source_wav.exists():
                raise FileNotFoundError(f"CV3 prompt audio not found in source archive: {source_wav}")
            local_wav.parent.mkdir(parents=True, exist_ok=True)
            if not local_wav.exists() or local_wav.stat().st_size != source_wav.stat().st_size:
                shutil.copy2(source_wav, local_wav)
            prompt_text = prompt_text_rows.get(key, "")
            sample_id = f"{dataset_name}_{key}"
            yield {
                "key": sample_id,
                "sample_id": sample_id,
                "path": str(local_wav.resolve()),
                "reference_audio": str(local_wav.resolve()),
                "target": text_rows[key],
                "reference_text": text_rows[key],
                "prompt_text": prompt_text,
                "task": "TTS",
                "language": language,
                "dataset": dataset_name,
                "source_dataset": "cv3_eval",
                "subset": split,
                "split": split,
                "prompt_wav": remote_wav,
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    return write_jsonl(JSONL_ROOT / f"{dataset_name}.jsonl", rows())


def prepare_cv3_eval(limit: int | None = None) -> int:
    total = 0
    for split in CV3_PARQUET_SPLITS:
        total += sum(prepare_cv3_parquet_split(split, limit=limit).values())
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=[
            "librispeech_other",
            "wenetspeech_test_net",
            "wenetspeech_test_meeting",
            "gigaspeech_test",
            "slidespeech_test",
            "seedtts_test_eval",
            "cv3_eval",
        ],
    )
    parser.add_argument("--limit", type=int, help="Debug limit per generated dataset")
    args = parser.parse_args()

    prepared: dict[str, int] = {}
    for dataset in args.dataset:
        if dataset == "librispeech_other":
            prepared[dataset] = prepare_librispeech_other()
        elif dataset == "wenetspeech_test_net":
            prepared[dataset] = prepare_wenetspeech("test_net", args.limit)
        elif dataset == "wenetspeech_test_meeting":
            prepared[dataset] = prepare_wenetspeech("test_meeting", args.limit)
        elif dataset == "gigaspeech_test":
            prepared[dataset] = prepare_gigaspeech_test(args.limit)
        elif dataset == "slidespeech_test":
            prepared[dataset] = prepare_slidespeech_test(args.limit)
        elif dataset == "seedtts_test_eval":
            prepared[dataset] = prepare_seedtts_test_eval(args.limit)
        elif dataset in {"seedtts_test_eval_en", "seedtts_test_eval_zh", "seedtts_test_eval_zh_hard"}:
            split = dataset.removeprefix("seedtts_test_eval_")
            prepared[dataset] = prepare_seedtts_split(split, args.limit)
        elif dataset == "cv3_eval":
            prepared[dataset] = prepare_cv3_eval(args.limit)
        elif dataset.startswith("cv3_eval_"):
            prepared[dataset] = prepare_cv3_dataset(dataset, args.limit)
        else:
            raise ValueError(f"Unknown local dataset: {dataset}")
    print(json.dumps({"prepared": prepared}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
