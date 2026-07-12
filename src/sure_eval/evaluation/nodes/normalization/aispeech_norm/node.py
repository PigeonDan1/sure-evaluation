"""AISpeech-compatible ASR normalization wrappers.

This node preserves the normalization behavior currently embedded in
``SUREEvaluator._eval_asr`` and ``SUREEvaluator._eval_asr_codeswitch``.
"""

from __future__ import annotations

import glob
import os
import tempfile
from pathlib import Path

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.nodes.normalization.aispeech_norm.normalization_impl import default_map_dir
from sure_eval.evaluation.sure_evaluator import (
    _strip_eval_punct_file,
    split_tokens,
    tokenize_codeswitch,
)

NODE_ID = "normalization/aispeech_norm"
NODE_VERSION = "v1"


def _default_map_dir() -> str:
    return default_map_dir()


def normalize_asr_files(files: KeyTextFiles, *, language: str) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize key-text ASR reference and hypothesis files."""

    map_dir = _default_map_dir()
    norm_context = _make_norm_context(language, map_dir)

    ref_norm_file = _new_temp_file()
    hyp_norm_file = _new_temp_file()
    try:
        ref_stats = _write_normalized_file(
            files.ref_file, ref_norm_file, language=language, norm_context=norm_context
        )
        hyp_stats = _write_normalized_file(
            files.hyp_file, hyp_norm_file, language=language, norm_context=norm_context
        )
        _strip_eval_punct_file(ref_norm_file)
        _strip_eval_punct_file(hyp_norm_file)
    except Exception:
        Path(ref_norm_file).unlink(missing_ok=True)
        Path(hyp_norm_file).unlink(missing_ok=True)
        raise

    return (
        KeyTextFiles(ref_file=ref_norm_file, hyp_file=hyp_norm_file),
        PipelineNodeResult(
            stage="normalization",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "language": language,
                "profile": language,
                "text_normalizer": "asr_num2words",
                "punctuation_policy": "SUREEvaluator._strip_eval_punct_file",
                "ref_file": ref_norm_file,
                "hyp_file": hyp_norm_file,
                "row_stats": {"ref": ref_stats, "hyp": hyp_stats},
            },
            internal_stages=("number_text_normalization", "punctuation_stripping"),
        ),
    )


def normalize_codeswitch_asr_files(files: KeyTextFiles) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize code-switch ASR files into mixed, zh-only, and en-only key-text files."""

    map_dir = _default_map_dir()

    class Preprocessor:
        def __init__(self, lang: str):
            self.lang = lang
            self.norm_context = _make_norm_context(lang, map_dir)

        def normalize(self, text: str) -> str:
            return _normalize_text(text, self.lang, self.norm_context)

    proc_en = Preprocessor("en")
    proc_zh = Preprocessor("zh")
    ref_lines, ref_zh_lines, ref_en_lines, ref_stats = _codeswitch_rows(files.ref_file, proc_en, proc_zh)
    hyp_lines, hyp_zh_lines, hyp_en_lines, hyp_stats = _codeswitch_rows(files.hyp_file, proc_en, proc_zh)

    temp_paths: list[str] = []
    try:
        outputs = {
            "ref_file": _write_temp_rows(ref_lines, temp_paths),
            "hyp_file": _write_temp_rows(hyp_lines, temp_paths),
            "ref_zh_file": _write_temp_rows(ref_zh_lines, temp_paths),
            "hyp_zh_file": _write_temp_rows(hyp_zh_lines, temp_paths),
            "ref_en_file": _write_temp_rows(ref_en_lines, temp_paths),
            "hyp_en_file": _write_temp_rows(hyp_en_lines, temp_paths),
        }
    except Exception:
        for path in temp_paths:
            Path(path).unlink(missing_ok=True)
        raise

    return (
        KeyTextFiles(ref_file=outputs["ref_file"], hyp_file=outputs["hyp_file"]),
        PipelineNodeResult(
            stage="normalization",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "language": "cs",
                "profile": "cs",
                "text_normalizer": "asr_num2words",
                "side_outputs": outputs,
                "row_stats": {"ref": ref_stats, "hyp": hyp_stats},
            },
            internal_stages=("codeswitch_tokenization", "number_text_normalization", "language_split"),
        ),
    )


def _new_temp_file() -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    path = handle.name
    handle.close()
    return path


def _load_num2words_maps(language: str, map_dir: str) -> tuple[list, list]:
    """Preload the map tables ``asr_num2words`` would otherwise re-read per call.

    Mirrors the function's own fallback lookup: non-digit ``<map_dir>/*.map``
    files for text replacement, then ``<map_dir>/<language>/digit.map`` (or the
    root ``digit.map``) for digit-by-digit conversion.
    """

    from sure_eval.evaluation.nodes.normalization.aispeech_norm.normalization_impl.asr_simple_tn import (
        load_and_sort_map,
    )

    other_map_sorted: list = []
    for map_file in sorted(glob.glob(os.path.join(map_dir, "*.map"))):
        if os.path.basename(map_file) != "digit.map":
            other_map_sorted.extend(load_and_sort_map(map_file, language))
    other_map_sorted.sort(key=lambda pair: len(pair[0]), reverse=True)

    lang_map_file = os.path.join(map_dir, language, "digit.map")
    root_map_file = os.path.join(map_dir, "digit.map")
    if os.path.exists(lang_map_file):
        digit_map_file = lang_map_file
    elif os.path.exists(root_map_file):
        digit_map_file = root_map_file
    else:
        digit_map_file = None
    digit_map_sorted = load_and_sort_map(digit_map_file, language) if digit_map_file else []
    return other_map_sorted, digit_map_sorted


def _make_norm_context(language: str, map_dir: str) -> dict:
    """Bundle preloaded maps and a shared number cache for repeated calls."""

    other_map_sorted, digit_map_sorted = _load_num2words_maps(language, map_dir)
    return {
        "map_dir": map_dir,
        "other_map_sorted": other_map_sorted,
        "digit_map_sorted": digit_map_sorted,
        "cached_num_map": {},
    }


def _normalize_text(text: str, language: str, norm_context: dict) -> str:
    from sure_eval.evaluation.nodes.normalization.aispeech_norm.normalization_impl.asr_simple_tn import asr_num2words

    return asr_num2words(
        text,
        language,
        map_dir=norm_context["map_dir"],
        debug=False,
        other_map_sorted=norm_context["other_map_sorted"],
        digit_map_sorted=norm_context["digit_map_sorted"],
        cached_num_map=norm_context["cached_num_map"],
    )


def _write_normalized_file(input_file: str, output_file: str, *, language: str, norm_context: dict) -> dict[str, int]:
    rows: list[str] = []
    num_empty_text_rows = 0
    num_dropped_malformed_rows = 0
    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            key, text = _parse_key_text_line(line)
            if key is None:
                if text is not None:
                    num_dropped_malformed_rows += 1
                continue
            if text:
                text_norm = _normalize_text(text, language, norm_context)
            else:
                num_empty_text_rows += 1
                text_norm = ""
            rows.append(f"{key}\t{text_norm}")
    _require_parsed_rows(input_file, num_rows=len(rows), num_malformed=num_dropped_malformed_rows)
    Path(output_file).write_text("\n".join(rows) + "\n" if rows else "", encoding="utf-8")
    return {
        "num_rows": len(rows),
        "num_empty_text_rows": num_empty_text_rows,
        "num_dropped_malformed_rows": num_dropped_malformed_rows,
    }


def _parse_key_text_line(line: str) -> tuple[str | None, str | None]:
    """Parse one ``<key>\\t<text>`` row; empty text is a valid recognition result.

    Returns ``(key, text)`` for parsed rows, ``(None, None)`` for blank lines,
    and ``(None, raw)`` for malformed rows without a tab separator or key.
    """

    raw = line.rstrip("\r\n")
    if not raw.strip():
        return None, None
    if "\t" not in raw:
        return None, raw
    key, text = raw.split("\t", 1)
    key = key.strip()
    if not key:
        return None, raw
    return key, text.strip()


def _require_parsed_rows(input_file: str, *, num_rows: int, num_malformed: int) -> None:
    if num_rows == 0 and num_malformed > 0:
        raise ValueError(
            f"No <key>\\t<text> rows could be parsed from {input_file!r} "
            f"({num_malformed} non-empty lines without a tab separator). "
            "ASR key-text inputs must be tab-separated."
        )


def _codeswitch_rows(input_file: str, proc_en, proc_zh) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    mixed_lines: list[str] = []
    zh_lines: list[str] = []
    en_lines: list[str] = []
    num_empty_text_rows = 0
    num_dropped_malformed_rows = 0
    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            key, text = _parse_key_text_line(line)
            if key is None:
                if text is not None:
                    num_dropped_malformed_rows += 1
                continue
            if not text:
                num_empty_text_rows += 1
            tokens = tokenize_codeswitch(text, proc_en, proc_zh) if text else []
            zh_tokens, en_tokens = split_tokens(tokens)
            mixed_lines.append(f"{key}\t{' '.join(tokens)}")
            zh_lines.append(f"{key}\t{' '.join(zh_tokens)}")
            en_lines.append(f"{key}\t{' '.join(en_tokens)}")
    _require_parsed_rows(input_file, num_rows=len(mixed_lines), num_malformed=num_dropped_malformed_rows)
    stats = {
        "num_rows": len(mixed_lines),
        "num_empty_text_rows": num_empty_text_rows,
        "num_dropped_malformed_rows": num_dropped_malformed_rows,
    }
    return mixed_lines, zh_lines, en_lines, stats


def _write_temp_rows(rows: list[str], temp_paths: list[str]) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    handle.write("\n".join(rows) + "\n")
    handle.close()
    temp_paths.append(handle.name)
    return handle.name
