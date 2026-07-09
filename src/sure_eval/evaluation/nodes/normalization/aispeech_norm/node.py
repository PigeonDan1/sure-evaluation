"""AISpeech-compatible ASR normalization wrappers.

This node preserves the normalization behavior currently embedded in
``SUREEvaluator._eval_asr`` and ``SUREEvaluator._eval_asr_codeswitch``.
"""

from __future__ import annotations

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

    ref_norm_file = _new_temp_file()
    hyp_norm_file = _new_temp_file()
    try:
        _write_normalized_file(files.ref_file, ref_norm_file, language=language, map_dir=map_dir)
        _write_normalized_file(files.hyp_file, hyp_norm_file, language=language, map_dir=map_dir)
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
            },
            internal_stages=("number_text_normalization", "punctuation_stripping"),
        ),
    )


def normalize_codeswitch_asr_files(files: KeyTextFiles) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize code-switch ASR files into mixed, zh-only, and en-only key-text files."""

    from sure_eval.evaluation.nodes.normalization.aispeech_norm.normalization_impl.asr_simple_tn import asr_num2words

    map_dir = _default_map_dir()

    class Preprocessor:
        def __init__(self, lang: str):
            self.lang = lang

        def normalize(self, text: str) -> str:
            return asr_num2words(text, self.lang, map_dir=map_dir, debug=False)

    proc_en = Preprocessor("en")
    proc_zh = Preprocessor("zh")
    ref_lines, ref_zh_lines, ref_en_lines = _codeswitch_rows(files.ref_file, proc_en, proc_zh)
    hyp_lines, hyp_zh_lines, hyp_en_lines = _codeswitch_rows(files.hyp_file, proc_en, proc_zh)

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
            },
            internal_stages=("codeswitch_tokenization", "number_text_normalization", "language_split"),
        ),
    )


def _new_temp_file() -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    path = handle.name
    handle.close()
    return path


def _write_normalized_file(input_file: str, output_file: str, *, language: str, map_dir: str) -> None:
    from sure_eval.evaluation.nodes.normalization.aispeech_norm.normalization_impl.asr_simple_tn import asr_num2words

    rows = []
    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                key, text = parts
                text_norm = asr_num2words(text, language, map_dir=map_dir, debug=False)
                rows.append(f"{key}\t{text_norm}")
    Path(output_file).write_text("\n".join(rows) + "\n", encoding="utf-8")


def _codeswitch_rows(input_file: str, proc_en, proc_zh) -> tuple[list[str], list[str], list[str]]:
    mixed_lines: list[str] = []
    zh_lines: list[str] = []
    en_lines: list[str] = []
    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            key, text = parts
            tokens = tokenize_codeswitch(text, proc_en, proc_zh)
            zh_tokens, en_tokens = split_tokens(tokens)
            mixed_lines.append(f"{key}\t{' '.join(tokens)}")
            zh_lines.append(f"{key}\t{' '.join(zh_tokens)}")
            en_lines.append(f"{key}\t{' '.join(en_tokens)}")
    return mixed_lines, zh_lines, en_lines


def _write_temp_rows(rows: list[str], temp_paths: list[str]) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    handle.write("\n".join(rows) + "\n")
    handle.close()
    temp_paths.append(handle.name)
    return handle.name
