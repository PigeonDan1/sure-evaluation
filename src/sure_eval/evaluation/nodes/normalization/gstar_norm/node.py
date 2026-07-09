"""G-STAR-compatible text normalization."""

from __future__ import annotations

import tempfile
from pathlib import Path

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult
from sure_eval.evaluation.sure_evaluator import _normalize_text

NODE_ID = "normalization/gstar_norm"
NODE_VERSION = "v1"


def normalize_gstar_sa_asr_files(
    files: KeyTextFiles,
    *,
    language: str = "en",
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize key-text files using the G-STAR SA-ASR text rule."""

    ref_file = _new_temp_txt()
    hyp_file = _new_temp_txt()
    ref_rows = _normalize_key_text_file(files.ref_file, ref_file, language=language)
    hyp_rows = _normalize_key_text_file(files.hyp_file, hyp_file, language=language)
    return (
        KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file),
        PipelineNodeResult(
            stage="normalization",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "language": language,
                "input_schema": "key_text_files",
                "output_schema": "key_text_files",
                "ref_file": ref_file,
                "hyp_file": hyp_file,
                "num_rows": {"ref": len(ref_rows), "hyp": len(hyp_rows)},
                "normalization": {
                    "backend": "gstar_norm",
                    "case_sensitive": False,
                    "remove_tag": True,
                    "text_rule": "SUREEvaluator._normalize_text",
                },
                "ref_rows": ref_rows,
                "hyp_rows": hyp_rows,
            },
            internal_stages=("key_text_parse", "gstar_text_normalize"),
        ),
    )


def _normalize_key_text_file(input_file: str, output_file: str, *, language: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if "\t" not in line:
                continue
            key, original_text = line.rstrip("\n").split("\t", 1)
            normalized_text = _normalize_text(
                original_text,
                case_sensitive=False,
                remove_tag=True,
                language=language,
            )
            fout.write(f"{key}\t{normalized_text}\n")
            rows.append(
                {
                    "key": key,
                    "original_text": original_text,
                    "normalized_text": normalized_text,
                }
            )
    return rows


def _new_temp_txt() -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    path = handle.name
    handle.close()
    return path


def cleanup_gstar_norm_outputs(trace: tuple[PipelineNodeResult, ...]) -> None:
    """Remove temp files emitted by this node."""

    for result in trace:
        if result.node_id != NODE_ID:
            continue
        for key in ("ref_file", "hyp_file"):
            value = result.details.get(key)
            if isinstance(value, str):
                Path(value).unlink(missing_ok=True)
