"""Convert key-text files back to STM using sidecar metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def convert_txt_to_stm(
    *,
    input_txt: str,
    sidecar_json: str,
    output_stm: str,
    conversion_id: str,
) -> dict[str, Any]:
    """Write STM rows from key-text rows and sidecar timing metadata."""

    sidecar = json.loads(Path(sidecar_json).read_text(encoding="utf-8"))
    row_by_key = {str(row["key"]): row for row in sidecar.get("rows") or []}
    text_by_key: dict[str, str] = {}
    with open(input_txt, "r", encoding="utf-8") as handle:
        for line in handle:
            if "\t" not in line:
                continue
            key, text = line.rstrip("\n").split("\t", 1)
            text_by_key[key] = text

    stm_rows: list[str] = []
    for key, metadata in row_by_key.items():
        text = text_by_key.get(key, "")
        stm_rows.append(
            " ".join(
                [
                    str(metadata["session_id"]),
                    str(metadata["channel"]),
                    str(metadata["speaker_id"]),
                    str(metadata["start"]),
                    str(metadata["end"]),
                    text,
                ]
            )
        )

    Path(output_stm).write_text("\n".join(stm_rows) + ("\n" if stm_rows else ""), encoding="utf-8")
    return {
        "id": conversion_id,
        "source_format": "key_text",
        "target_format": "stm",
        "script": "src/sure_eval/evaluation/conversion/sa_asr__cpwer/txt_to_stm.py",
        "sidecar": sidecar_json,
        "output": output_stm,
        "num_rows": len(stm_rows),
        "affects_metric": True,
    }
