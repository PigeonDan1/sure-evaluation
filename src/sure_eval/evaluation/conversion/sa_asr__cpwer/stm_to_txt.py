"""Convert STM annotation files to key-text files with sidecar metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def convert_stm_to_txt(
    *,
    input_stm: str,
    output_txt: str,
    sidecar_json: str,
    conversion_id: str,
) -> dict[str, Any]:
    """Write ``key<TAB>text`` rows from STM six-field rows."""

    rows: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    with open(input_stm, "r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            parts = line.strip().split(maxsplit=5)
            if len(parts) != 6:
                continue
            session_id, channel, speaker_id, start, end, transcript = parts
            key = "|".join([session_id, channel, speaker_id, start, end, str(row_index)])
            rows.append(f"{key}\t{transcript}")
            metadata_rows.append(
                {
                    "key": key,
                    "session_id": session_id,
                    "channel": channel,
                    "speaker_id": speaker_id,
                    "start": start,
                    "end": end,
                    "row_index": row_index,
                }
            )

    Path(output_txt).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    sidecar = {
        "schema": "sure.conversion.stm_sidecar.v1",
        "conversion_id": conversion_id,
        "source_format": "stm",
        "target_format": "key_text",
        "rows": metadata_rows,
    }
    Path(sidecar_json).write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "id": conversion_id,
        "source_format": "stm",
        "target_format": "key_text",
        "script": "src/sure_eval/evaluation/conversion/sa_asr__cpwer/stm_to_txt.py",
        "sidecar": sidecar_json,
        "output": output_txt,
        "num_rows": len(metadata_rows),
        "affects_metric": True,
    }
