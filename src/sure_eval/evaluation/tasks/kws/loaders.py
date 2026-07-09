"""KWS input loaders for supported file formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sure_eval.evaluation.nodes.scoring.wekws_det.metrics import KWSSample, normalize_keyword


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _expected_from_reference(row: dict[str, Any]) -> tuple[bool, str | None]:
    expected_value = row.get("expected", row.get("label", row.get("expected_detected")))
    if isinstance(expected_value, bool):
        expected_detected = expected_value
    elif expected_value is None:
        expected_detected = str(row.get("txt", row.get("text", ""))).strip() != ""
    else:
        expected_detected = str(expected_value).strip().lower() in {
            "detect",
            "detected",
            "positive",
            "true",
            "1",
            "yes",
        }
    expected_keyword = row.get("expected_keyword")
    if expected_keyword is None and expected_detected:
        expected_keyword = row.get("text", row.get("txt"))
    return expected_detected, expected_keyword


def _result_payload(output: dict[str, Any]) -> dict[str, Any]:
    result = output.get("result", output)
    if not isinstance(result, dict):
        raise ValueError(f"KWS output for key={output.get('key')} is not an object")
    return result


def load_samples_from_jsonl_and_outputs(
    reference_jsonl: Path | str,
    output_json: Path | str,
) -> list[KWSSample]:
    """Load SURE fixture JSONL and wrapper sample_output.json."""
    references = {row["key"]: row for row in _load_jsonl(Path(reference_jsonl))}
    outputs = json.loads(Path(output_json).read_text(encoding="utf-8"))
    if isinstance(outputs, dict) and "rows" in outputs:
        outputs = outputs["rows"]
    if not isinstance(outputs, list):
        raise ValueError("KWS output JSON must be a list or an object with rows")

    samples: list[KWSSample] = []
    for output in outputs:
        key = output["key"]
        if key not in references:
            raise KeyError(f"KWS output key not found in reference: {key}")
        ref = references[key]
        expected_detected, expected_keyword = _expected_from_reference(ref)
        result = _result_payload(output)
        detected = bool(result.get("detected", False))
        score = result.get("score")
        samples.append(
            KWSSample(
                key=key,
                expected_detected=expected_detected,
                expected_keyword=expected_keyword,
                duration=ref.get("duration"),
                detected=detected,
                predicted_keyword=result.get("keyword"),
                score=float(score) if score is not None else None,
                metadata={
                    "audio": ref.get("audio", ref.get("wav")),
                    "raw": result.get("raw", {}),
                },
            )
        )
    return samples


def load_samples_from_wekws_score_file(
    label_file: Path | str,
    score_file: Path | str,
    *,
    keyword: str,
) -> list[KWSSample]:
    """Load utterance-level WekWS CTC score output."""
    labels = {row["key"]: row for row in _load_jsonl(Path(label_file))}
    outputs: dict[str, dict[str, Any]] = {}
    with Path(score_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            key = parts[0]
            if len(parts) >= 4 and parts[1] == "detected":
                outputs[key] = {
                    "detected": True,
                    "keyword": parts[2],
                    "score": float(parts[3]),
                }
            else:
                outputs[key] = {
                    "detected": False,
                    "keyword": None,
                    "score": None,
                }

    samples: list[KWSSample] = []
    for key, ref in labels.items():
        output = outputs.get(key)
        if output is None:
            raise KeyError(f"KWS score key not found for reference: {key}")
        ref_text = str(ref.get("txt", ref.get("text", "")))
        expected_detected = normalize_keyword(keyword) in normalize_keyword(ref_text) if ref_text else False
        samples.append(
            KWSSample(
                key=key,
                expected_detected=bool(expected_detected),
                expected_keyword=keyword if expected_detected else None,
                duration=ref.get("duration"),
                detected=bool(output["detected"]),
                predicted_keyword=output.get("keyword"),
                score=output.get("score"),
                metadata={"wav": ref.get("wav", ref.get("audio"))},
            )
        )
    return samples


def load_samples_from_wekws_frame_score_file(
    label_file: Path | str,
    score_file: Path | str,
    *,
    keyword: str,
    threshold: float = 0.5,
) -> list[KWSSample]:
    """Load frame-level WekWS ``score.py`` output for one keyword."""
    labels = {row["key"]: row for row in _load_jsonl(Path(label_file))}
    score_table: dict[str, list[float]] = {}
    with Path(score_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            key = parts[0]
            current_keyword = parts[1]
            if normalize_keyword(current_keyword) != normalize_keyword(keyword):
                continue
            score_table[key] = [float(item) for item in parts[2:]]

    samples: list[KWSSample] = []
    for key, ref in labels.items():
        if key not in score_table:
            raise KeyError(f"KWS frame score key not found for reference: {key}")
        frame_scores = score_table[key]
        max_score = max(frame_scores) if frame_scores else 0.0
        ref_text = str(ref.get("txt", ref.get("text", "")))
        expected_detected = normalize_keyword(keyword) in normalize_keyword(ref_text) if ref_text else False
        samples.append(
            KWSSample(
                key=key,
                expected_detected=bool(expected_detected),
                expected_keyword=keyword if expected_detected else None,
                duration=ref.get("duration"),
                detected=max_score >= threshold,
                predicted_keyword=keyword if max_score >= threshold else None,
                score=max_score,
                scores=frame_scores,
                metadata={"wav": ref.get("wav", ref.get("audio"))},
            )
        )
    return samples


__all__ = [
    "load_samples_from_wekws_frame_score_file",
    "load_samples_from_jsonl_and_outputs",
    "load_samples_from_wekws_score_file",
]
