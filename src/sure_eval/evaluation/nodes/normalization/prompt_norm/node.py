"""Normalize prompt-based classification answers to comparable choices."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from string import punctuation
from typing import Any

from sure_eval.evaluation.core.types import KeyTextFiles, PipelineNodeResult

NODE_ID = "normalization/prompt_norm"
NODE_VERSION = "v1"


@dataclass(frozen=True)
class Choice:
    """One selectable answer option."""

    id: str
    text: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptChoiceSpec:
    """Prompt choices indexed by sample key."""

    choices_by_key: dict[str, tuple[Choice, ...]]
    prompt_by_key: dict[str, str] = field(default_factory=dict)


def normalize_prompt_choice_files(
    files: KeyTextFiles,
    *,
    prompt_jsonl: str,
    output_mode: str = "choice_id",
    ambiguous_policy: str = "unresolved",
) -> tuple[KeyTextFiles, PipelineNodeResult]:
    """Normalize reference and hypothesis key-text files using prompt choices.

    ``output_mode`` may be ``choice_id`` for new pipelines or ``choice_text``
    for compatibility with the old SLU evaluator.
    """

    if output_mode not in {"choice_id", "choice_text"}:
        raise ValueError(f"Unsupported prompt_norm output_mode: {output_mode}")
    if ambiguous_policy not in {"unresolved", "first_match"}:
        raise ValueError(f"Unsupported ambiguous_policy: {ambiguous_policy}")

    spec = load_prompt_choice_spec(prompt_jsonl)
    ref_file = _new_temp_file()
    hyp_file = _new_temp_file()
    ref_rows, ref_details = _normalize_file(
        files.ref_file,
        spec,
        output_file=ref_file,
        output_mode=output_mode,
        ambiguous_policy=ambiguous_policy,
    )
    hyp_rows, hyp_details = _normalize_file(
        files.hyp_file,
        spec,
        output_file=hyp_file,
        output_mode=output_mode,
        ambiguous_policy=ambiguous_policy,
    )
    return (
        KeyTextFiles(ref_file=ref_file, hyp_file=hyp_file),
        PipelineNodeResult(
            stage="normalization",
            node_id=NODE_ID,
            version=NODE_VERSION,
            details={
                "prompt_jsonl": prompt_jsonl,
                "output_mode": output_mode,
                "ambiguous_policy": ambiguous_policy,
                "ref_file": ref_file,
                "hyp_file": hyp_file,
                "num_rows": {"ref": len(ref_rows), "hyp": len(hyp_rows)},
                "num_choices_by_key": {
                    key: len(choices) for key, choices in spec.choices_by_key.items()
                },
                "ref_rows": ref_details,
                "hyp_rows": hyp_details,
            },
            internal_stages=(
                "prompt_choice_loading",
                "choice_id_match",
                "choice_text_match",
                "alias_match",
                "raw_fallback",
            ),
        ),
    )


def load_prompt_choice_spec(prompt_jsonl: str | Path) -> PromptChoiceSpec:
    """Load structured or legacy prompt choices from JSONL."""

    choices_by_key: dict[str, tuple[Choice, ...]] = {}
    prompt_by_key: dict[str, str] = {}
    with Path(prompt_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            key = str(item["key"])
            prompt = str(item.get("prompt") or "")
            prompt_by_key[key] = prompt
            choices = _choices_from_item(item)
            if choices:
                choices_by_key[key] = tuple(choices)
    return PromptChoiceSpec(choices_by_key=choices_by_key, prompt_by_key=prompt_by_key)


def _choices_from_item(item: dict[str, Any]) -> list[Choice]:
    choices = item.get("choices") or item.get("options")
    if isinstance(choices, dict):
        return [Choice(id=str(key), text=str(value)) for key, value in choices.items()]
    if isinstance(choices, list):
        parsed: list[Choice] = []
        for index, choice in enumerate(choices, start=1):
            if isinstance(choice, dict):
                choice_id = choice.get("id", index)
                text = choice.get("text", choice.get("label", choice.get("value", "")))
                aliases = choice.get("aliases", ())
                if isinstance(aliases, str):
                    aliases = (aliases,)
                parsed.append(
                    Choice(
                        id=str(choice_id),
                        text=str(text),
                        aliases=tuple(str(alias) for alias in aliases),
                    )
                )
            else:
                parsed.append(Choice(id=str(index), text=str(choice)))
        return parsed
    prompt = str(item.get("prompt") or "")
    return _choices_from_prompt(prompt)


def _choices_from_prompt(prompt: str) -> list[Choice]:
    parsed: list[Choice] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^\s*([A-Za-z]+|\d+|[①-⑳])\s*[\.\):：、]\s*(.+?)\s*$", stripped)
        if match:
            parsed.append(Choice(id=match.group(1), text=match.group(2).strip()))
    return parsed


def _normalize_file(
    input_file: str,
    spec: PromptChoiceSpec,
    *,
    output_file: str,
    output_mode: str,
    ambiguous_policy: str,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    key2full: dict[str, str] = {}
    key_order: list[str] = []
    with open(input_file, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            key, value = line.split("\t", 1)
            if key not in key2full:
                key_order.append(key)
                key2full[key] = ""
            key2full[key] = f"{key2full[key]} {value.strip()}".strip()

    rows: list[tuple[str, str]] = []
    details: list[dict[str, Any]] = []
    for key in key_order:
        raw = key2full[key]
        normalized, detail = normalize_prompt_answer(
            key,
            raw,
            spec.choices_by_key.get(key, ()),
            output_mode=output_mode,
            ambiguous_policy=ambiguous_policy,
        )
        rows.append((key, normalized))
        details.append(detail)
    _write_rows(output_file, rows)
    return rows, details


def normalize_prompt_answer(
    key: str,
    raw: str,
    choices: tuple[Choice, ...],
    *,
    output_mode: str = "choice_id",
    ambiguous_policy: str = "unresolved",
) -> tuple[str, dict[str, Any]]:
    """Normalize one raw answer against arbitrary prompt choices."""

    if not choices:
        return raw.strip(), _detail(key, raw, None, "fallback_raw", output=raw.strip())

    matches = _candidate_matches(raw, choices)
    if not matches:
        return raw.strip(), _detail(key, raw, None, "fallback_raw", output=raw.strip())
    if len(matches) > 1 and ambiguous_policy == "unresolved":
        return raw.strip(), _detail(
            key,
            raw,
            None,
            "ambiguous",
            output=raw.strip(),
            matched_choice_ids=[choice.id for choice in matches],
        )

    choice = matches[0]
    output = choice.text if output_mode == "choice_text" else choice.id
    return output, _detail(key, raw, choice, "matched_choice", output=output)


def _candidate_matches(raw: str, choices: tuple[Choice, ...]) -> list[Choice]:
    raw_stripped = raw.strip()
    raw_norm = _norm(raw_stripped)
    if not raw_norm:
        return []

    matches: list[Choice] = []
    for choice in choices:
        id_norm = _norm(choice.id)
        if raw_norm == id_norm or _contains_choice_id(raw_stripped, choice.id):
            matches.append(choice)
    if matches:
        return _dedupe(matches)

    for choice in choices:
        values = (choice.text, *choice.aliases)
        for value in values:
            value_norm = _norm(value)
            if raw_norm == value_norm or (value_norm and value_norm in raw_norm):
                matches.append(choice)
                break
    return _dedupe(matches)


def _contains_choice_id(raw: str, choice_id: str) -> bool:
    escaped = re.escape(choice_id.strip())
    if not escaped:
        return False
    if re.fullmatch(r"[A-Za-z0-9_+-]+", choice_id.strip()):
        return re.search(rf"(?<![A-Za-z0-9_+-]){escaped}(?![A-Za-z0-9_+-])", raw, re.IGNORECASE) is not None
    return choice_id.strip() in raw


def _norm(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(punctuation + "，。！？；：（）【】《》“”‘’、")
    return normalized


def _dedupe(choices: list[Choice]) -> list[Choice]:
    seen: set[str] = set()
    result: list[Choice] = []
    for choice in choices:
        if choice.id not in seen:
            result.append(choice)
            seen.add(choice.id)
    return result


def _detail(
    key: str,
    raw: str,
    choice: Choice | None,
    method: str,
    *,
    output: str,
    matched_choice_ids: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "key": key,
        "raw": raw,
        "output": output,
        "method": method,
    }
    if choice is not None:
        payload["choice_id"] = choice.id
        payload["choice_text"] = choice.text
    if matched_choice_ids is not None:
        payload["matched_choice_ids"] = matched_choice_ids
    return payload


def _new_temp_file() -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    path = handle.name
    handle.close()
    return path


def _write_rows(path: str, rows: list[tuple[str, str]]) -> None:
    Path(path).write_text("".join(f"{key}\t{value}\n" for key, value in rows), encoding="utf-8")
