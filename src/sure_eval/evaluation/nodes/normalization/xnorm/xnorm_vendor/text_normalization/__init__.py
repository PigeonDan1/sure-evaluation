# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Multilingual text-normalization package: register languages and export runtime entrypoints."""

import sys,os
import importlib
import json
from types import MappingProxyType
from .logger import logger
from .distribution import (
    default_nemo_backend,
    filter_distribution_kwargs,
    get_distribution_policy,
    get_language_capabilities,
    validate_distribution_request,
)
from .core import (
    CompiledPipeline,
    LocaleProfile,
    ModePolicy,
    NormalizationRequest,
    NormalizationResult,
    NormalizerRegistry,
)

### Auto-register languages below; do not edit this loop.

current_dir = os.path.dirname(__file__)
lang_dir = os.path.join(current_dir, "lang")
# LANG_CLASSES is read-only metadata; runtime must construct via LANG_CLASS_TYPES.
LANG_CLASSES = {}
LANG_CLASS_TYPES = {}
CODE_SWITCH_LANGUAGE_PREFIX = "cs_"
MIX_LANGUAGE_PREFIX = "mix_"
CODE_SWITCH_LANGUAGE_PREFIXES = (CODE_SWITCH_LANGUAGE_PREFIX, MIX_LANGUAGE_PREFIX)

# Language modules live in lang/; xx.py defines TextNormalization_XX.
for fname in os.listdir(lang_dir):
    if not fname.endswith(".py"):
        continue
    if fname in ("__init__.py", "template.py"):
        continue

    mod_name = fname[:-3]
    lang_code = mod_name.lower()
    expected_class_name = f"TextNormalization_{lang_code.upper()}"

    try:
        module = importlib.import_module(f".lang.{mod_name}", __package__)
        if hasattr(module, expected_class_name):
            cls = getattr(module, expected_class_name)
            metadata_instance = cls()
            LANG_CLASSES[lang_code] = metadata_instance
            LANG_CLASS_TYPES[lang_code] = cls
        else:
            print(f"Warning: {expected_class_name} not found in lang.{mod_name}")
    except Exception as e:
        print(f"Error loading lang.{mod_name}: {e}")


LANG_CLASSES = MappingProxyType(LANG_CLASSES)
LANG_CLASS_TYPES = MappingProxyType(LANG_CLASS_TYPES)
NORMALIZER_REGISTRY = NormalizerRegistry(LANG_CLASS_TYPES, LANG_CLASSES)


def create_normalizer(language: str):
    """Create an independent normalizer per request so config and counters cannot leak."""
    return NORMALIZER_REGISTRY.create(language)


def _log_text_change(text: str, normalized_text: str, debug: int):
    if debug > 0 and text != normalized_text:
        logger.debug(f"text_i: {text}")
        logger.debug(f"text_o: {normalized_text}")


def _is_inverse_segment_char(char: str) -> bool:
    """Whether a character belongs to an unspaced script whose internal spaces should be removed."""
    code = ord(char)
    return (
        0x0F00 <= code <= 0x0FFF       # Tibetan
        or 0x3040 <= code <= 0x309F    # Hiragana
        or 0x30A0 <= code <= 0x30FF    # Katakana
        or 0x31F0 <= code <= 0x31FF    # Katakana Phonetic Extensions
        or 0x3400 <= code <= 0x4DBF    # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF    # CJK Unified Ideographs
        or 0xA960 <= code <= 0xA97F    # Myanmar Extended-B
        or 0xF900 <= code <= 0xFAFF    # CJK Compatibility Ideographs
        or 0xFE30 <= code <= 0xFE4F    # CJK Compatibility Forms
        or 0xFF65 <= code <= 0xFF9F    # Halfwidth Katakana
        or 0x20000 <= code <= 0x2A6DF  # CJK Extension B
        or 0x2A700 <= code <= 0x2B73F  # CJK Extension C
        or 0x2B740 <= code <= 0x2B81F  # CJK Extension D
        or 0x2B820 <= code <= 0x2CEAF  # CJK Extension E/F
        or 0x2CEB0 <= code <= 0x2EBEF  # CJK Extension F/I
        or 0x30000 <= code <= 0x3134F  # CJK Extension G/H
        or 0x1000 <= code <= 0x109F    # Myanmar
        or 0xAA60 <= code <= 0xAA7F    # Myanmar Extended-A
    )


def _is_spaced_writing_char(char: str) -> bool:
    """Whether a character is a letter/digit that needs a space against an unspaced script."""
    return char.isalnum() and not _is_inverse_segment_char(char)


def _needs_inverse_segment_boundary(left: str, right: str) -> bool:
    """Keep one boundary space where an unspaced script meets a spaced script, so code-switch tokens do not glue."""
    return (
        _is_inverse_segment_char(left) and _is_spaced_writing_char(right)
    ) or (
        _is_spaced_writing_char(left) and _is_inverse_segment_char(right)
    )


def _append_single_space(output: list[str]) -> None:
    if output and not output[-1].isspace():
        output.append(" ")


def inverse_segment_non_spaced_scripts(text: str) -> str:
    """Remove internal spaces in unspaced scripts and restore unspaced/spaced script boundaries."""
    chars = list(text)
    output = []
    index = 0

    while index < len(chars):
        char = chars[index]
        if char.isspace():
            end = index + 1
            while end < len(chars) and chars[end].isspace():
                end += 1
            if (
                output
                and end < len(chars)
                and _is_inverse_segment_char(output[-1])
                and _is_inverse_segment_char(chars[end])
            ):
                index = end
                continue
            if (
                output
                and end < len(chars)
                and _needs_inverse_segment_boundary(output[-1], chars[end])
            ):
                _append_single_space(output)
                index = end
                continue
            output.extend(chars[index:end])
            index = end
            continue

        if (
            output
            and not output[-1].isspace()
            and _needs_inverse_segment_boundary(output[-1], char)
        ):
            _append_single_space(output)
        output.append(char)
        index += 1

    return "".join(output)


def _get_code_switch_language_body(language):
    """Return the mixed-language id body without prefix; ordinary languages return None."""
    if not isinstance(language, str):
        return None
    # Czech TTS is registered as cs_tts; resolve that before the virtual cs_ prefix.
    if language in LANG_CLASS_TYPES:
        return None
    for prefix in CODE_SWITCH_LANGUAGE_PREFIXES:
        if language.startswith(prefix):
            return language[len(prefix):]
    return None


def resolve_code_switch_base_language(language):
    """Parse cs_/mix_ language ids and return the non-English base language to reuse."""
    language_body = _get_code_switch_language_body(language)
    if language_body is None:
        return None

    lang_pair = language_body.split("-")
    if len(lang_pair) != 2 or "en" not in lang_pair or lang_pair[0] == lang_pair[1]:
        raise ValueError(
            f"code-switch language IDs only support English plus one other language; the mix_ prefix follows the same rule, e.g. cs_de-en or mix_zh-en: {language}"
        )

    target_languages = [lang for lang in lang_pair if lang != "en"]
    return target_languages[0]


def _resolve_code_switch_language(kwargs):
    """Parse the minimal code-switch/mix language id; only English plus one existing language."""
    language = kwargs.get('language', 'en')
    target_language = resolve_code_switch_base_language(language)
    if target_language is None:
        return language

    if target_language not in LANG_CLASS_TYPES:
        raise ValueError(f"Not supported language: {target_language}")

    kwargs["language"] = target_language
    # cs_/mix_ reuse the xx pipeline and must keep English so user flags cannot delete English spans.
    kwargs["keep_english_letters"] = True
    return target_language


def create_compiled_pipeline(
    requested_language: str = None,
    **kwargs,
) -> CompiledPipeline:
    """Snapshot legacy kwargs into an immutable request and compile once for the line loop."""

    filtered_kwargs = filter_distribution_kwargs(kwargs)
    language_option = filtered_kwargs.get("language")
    if requested_language is None:
        requested_language = language_option
    if requested_language is None:
        raise ValueError("language must not be empty")
    if language_option is not None and language_option != requested_language:
        raise ValueError("language does not match requested_language")
    validate_distribution_request(requested_language, filtered_kwargs.get("mode"))
    effective_kwargs = dict(filtered_kwargs)
    effective_kwargs["language"] = requested_language
    effective_language = _resolve_code_switch_language(effective_kwargs)
    request = NormalizationRequest.create(
        requested_language=requested_language,
        language=effective_language,
        options=effective_kwargs,
    )
    return NORMALIZER_REGISTRY.compile(request)


def _normalize_text(pipeline, text: str, debug: int, inverse_segmentation: bool = False) -> str:
    normalized_text = pipeline.normalize_text(text)
    if inverse_segmentation:
        # Merge intra-script spaces before ASR/CER scoring, but keep cross-script boundaries.
        normalized_text = inverse_segment_non_spaced_scripts(normalized_text)
    _log_text_change(text, normalized_text, debug)
    return normalized_text


def _normalize_json_item(item, pipeline, debug: int, inverse_segmentation: bool = False):
    if not isinstance(item, dict) or 'text_raw' not in item:
        raise ValueError(
            f"JSON object is missing the 'text_raw' field.\n"
            f"Currently using --json-input mode; every JSONL line must contain a 'text_raw' field.\n"
            f"Received JSON object: {item}"
        )

    output_item = dict(item)
    text = output_item['text_raw']
    output_item['text_norm'] = _normalize_text(pipeline, text, debug, inverse_segmentation)
    return output_item


def _load_json_input(infile):
    has_valid_line = False

    for i, line in enumerate(infile, start=1):
        line = line.strip()
        if not line:
            continue

        has_valid_line = True
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSONL parse failed: line {i} is not valid JSON.\n"
                f"Currently using --json-input mode.\n"
                f"JSONL format: one JSON object per line, e.g. {{\"text_raw\": \"text\"}}\n"
                f"Bad line: {line[:100]}{'...' if len(line) > 100 else ''}\n"
                f"Error: {str(e)}"
            ) from e

        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid JSONL: line {i} is not a JSON object.\n"
                f"Currently using --json-input mode.\n"
                f"JSONL format: one JSON object per line, e.g. {{\"text_raw\": \"text\"}}\n"
                f"Received type: {type(item).__name__}"
            )

        yield item

    if not has_valid_line:
        raise ValueError(
            f"Input file is empty.\n"
            f"Currently using --json-input mode, which requires JSONL with one JSON object per line."
        )


def _process_json_io(
    infile,
    outfile,
    pipeline,
    debug: int,
    json_output: bool,
    inverse_segmentation: bool = False,
    flush_each_line: bool = True,
):
    for item in _load_json_input(infile):
        output_item = _normalize_json_item(item, pipeline, debug, inverse_segmentation)
        if json_output:
            outfile.write(json.dumps(output_item, ensure_ascii=False) + '\n')
        else:
            outfile.write(output_item['text_norm'] + '\n')
        if flush_each_line:
            outfile.flush()


def _process_text_io(
    infile,
    outfile,
    pipeline,
    with_id_opt,
    keep_empty_lines,
    debug: int,
    json_output: bool,
    inverse_segmentation: bool = False,
    flush_each_line: bool = True,
):
    for line in infile:
        line = line.strip()

        if with_id_opt == 1:
            if not line and keep_empty_lines == 0:
                continue

            parts = line.split(maxsplit=1)
            if not parts:
                index = ""
                text = ""
            elif len(parts) < 2:
                index = parts[0]
                text = ""
            else:
                index, text = parts
        else:
            text = line

        normalized_text = _normalize_text(pipeline, text, debug, inverse_segmentation)

        if json_output:
            item = {'text_raw': text, 'text_norm': normalized_text}
            if with_id_opt == 1:
                item['sample_id'] = index # Default field name is sample_id
            outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            if with_id_opt == 1:
                outfile.write(f"{index} {normalized_text}\n")
            elif keep_empty_lines == 1 or normalized_text:
                outfile.write(f"{normalized_text}\n")

        if flush_each_line:
            outfile.flush()

# File-processing entry below
def text_normalization(
    input_file: str,
    output_file: str,
    **kwargs
):
    """
    Clean text file according to the specified method and language.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        language (str): Language code.
        keep_empty_lines (bool, optional): Whether to keep empty lines. Defaults to True.
        debug (int, optional): Whether to print debug info. Defaults to False.
        json_input (bool, optional): Whether input is JSON format. Defaults to False.
        json_output (bool, optional): Whether output is JSON format. Defaults to False.
    """

    kwargs = filter_distribution_kwargs(kwargs)
    validate_distribution_request(kwargs.get("language"), kwargs.get("mode"))
    requested_language = kwargs.get("language", "en")
    with_id_opt = kwargs.get("with_id_opt")
    keep_empty_lines = kwargs.get("keep_empty_lines")
    debug = kwargs.get("debug")
    json_input = kwargs.get("json_input", False)
    json_output = kwargs.get("json_output", False)
    inverse_segmentation = kwargs.get("inverse_segmentation", False)

    if debug > 0: 
        print(kwargs, file=sys.stderr)
    # Request, profile, and mode policy are parsed once before the line loop.
    pipeline = create_compiled_pipeline(requested_language, **kwargs)

    # Handle input
    if input_file and os.path.exists(input_file):
        infile = open(input_file, 'r', encoding='utf-8-sig')
        need_close_infile = True
    else:
        infile = sys.stdin
        need_close_infile = False

    # Handle output
    if output_file:
        outfile = open(output_file, 'w', encoding='utf-8')
        need_close_outfile = True
    else:
        outfile = sys.stdout
        need_close_outfile = False

    try:
        if json_input: # JSON format
            _process_json_io(
                infile, 
                outfile, 
                pipeline,
                debug, 
                json_output,
                inverse_segmentation,
                # Files flush on close; stdout still shows each line as before.
                flush_each_line=not need_close_outfile,
            )
        else:
            _process_text_io( # Plain-text format
                infile,
                outfile,
                pipeline,
                with_id_opt,
                keep_empty_lines,
                debug,
                json_output,
                inverse_segmentation,
                flush_each_line=not need_close_outfile,
            )
    finally:
        if need_close_infile:
            infile.close()
        if need_close_outfile:
            outfile.close()
