# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""CLI help and usage strings."""

from __future__ import annotations

from typing import Dict, Mapping

from text_normalization.distribution import get_distribution_policy

CLI_TEXT: Dict[str, Mapping[str, str]] = {
    "description": {
        "zh": (
            "文本正规化工具。通常按各语种默认配置运行即可。\n"
            "下面列出 {language_count} 个语种。TTS 请用对应的 xx_tts 代号（与下表重复，故不列出）。\n"
            "混合语种只支持英文+一个其它语种：cs_xx-en / mix_xx-en。实际走非英语种 xx 的处理，并保留英文字符。\n"
            "{language_table}"
        ),
        "en": (
            "XNorm: multilingual text normalization. Language defaults are usually enough.\n"
            "{language_count} language(s) below. For TTS, use xx_tts (same as the base row, so omitted).\n"
            "Mixed IDs cs_xx-en / mix_xx-en mean English plus one other language; the non-English pipeline runs and English letters are kept.\n"
            "{language_table}"
        ),
    },
    "info": {
        "zh": "查询语种信息，输出 JSON。带语种代号只返回该语种，否则返回全部。",
        "en": "Print language metadata as JSON. Pass a language code for one entry, or omit it for all.",
    },
    "language": {
        "zh": "语种代号，例如：{language_codes}。混合语种写法：cs_de-en、mix_zh-en。",
        "en": "Language code, for example: {language_codes}. Mixed IDs look like cs_de-en or mix_zh-en.",
    },
    "input_file": {
        "zh": "输入文件；省略则从标准输入读取。第一列是句子 ID 时加 -w 1。",
        "en": "Input file; omit to read stdin. Use -w 1 if the first column is an utterance ID.",
    },
    "output_file": {
        "zh": "输出文件；省略则写到标准输出。",
        "en": "Output file; omit to write stdout.",
    },
    "with_id_opt": {
        "zh": "第一列是否为句子 ID。[default: 0]",
        "en": "Whether the first column is an utterance ID. [default: 0]",
    },
    "debug": {
        "zh": "调试输出。0 不打印；1 打印有变化的步骤；2 再打印 Unicode。NeMo 同样：0 不打官方 INFO，1/2 打开详细日志。[default: 0]",
        "en": "Debug output. 0 silent; 1 print steps that change the text; 2 also print Unicode. Same for NeMo: 0 hides official INFO, 1/2 verbose. [default: 0]",
    },
    "mode": {
        "zh": "使用场景，必须指定。asr_eval 识别结果；asr_train 训练文本；tts 合成文本。",
        "en": "Scene mode: asr_eval for ASR output, asr_train for training text, tts for TTS input.",
    },
    "code_switch_tag": {
        "zh": "语种切换标记：delete 全删；keep_start 只留开头标记；keep_start_base 只留基础语种开头标记。asr_eval 默认 delete，asr_train/tts 默认 keep_start_base。",
        "en": "Code-switch tags: delete all; keep_start keep the opening tag; keep_start_base keep the base-language opening tag. Default: delete for asr_eval, keep_start_base for asr_train/tts.",
    },
    "case": {
        "zh": "大小写。upper 大写；lower 小写；none 或空表示不转换。德语、土耳其语有特殊规则。[default: none]",
        "en": "Case: upper, lower, or none/empty to leave unchanged. German and Turkish have special rules. [default: none]",
    },
    "normalize_text": {
        "zh": "是否走 NeMo TN 和 NumberGrammar。0 仍做清洗，但不把数字和长尾转成读法。不影响 --itn。[default: 1]",
        "en": "Run NeMo TN and NumberGrammar. 0 keeps cleanup but leaves numbers and long-tail text as written. Does not affect --itn. [default: 1]",
    },
    "itn": {
        "zh": "是否做 ITN（把读法写回书面形式）。只走 NeMo。[default: 0]",
        "en": "Inverse text normalization (spoken form to written form). Uses NeMo only. [default: 0]",
    },
    "nemo_backend": {
        "zh": "NeMo 后端。official 官方实现；direct 加速构图，必要时回退 official。--itn 0 走 TN，1 走 ITN。默认 direct。",
        "en": "NeMo backend. official uses upstream graphs; direct compiles faster graphs and falls back to official. --itn 0 runs TN, 1 runs ITN. Default: direct.",
    },
    "nemo_input_case": {
        "zh": "NeMo TN 的输入大小写：cased 或 lower_cased。[default: cased]",
        "en": "NeMo TN input case: cased or lower_cased. [default: cased]",
    },
    "nemo_itn_input_case": {
        "zh": "NeMo ITN 的输入大小写。[default: lower_cased]",
        "en": "NeMo ITN input case. [default: lower_cased]",
    },
    "nemo_cache_dir": {
        "zh": "NeMo 缓存目录，用来保存已编译的图。[default: <project>/.cache/nemo]",
        "en": "NeMo cache directory for compiled graphs. [default: <project>/.cache/nemo]",
    },
    "nemo_overwrite_cache": {
        "zh": "是否重建并覆盖 NeMo 缓存。[default: 0]",
        "en": "Rebuild and overwrite the NeMo cache. [default: 0]",
    },
    "nemo_whitelist": {
        "zh": "NeMo 白名单文件，用于额外的替换规则。",
        "en": "Extra NeMo whitelist replacements.",
    },
    "normalize_digit_maxlen": {
        "zh": "按位朗读数字的最大位数，超过则不再这样转换。[default: 12]",
        "en": "Max length for digit-by-digit reading; longer strings are left as-is. [default: 12]",
    },
    "keep_empty_lines": {
        "zh": "是否保留空行。[default: 1]",
        "en": "Keep empty lines. [default: 1]",
    },
    "keep_english_letters": {
        "zh": "是否保留英文字母。混合语种 ID 会强制保留。[default: 1]",
        "en": "Keep English letters. Mixed language IDs force this on. [default: 1]",
    },
    "remove_lines": {
        "zh": (
            "行内出现非法字符时是否丢掉整行，例如日语里出现韩文。\n"
            "英文字母默认算合法；要当成非法请加 --keep_english_letters 0。\n"
            "[default: 0]"
        ),
        "en": (
            "Drop the whole line if it contains an illegal character, for example Hangul inside Japanese.\n"
            "English letters are legal unless you pass --keep_english_letters 0.\n"
            "[default: 0]"
        ),
    },
    "remove_dashes": {
        "zh": "是否删除短横 \"-\"。0 只删不是连字符的；1 全部删除。[default: 0]",
        "en": 'Remove hyphens "-". 0 keeps word-internal hyphens; 1 removes all. [default: 0]',
    },
    "remove_single_quotes": {
        "zh": "是否删除单引号。0 只删不是单词内撇号的；1 全部删除。[default: 0]",
        "en": "Remove single quotes. 0 keeps word-internal apostrophes; 1 removes all. [default: 0]",
    },
    "remove_brackets": {
        "zh": "是否删除圆括号及其中内容，多是注释（如 Common Voice）。[default: 1]",
        "en": "Remove parentheses and the text inside them, often comments such as in Common Voice. [default: 1]",
    },
    "chinese_conversion": {
        "zh": "中文/粤语繁简转换。默认繁体→简体（t2s），也支持台湾、香港。[default: t2s]",
        "en": "Chinese/Cantonese Traditional↔Simplified conversion. Default t2s (Traditional to Simplified); Taiwan and Hong Kong variants are supported. [default: t2s]",
    },
    "json_input": {
        "zh": "输入是否为 JSONL。每行一个对象，读取 text_raw 字段。[default: 0]",
        "en": "JSONL input. Each line is an object; the text_raw field is normalized. [default: 0]",
    },
    "json_output": {
        "zh": "输出是否为 JSONL。每行 {\"text_raw\": \"...\", \"text_norm\": \"...\"}。[default: 0]",
        "en": 'JSONL output. Each line is {"text_raw": "...", "text_norm": "..."}. [default: 0]',
    },
    "json": {
        "zh": "同时用 JSONL 输入和输出（--json-input 与 --json-output 的简写）。[default: 0]",
        "en": "JSONL input and output (shorthand for --json-input and --json-output). [default: 0]",
    },
    "inverse_segmentation": {
        "zh": "逆分词：去掉中/日/藏/缅/粤等无空格文字内部的分词空格，并在与英文或数字相邻处补一个空格。只写开关、不写 0/1 时等于 1。[default: 0]",
        "en": "Undo segmentation: strip extra spaces inside unspaced scripts (Chinese, Japanese, Tibetan, Burmese, Cantonese) and insert a space next to English or digits. A bare flag means 1. [default: 0]",
    },
    "error_python_version": {
        "zh": "需要 Python 3.9 或更高版本",
        "en": "Python 3.9 or newer is required",
    },
    "error_missing_language": {
        "zh": "做文本正规化时必须指定语种（<language>）",
        "en": "Language is required when normalizing text (<language>)",
    },
    "error_missing_mode": {
        "zh": "必须指定 -m/--mode。asr_train 和 asr_eval 在部分语种上行为不同，请按场景选择 asr_train、asr_eval 或 tts。",
        "en": "-m/--mode is required. Choose asr_train, asr_eval, or tts.",
    },
    "error_unknown_language": {
        "zh": "没有找到语种代号 {language} 的信息",
        "en": "No language metadata for code {language}",
    },
    "error_file_failed": {
        "zh": "处理文件失败：{path}",
        "en": "Failed to process file: {path}",
    },
    "error_detail": {
        "zh": "错误信息：{error}",
        "en": "Error: {error}",
    },
}


def cli_locale() -> str:
    """Return the help locale for this build."""
    return "en" if get_distribution_policy().public_build else "zh"


def cli_text(key: str, **kwargs: object) -> str:
    """Return one catalog string for the current build locale."""
    try:
        variants = CLI_TEXT[key]
    except KeyError as exc:
        raise KeyError(f"CLI text {key!r} is not in the catalog") from exc
    locale = cli_locale()
    try:
        template = variants[locale]
    except KeyError as exc:
        raise KeyError(f"CLI text {key!r} is missing locale {locale!r}") from exc
    if kwargs:
        return template.format(**kwargs)
    return template
