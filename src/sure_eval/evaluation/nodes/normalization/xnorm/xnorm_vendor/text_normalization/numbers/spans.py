# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Hand numeric spans to NumberGrammar; no language special cases here."""

from dataclasses import dataclass
import re
from typing import Callable, Match, Optional, Tuple

from ..logger import logger
from .engine import digits_to_words, is_number_language_supported, number_to_words
from .protocol import UnsupportedNumberConversionError

NUMBER_SPAN_PATTERN = re.compile(r"\d+(?:,\d\d\d)*(?:\.\d*)?")

# (text, match) -> (spoken, new_end). If later context is consumed, new_end must cover it or the cache poisons bare numbers.
ClassifyMatch = Callable[
    [str, Match[str]],
    Optional[Tuple[str, int]],
]


def _increment_counter(counters, name, amount=1):
    if counters is not None:
        counters.increment(name, amount)


@dataclass(frozen=True)
class NumberSpanPolicy:
    """Global number-span contract, not a language-specific reading rule."""

    max_len: int = 12

    def should_read_digits(self, num: str) -> bool:
        if not num.isdigit():
            return False
        if num[0] == "0":
            return True
        return (
            len(num) < self.max_len
            and len(num) > 4
            and (10 ** (len(num) - 1) != int(num))
        )


def _cardinal_or_original(num, language, debug, counters):
    try:
        result = number_to_words(
            num,
            language=language,
            conversion="cardinal",
            debug=debug,
        )
        if not result:
            raise ValueError("conversion result is empty")
        return result
    except UnsupportedNumberConversionError:
        _increment_counter(counters, "fallbacks")
        logger.warning(
            f"unsupported language/conversion combination: num = {num}, lang = {language}, to = cardinal"
        )
        return str(num)
    except Exception as exc:
        _increment_counter(counters, "fallbacks")
        logger.warning(
            f"number conversion failed: num = {num}, lang = {language}, reason: {exc}",
            exc_info=debug,
        )
        return str(num)


def rewrite_number_spans(
    text,
    language,
    debug=False,
    cached_num_map=None,
    max_len=12,
    counters=None,
    classify_match: Optional[ClassifyMatch] = None,
):
    """Replace digit strings only. Language context (year, percent) comes from classify_match."""

    policy = NumberSpanPolicy(max_len=max_len)
    if debug:
        logger.debug(f"normalize_digit_maxlen: {max_len}")

    text_ori = text
    if cached_num_map is None:
        cached_num_map = {}

    matches = list(NUMBER_SPAN_PATTERN.finditer(text))
    _increment_counter(counters, "number_matches", len(matches))
    if not matches:
        return text

    supported = is_number_language_supported(language)
    pieces = []
    pre_pos = 0
    for match in matches:
        num_raw = match.group()
        cur_start = match.start()
        cur_end = match.end()
        if cur_start < pre_pos:
            raise ValueError("overlapping number spans")
        num = num_raw.rstrip(".")
        replacement = num_raw

        if policy.should_read_digits(num):
            if len(num) > policy.max_len:
                _increment_counter(counters, "fallbacks")
                if debug or len(num) > 12:
                    logger.warning(
                        f"number length > threshold {policy.max_len}, "
                        f'failed to convert "{num}"'
                    )
            elif num_raw in cached_num_map:
                _increment_counter(counters, "number_cache_hits")
                replacement = cached_num_map[num_raw]
            else:
                _increment_counter(counters, "number_cache_misses")
                try:
                    converted = digits_to_words(num, language)
                except Exception as exc:
                    _increment_counter(counters, "fallbacks")
                    logger.warning(
                        f"digit-by-digit conversion failed: num = {num}, lang = {language}, "
                        f"reason: {exc}"
                    )
                    converted = num
                cached_num_map[num_raw] = converted
                replacement = converted
            if debug:
                logger.debug(f"fun_i: digit_sequence {num_raw}")
                logger.debug(f"fun_o: digit_sequence {replacement}")
        else:
            stripped = num.replace(",", "")
            if len(stripped) > policy.max_len:
                _increment_counter(counters, "fallbacks")
                if debug or len(stripped) > 12:
                    logger.warning(
                        f"number length > threshold {policy.max_len}, "
                        f'failed to convert "{stripped}"'
                    )
                replacement = stripped
            else:
                extra = (
                    classify_match(text, match)
                    if classify_match is not None
                    else None
                )
                if extra is not None:
                    spoken, new_end = extra
                    cache_key = text[cur_start:new_end]
                    if cache_key in cached_num_map:
                        _increment_counter(counters, "number_cache_hits")
                        replacement = cached_num_map[cache_key]
                    else:
                        _increment_counter(counters, "number_cache_misses")
                        cached_num_map[cache_key] = spoken
                        replacement = spoken
                    cur_end = new_end
                elif not supported:
                    _increment_counter(counters, "fallbacks")
                    replacement = stripped
                elif num_raw in cached_num_map:
                    _increment_counter(counters, "number_cache_hits")
                    replacement = cached_num_map[num_raw]
                else:
                    _increment_counter(counters, "number_cache_misses")
                    value = stripped
                    try:
                        value = int(stripped)
                    except ValueError:
                        value = float(stripped)
                    if debug:
                        logger.debug(f"before_num2words: {value}")
                    spoken = _cardinal_or_original(
                        value, language, debug, counters
                    )
                    if debug:
                        logger.debug(f"after_num2words: {spoken}")
                    replacement = " " + spoken + " "
                    cached_num_map[num_raw] = replacement

        pieces.append(text[pre_pos:cur_start] + replacement)
        pre_pos = cur_end
    pieces.append(text[pre_pos:])
    text = "".join(pieces)
    if debug and text != text_ori:
        logger.debug(f"fun_i: {text_ori}")
        logger.debug(f"fun_o: {text}")
    return text
