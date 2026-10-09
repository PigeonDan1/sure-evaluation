# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""NeMo direct-output acceleration."""

from dataclasses import dataclass
from functools import lru_cache
import importlib
import inspect
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .nemo import (
    DEFAULT_NEMO_CACHE_DIR,
    NEMO_ITN_LANGUAGES,
    NemoDependencyError,
    NemoFarError,
    NemoInverseNormalizerConfig,
    NemoLanguageNotSupportedError,
    NemoNormalizerConfig,
    NemoTextNormalizationEngine,
    _fst_call_or_original,
    _validate_text,
    _validated_texts,
    get_nemo_engine,
    get_nemo_itn_engine,
    call_with_cache_rebuild,
    hash_optional_file,
    hash_source,
    make_cache_fingerprint,
    nemo_dependency_versions,
    prepare_nemo_cache_file,
    resolve_nemo_cache_layout,
    unique_nemo_cache_tmp,
)


DEFAULT_DIRECT_ITN_RULE = "tokenize_and_classify"
DIRECT_ITN_LANGUAGES = NEMO_ITN_LANGUAGES
DIRECT_TN_LANGUAGES = frozenset({"en", "de", "es", "fr", "it", "pt", "sv", "hu", "vi", "hy", "ko", "ar", "hi", "ja", "zh"})

# German official TN is spaced (ein und zwanzig / ein hundert); do not reuse compact NumberGrammar.
_DE_SMALL = (
    "null",
    "eins",
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
    "zehn",
    "elf",
    "zwölf",
    "dreizehn",
    "vierzehn",
    "fünfzehn",
    "sechzehn",
    "siebzehn",
    "achtzehn",
    "neunzehn",
)
_DE_TENS = {
    2: "zwanzig",
    3: "dreißig",
    4: "vierzig",
    5: "fünfzig",
    6: "sechzig",
    7: "siebzig",
    8: "achtzig",
    9: "neunzig",
}
_DE_ORDINAL_SMALL = {
    1: "erste",
    2: "zweite",
    3: "dritte",
    4: "vierte",
    5: "fünfte",
    6: "sechste",
    7: "siebte",
    8: "achte",
    9: "neunte",
    10: "zehnte",
    11: "elfte",
    12: "zwölfte",
    13: "dreizehnte",
    14: "vierzehnte",
    15: "fünfzehnte",
    16: "sechzehnte",
    17: "siebzehnte",
    18: "achtzehnte",
    19: "neunzehnte",
}
_DE_MEASURE_UNITS = {
    "kg": "kilogramm",
    "g": "gramm",
    "mg": "milligramm",
    "km": "kilometer",
    "cm": "zentimeter",
    "mm": "millimeter",
    "m": "meter",
}
_DE_DECIMAL_RE = re.compile(r"^(-?)(\d+),(\d+)$")
_DE_PERCENT_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?%$")
_DE_MEASURE_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?(kg|g|mg|km|cm|mm|m)$")
_DE_DOT_MEASURE_RE = re.compile(r"^(-?)(\d+)\.(\d+)\s(kg|g|mg|km|cm|mm|m)$")
_DE_ORDINAL_RE = re.compile(r"^(\d+)(\.|tes|te|ter|ten|tem)$")
_DE_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+1/2$")
_DE_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_DE_PUNKT_RE = re.compile(r"^(\d+)\.(\d+)$")
_DE_MONEY_RE = re.compile(r"^([€£$])\s?(\d+)(?:,(\d+))?$")
_DE_MONEY_SUFFIX_RE = re.compile(r"^(\d+)(?:,(\d+))?\s?([€£$])$")
_DE_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_DE_MONTHS = (
    "",
    "januar",
    "februar",
    "märz",
    "april",
    "mai",
    "juni",
    "juli",
    "august",
    "september",
    "oktober",
    "november",
    "dezember",
)
_DE_COMPOUND_RE = re.compile(r"^([A-Za-z]+)-(\d+)$")
_DE_WHITELIST = {
    "z.b.": "zum beispiel",
    "d.h.": "dass heißt",
    "dr.": "doktor",
    "mr.": "mister",
    "mrs.": "misses",
    "ms.": "miss",
    "nr.": "nummer",
}
_DE_MONEY_WORDS = {
    "€": ("euro", "cent"),
    "£": ("pfund", "pence"),
    "$": ("dollar", "cent"),
}


def _de_under_100(value: int, one: str = "eins") -> str:
    if value < 20:
        if value == 1:
            return one
        return _DE_SMALL[value]
    tens, units = divmod(value, 10)
    if units == 0:
        return _DE_TENS[tens]
    unit = "ein" if units == 1 else _DE_SMALL[units]
    return f"{unit} und {_DE_TENS[tens]}"


def _de_under_1000(value: int) -> str:
    if value < 100:
        return _de_under_100(value)
    hundreds, remainder = divmod(value, 100)
    head = "ein hundert" if hundreds == 1 else f"{_DE_SMALL[hundreds]} hundert"
    if remainder == 0:
        return head
    return f"{head} {_de_under_100(remainder)}"


def german_nemo_cardinal(value: int) -> str:
    """Match official deterministic German cardinals below one million."""

    if value < 0:
        raise ValueError("cardinal only accepts non-negative integers")
    if value == 0:
        return "null"
    if value < 1000:
        return _de_under_1000(value)
    thousands, remainder = divmod(value, 1000)
    if thousands == 1:
        head = "ein tausend"
    else:
        head = f"{_de_under_1000(thousands)} tausend"
    if remainder == 0:
        return head
    return f"{head} {_de_under_1000(remainder)}"


def german_nemo_ordinal(value: int) -> Optional[str]:
    if value <= 0 or value > 999:
        return None
    if value < 20:
        return _DE_ORDINAL_SMALL[value]
    if value < 100:
        return german_nemo_cardinal(value) + "ste"
    hundreds, remainder = divmod(value, 100)
    head = "ein hundert" if hundreds == 1 else f"{_DE_SMALL[hundreds]} hundert"
    if remainder == 0:
        return head + "ste"
    return f"{head} {german_nemo_ordinal(remainder)}"


def _de_digit_words(digits: str) -> str:
    return " ".join(_DE_SMALL[int(char)] for char in digits)


def _de_signed(sign: str, body: str) -> str:
    return f"minus {body}" if sign == "-" else body


def _de_decimal_words(integer: str, fraction: str, one: str = "eins") -> str:
    value = int(integer)
    if value == 1:
        integer_words = one
    else:
        integer_words = german_nemo_cardinal(value)
    return f"{integer_words} komma {_de_digit_words(fraction)}"


def _fast_german_integer(span: str) -> Optional[str]:
    if span.startswith("+"):
        return None
    sign = ""
    body = span
    if body.startswith("-"):
        sign = "minus "
        body = body[1:]
    if not body.isdigit():
        return None
    if body.startswith("0") and len(body) > 1:
        zero_count = len(body) - len(body.lstrip("0"))
        rest = body.lstrip("0")
        zeros = " ".join(["null"] * zero_count)
        if not rest:
            return sign + zeros
        return sign + zeros + " " + german_nemo_cardinal(int(rest))
    value = int(body)
    if value > 999_999:
        return None
    return sign + german_nemo_cardinal(value)


def _fast_german_decimal(span: str) -> Optional[str]:
    match = _DE_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    return _de_signed(sign, _de_decimal_words(integer, fraction, one="eins"))


def _fast_german_percent(span: str) -> Optional[str]:
    match = _DE_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _de_decimal_words(integer, fraction, one="eins")
    else:
        value = int(number)
        words = "ein" if value == 1 else german_nemo_cardinal(value)
    return _de_signed(sign, f"{words} prozent")


def _fast_german_ordinal(span: str) -> Optional[str]:
    match = _DE_ORDINAL_RE.fullmatch(span)
    if match is None:
        return None
    return german_nemo_ordinal(int(match.group(1)))


def _fast_german_fraction(span: str) -> Optional[str]:
    if span == "1/2":
        return "ein halb"
    match = _DE_MIXED_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    return f"{german_nemo_cardinal(int(match.group(1)))} ein halb"


def _fast_german_measure(span: str) -> Optional[str]:
    match = _DE_MEASURE_RE.fullmatch(span)
    if match is not None:
        sign, number, unit = match.groups()
        unit_word = _DE_MEASURE_UNITS[unit]
        if "," in number:
            integer, fraction = number.split(",", 1)
            words = _de_decimal_words(integer, fraction, one="eins")
        else:
            value = int(number)
            words = "ein" if value == 1 else german_nemo_cardinal(value)
        return _de_signed(sign, f"{words} {unit_word}")
    match = _DE_DOT_MEASURE_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction, unit = match.groups()
    unit_word = _DE_MEASURE_UNITS[unit]
    if fraction.startswith("0") and len(fraction) >= 2:
        words = f"{_de_digit_words(integer)} punkt {_de_digit_words(fraction)} {unit_word}"
    else:
        ordinal = german_nemo_ordinal(int(integer))
        if ordinal is None:
            return None
        words = f"{ordinal} {german_nemo_cardinal(int(fraction))} {unit_word}"
    return _de_signed(sign, words)


def _fast_german_dotted(span: str) -> Optional[str]:
    match = _DE_PUNKT_RE.fullmatch(span)
    if match is None:
        return None
    integer, fraction = match.groups()
    return f"{_de_digit_words(integer)} punkt {_de_digit_words(fraction)}"


def _de_clock_field(raw: str) -> str:
    if raw.startswith("0") and len(raw) > 1:
        return _de_digit_words(raw)
    return german_nemo_cardinal(int(raw))


def _fast_german_clock(span: str) -> Optional[str]:
    match = _DE_CLOCK_RE.fullmatch(span)
    if match is None:
        return None
    hour, minute = match.groups()
    hour_words = _de_clock_field(hour)
    if int(minute) == 0:
        return f"{hour_words} Uhr"
    return f"{hour_words} Uhr {_de_clock_field(minute)}"


def _de_year(year: int) -> str:
    if year >= 2000:
        return german_nemo_cardinal(year)
    century, yy = divmod(year, 100)
    head = german_nemo_cardinal(century)
    if yy == 0:
        return f"{head} hundert"
    return f"{head} {german_nemo_cardinal(yy)}"


def _fast_german_iso_date(span: str) -> Optional[str]:
    match = _DE_ISO_RE.fullmatch(span)
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    day_word = german_nemo_ordinal(day)
    if day_word is None:
        return None
    if day_word.endswith("e"):
        day_word += "r"
    return f"{day_word} {_DE_MONTHS[month]} {_de_year(year)}"


def _fast_german_money(span: str) -> Optional[str]:
    match = _DE_MONEY_RE.fullmatch(span)
    if match is not None:
        currency, integer, fraction = match.groups()
    else:
        match = _DE_MONEY_SUFFIX_RE.fullmatch(span)
        if match is None:
            return None
        integer, fraction, currency = match.groups()
    major, minor = _DE_MONEY_WORDS[currency]
    value = int(integer)
    integer_words = "ein" if value == 1 else german_nemo_cardinal(value)
    if fraction is None:
        return f"{integer_words} {major}"
    cents = int(fraction) * 10 if len(fraction) == 1 else int(fraction)
    return f"{integer_words} {major} {german_nemo_cardinal(cents)} {minor}"


def _fast_german_compound(span: str) -> Optional[str]:
    match = _DE_COMPOUND_RE.fullmatch(span)
    if match is None:
        return None
    letters, digits = match.groups()
    words = _fast_german_integer(digits)
    if words is None:
        return None
    return f"{letters} {words}"


def _fast_german_abbreviation(span: str) -> Optional[str]:
    return _DE_WHITELIST.get(span.lower())


def fast_german_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe German classes; return None to the small graph when alignment or coverage fails."""

    if category == "date":
        return _fast_german_iso_date(span)
    if category == "integer":
        return _fast_german_integer(span)
    if category == "decimal":
        return _fast_german_decimal(span)
    if category == "percent":
        return _fast_german_percent(span)
    if category == "ordinal":
        return _fast_german_ordinal(span)
    if category == "fraction":
        return _fast_german_fraction(span)
    if category == "measure":
        return _fast_german_measure(span)
    if category == "dotted":
        return _fast_german_dotted(span)
    if category == "time":
        if "uhr" in span.lower():
            return None
        return _fast_german_clock(span)
    if category == "money":
        return _fast_german_money(span)
    if category == "compound":
        return _fast_german_compound(span)
    if category == "abbreviation":
        return _fast_german_abbreviation(span)
    return None


# Spanish official TN: European thousands (1.000 / 1 000), comma decimals; standalone cardinals keep apocope.
_ES_ONES = (
    "cero",
    "un",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
)
_ES_TEENS = (
    "diez",
    "once",
    "doce",
    "trece",
    "catorce",
    "quince",
    "dieciséis",
    "diecisiete",
    "dieciocho",
    "diecinueve",
)
_ES_TWENTIES = (
    "veinte",
    "veintiún",
    "veintidós",
    "veintitrés",
    "veinticuatro",
    "veinticinco",
    "veintiséis",
    "veintisiete",
    "veintiocho",
    "veintinueve",
)
_ES_TENS = {
    3: "treinta",
    4: "cuarenta",
    5: "cincuenta",
    6: "sesenta",
    7: "setenta",
    8: "ochenta",
    9: "noventa",
}
_ES_HUNDREDS = {
    2: "doscientos",
    3: "trescientos",
    4: "cuatrocientos",
    5: "quinientos",
    6: "seiscientos",
    7: "setecientos",
    8: "ochocientos",
    9: "novecientos",
}
_ES_DIGIT_WORDS = (
    "cero",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
)
_ES_MEASURE_UNITS = {
    "kg": ("kilogramo", "kilogramos"),
    "g": ("gramo", "gramos"),
    "km": ("kilómetro", "kilómetros"),
    "cm": ("centímetro", "centímetros"),
    "mm": ("milímetro", "milímetros"),
    "m": ("metro", "metros"),
}
_ES_DECIMAL_RE = re.compile(r"^(-?)(\d+),(\d+)$")
_ES_PERCENT_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?%$")
_ES_MEASURE_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?(kg|km|cm|mm|g|m)$")
_ES_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+1/2$")
_ES_TIME_H_RE = re.compile(r"^([1-9]\d?)\s?h$")


def _es_under_100(value: int) -> str:
    if value < 10:
        return _ES_ONES[value]
    if value < 20:
        return _ES_TEENS[value - 10]
    if value < 30:
        return _ES_TWENTIES[value - 20]
    tens, units = divmod(value, 10)
    if units == 0:
        return _ES_TENS[tens]
    return f"{_ES_TENS[tens]} y {_ES_ONES[units]}"


def _es_under_1000(value: int) -> str:
    if value < 100:
        return _es_under_100(value)
    if value == 100:
        return "cien"
    hundreds, remainder = divmod(value, 100)
    head = "ciento" if hundreds == 1 else _ES_HUNDREDS[hundreds]
    if remainder == 0:
        return head
    return f"{head} {_es_under_100(remainder)}"


def spanish_nemo_cardinal(value: int) -> str:
    """Match official deterministic Spanish cardinals below one billion."""

    if value < 0:
        raise ValueError("cardinal only accepts non-negative integers")
    if value == 0:
        return "cero"
    if value < 1000:
        return _es_under_1000(value)
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        head = "mil" if thousands == 1 else f"{_es_under_1000(thousands)} mil"
        if remainder == 0:
            return head
        return f"{head} {_es_under_1000(remainder)}"
    millions, remainder = divmod(value, 1_000_000)
    if millions == 1:
        head = "un millón"
    else:
        head = f"{_es_under_1000(millions)} millones"
    if remainder == 0:
        return head
    return f"{head} {spanish_nemo_cardinal(remainder)}"


def _es_strip_apocope(words: str) -> str:
    if words == "un":
        return "uno"
    if words.endswith("veintiún"):
        return words[: -len("veintiún")] + "veintiuno"
    if words.endswith(" un"):
        return words[:-3] + " uno"
    return words


def _es_signed(sign: str, body: str) -> str:
    return f"menos {body}" if sign == "-" else body


def _es_digit_words(digits: str) -> str:
    parts = []
    last_index = len(digits) - 1
    for index, char in enumerate(digits):
        if char == "1":
            parts.append("un" if index == last_index else "uno")
        else:
            parts.append(_ES_DIGIT_WORDS[int(char)])
    return " ".join(parts)


def _es_fractional_words(digits: str) -> str:
    if digits.startswith("0"):
        return _es_digit_words(digits)
    return spanish_nemo_cardinal(int(digits))


def _es_decimal_words(integer: str, fraction: str) -> str:
    return (
        f"{_es_strip_apocope(spanish_nemo_cardinal(int(integer)))}"
        f" coma {_es_fractional_words(fraction)}"
    )


def _fast_spanish_integer(span: str) -> Optional[str]:
    if span.startswith("+"):
        return None
    sign = ""
    body = span
    if body.startswith("-"):
        sign = "menos "
        body = body[1:]
    compact = body.replace(" ", "").replace(".", "")
    if not compact.isdigit():
        return None
    if compact.startswith("0") and len(compact) > 1:
        return span
    value = int(compact)
    if value > 999_999_999:
        return None
    return sign + spanish_nemo_cardinal(value)


def _fast_spanish_decimal(span: str) -> Optional[str]:
    match = _ES_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    return _es_signed(sign, _es_decimal_words(integer, fraction))


def _fast_spanish_percent(span: str) -> Optional[str]:
    match = _ES_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _es_decimal_words(integer, fraction)
    else:
        words = _es_strip_apocope(spanish_nemo_cardinal(int(number)))
    return _es_signed(sign, f"{words} por ciento")


def _fast_spanish_fraction(span: str) -> Optional[str]:
    if span == "1/2":
        return "medio"
    match = _ES_MIXED_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    return f"{spanish_nemo_cardinal(int(match.group(1)))} y medio"


def _fast_spanish_measure(span: str) -> Optional[str]:
    match = _ES_MEASURE_RE.fullmatch(span)
    if match is None:
        return None
    sign, number, unit = match.groups()
    singular, plural = _ES_MEASURE_UNITS[unit]
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _es_decimal_words(integer, fraction)
        unit_word = plural
    else:
        value = int(number)
        words = spanish_nemo_cardinal(value)
        unit_word = singular if value == 1 else plural
    return _es_signed(sign, f"{words} {unit_word}")


def _es_feminine_cardinal(value: int) -> str:
    if value == 1:
        return "una"
    words = spanish_nemo_cardinal(value)
    words = words.replace("veintiún", "veintiuna")
    if words.endswith(" un"):
        return words[:-3] + " una"
    return words


def _fast_spanish_time(span: str) -> Optional[str]:
    match = _ES_TIME_H_RE.fullmatch(span)
    if match is None:
        return None
    value = int(match.group(1))
    if value == 1:
        return "una hora"
    return f"{_es_feminine_cardinal(value)} horas"


_ES_MONTHS = (
    "",
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)
_ES_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_ES_ORDINAL_RE = re.compile(r"^(\d+)(?:\.?º|°|ª|er|ra|do|to|mo|o)$")
_ES_ORDINAL_MASC = {
    1: "primero",
    2: "segundo",
    3: "tercero",
    4: "cuarto",
    5: "quinto",
    6: "sexto",
    7: "séptimo",
    8: "octavo",
    9: "noveno",
    10: "décimo",
    11: "undécimo",
    12: "duodécimo",
}
_ES_MONEY_MAJOR = {
    "$": ("dólar", "dólares"),
    "€": ("euro", "euros"),
    "£": ("libra", "libras"),
    "¥": ("yen", "yenes"),
}
_ES_MONEY_MINOR = {
    "$": "centavos",
    "€": "céntimos",
    "£": "peniques",
    "¥": None,
}
_ES_MONEY_PREFIX_RE = re.compile(r"^([€£$¥])\s?(\d+)(?:[.,](\d+))?$")
_ES_MONEY_SUFFIX_RE = re.compile(r"^(\d+)(?:[.,](\d+))?\s?([€£$¥])$")


def _es_major_article(value: int, singular: str) -> str:
    if value != 1:
        return spanish_nemo_cardinal(value)
    if singular.endswith("a"):
        return "una"
    return "un"


def _fast_spanish_iso_date(span: str) -> Optional[str]:
    match = _ES_ISO_RE.fullmatch(span)
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return (
        f"{_es_strip_apocope(spanish_nemo_cardinal(day))} de "
        f"{_ES_MONTHS[month]} de {spanish_nemo_cardinal(year)}"
    )


def _fast_spanish_ordinal(span: str) -> Optional[str]:
    match = _ES_ORDINAL_RE.fullmatch(span)
    if match is None:
        return None
    return _ES_ORDINAL_MASC.get(int(match.group(1)))


def _fast_spanish_money(span: str) -> Optional[str]:
    match = _ES_MONEY_PREFIX_RE.fullmatch(span)
    if match is not None:
        currency, integer, fraction = match.groups()
    else:
        match = _ES_MONEY_SUFFIX_RE.fullmatch(span)
        if match is None:
            return None
        integer, fraction, currency = match.groups()
    value = int(integer)
    singular, plural = _ES_MONEY_MAJOR[currency]
    head = _es_major_article(value, singular)
    major = singular if value == 1 else plural
    if fraction is None:
        return f"{head} {major}"
    cents = int(fraction) * 10 if len(fraction) == 1 else int(fraction)
    minor = _ES_MONEY_MINOR[currency]
    if minor is None:
        return None
    minor_word = "céntimo" if currency == "€" and cents == 1 else minor
    if currency == "$" and cents == 1:
        minor_word = "centavo"
    return f"{head} {major} {spanish_nemo_cardinal(cents)} {minor_word}"


def fast_spanish_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Spanish classes; return None to the small graph when alignment or coverage fails."""

    if category == "date":
        return _fast_spanish_iso_date(span)
    if category == "integer":
        return _fast_spanish_integer(span)
    if category == "decimal":
        return _fast_spanish_decimal(span)
    if category == "percent":
        return _fast_spanish_percent(span)
    if category == "fraction":
        return _fast_spanish_fraction(span)
    if category == "measure":
        return _fast_spanish_measure(span)
    if category == "time":
        return _fast_spanish_time(span)
    if category == "ordinal":
        return _fast_spanish_ordinal(span)
    if category == "money":
        return _fast_spanish_money(span)
    return None


# French official TN: space thousands, comma decimals; '.' clashes with dates and is not a thousands sep.
_FR_ONES = (
    "zéro",
    "un",
    "deux",
    "trois",
    "quatre",
    "cinq",
    "six",
    "sept",
    "huit",
    "neuf",
)
_FR_TEENS = (
    "dix",
    "onze",
    "douze",
    "treize",
    "quatorze",
    "quinze",
    "seize",
    "dix-sept",
    "dix-huit",
    "dix-neuf",
)
_FR_TENS = {
    2: "vingt",
    3: "trente",
    4: "quarante",
    5: "cinquante",
    6: "soixante",
}
_FR_DECIMAL_RE = re.compile(r"^(-?)(\d+),(\d+)$")
_FR_PERCENT_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?%$")
_FR_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+1/2$")


def _fr_under_100(value: int) -> str:
    if value < 10:
        return _FR_ONES[value]
    if value < 20:
        return _FR_TEENS[value - 10]
    if value < 70:
        tens, units = divmod(value, 10)
        if units == 0:
            return _FR_TENS[tens]
        if units == 1:
            return f"{_FR_TENS[tens]} et un"
        return f"{_FR_TENS[tens]}-{_FR_ONES[units]}"
    if value < 80:
        if value == 71:
            return "soixante et onze"
        return "soixante-" + _FR_TEENS[value - 70]
    if value < 90:
        if value == 80:
            return "quatre-vingts"
        return "quatre-vingt-" + _FR_ONES[value - 80]
    return "quatre-vingt-" + _FR_TEENS[value - 90]


def _fr_under_1000(value: int) -> str:
    if value < 100:
        return _fr_under_100(value)
    hundreds, remainder = divmod(value, 100)
    if hundreds == 1:
        if remainder == 0:
            return "cent"
        return "cent " + _fr_under_100(remainder)
    head = f"{_FR_ONES[hundreds]} cent"
    if remainder == 0:
        return head + "s"
    return head + " " + _fr_under_100(remainder)


def french_nemo_cardinal(value: int) -> str:
    """Match official deterministic French cardinals below one billion."""

    if value < 0:
        raise ValueError("cardinal only accepts non-negative integers")
    if value == 0:
        return "zéro"
    if value < 1000:
        return _fr_under_1000(value)
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        head = "mille" if thousands == 1 else f"{_fr_under_1000(thousands)} mille"
        if remainder == 0:
            return head
        if remainder == 1:
            return head + " et un"
        return f"{head} {_fr_under_1000(remainder)}"
    millions, remainder = divmod(value, 1_000_000)
    head = "un million" if millions == 1 else f"{_fr_under_1000(millions)} millions"
    if remainder == 0:
        return head
    if remainder == 1:
        return head + " et un"
    return f"{head} {french_nemo_cardinal(remainder)}"


def _fr_signed(sign: str, body: str) -> str:
    return f"moins {body}" if sign == "-" else body


def _fr_digit_words(digits: str) -> str:
    return " ".join(_FR_ONES[int(char)] for char in digits)


def _fr_fractional_words(digits: str) -> str:
    if digits.startswith("0"):
        return _fr_digit_words(digits)
    return french_nemo_cardinal(int(digits))


def _fr_decimal_words(integer: str, fraction: str) -> str:
    return f"{french_nemo_cardinal(int(integer))} virgule {_fr_fractional_words(fraction)}"


def _fast_french_integer(span: str) -> Optional[str]:
    if span.startswith("+"):
        return None
    sign = ""
    body = span
    if body.startswith("-"):
        sign = "moins "
        body = body[1:]
    compact = body.replace(" ", "")
    if "." in compact or not compact.isdigit():
        return None
    if compact.startswith("0") and len(compact) > 1:
        zero_count = len(compact) - len(compact.lstrip("0"))
        rest = compact.lstrip("0")
        zeros = " ".join(["zéro"] * zero_count)
        if not rest:
            return sign + zeros
        return sign + zeros + " " + french_nemo_cardinal(int(rest))
    value = int(compact)
    if value > 999_999_999:
        return None
    return sign + french_nemo_cardinal(value)


def _fast_french_decimal(span: str) -> Optional[str]:
    match = _FR_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    return _fr_signed(sign, _fr_decimal_words(integer, fraction))


def _fast_french_percent(span: str) -> Optional[str]:
    match = _FR_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _fr_decimal_words(integer, fraction)
    else:
        words = french_nemo_cardinal(int(number))
    return _fr_signed(sign, f"{words} pour cent")


def _fast_french_fraction(span: str) -> Optional[str]:
    if span == "1/2":
        return "un demi"
    if span == "1/4":
        return "un quart"
    if span == "2/3":
        return "deux tiers"
    match = _FR_MIXED_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    return f"{french_nemo_cardinal(int(match.group(1)))} et demi"


_FR_MONTHS = (
    "",
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)
_FR_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_FR_PUNKT_RE = re.compile(r"^(\d+)\.(\d+)$")
_FR_DAY_MONTH_RE = re.compile(
    r"^(0?[1-9]|[12]\d|3[01])\.(0[1-9]|1[0-2]|[1-9])(\d*)$"
)
_FR_MONTHS_TRAILING_SPACE = frozenset({2, 5, 8, 11})


def _fast_french_iso_date(span: str) -> Optional[str]:
    # Official ISO graph is broken (hyphen -> moins). Match the working slash date reading.
    match = _FR_ISO_RE.fullmatch(span)
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return (
        f"{french_nemo_cardinal(day)} {_FR_MONTHS[month]} "
        f"{french_nemo_cardinal(year)}"
    )


def _fast_french_day_month(span: str) -> Optional[str]:
    match = _FR_DAY_MONTH_RE.fullmatch(span)
    if match is None:
        return None
    day = int(match.group(1))
    month = int(match.group(2))
    leftover = match.group(3)
    words = f"{french_nemo_cardinal(day)} {_FR_MONTHS[month]}"
    if month in _FR_MONTHS_TRAILING_SPACE:
        words += " "
    if not leftover:
        return words
    if leftover.startswith("0"):
        return words + _fr_digit_words(leftover)
    return words.rstrip() + " " + french_nemo_cardinal(int(leftover))


def _fast_french_dotted(span: str) -> Optional[str]:
    match = _FR_PUNKT_RE.fullmatch(span)
    if match is None:
        return None
    integer, fraction = match.groups()
    return f"{french_nemo_cardinal(int(integer))} . {_fr_digit_words(fraction)}"


_FR_ORDINAL_SMALL = {
    1: "premier",
    2: "deuxième",
    3: "troisième",
    4: "quatrième",
    5: "cinquième",
    6: "sixième",
    7: "septième",
    8: "huitième",
    9: "neuvième",
    10: "dixième",
    11: "onzième",
    12: "douzième",
    13: "treizième",
    14: "quatorzième",
    15: "quinzième",
    16: "seizième",
    17: "dix-septième",
    18: "dix-huitième",
    19: "dix-neuvième",
    20: "vingtième",
}
_FR_DEGREE_ORDINAL_RE = re.compile(r"^(\d+)°$")


def _fast_french_ordinal(span: str) -> Optional[str]:
    match = _FR_DEGREE_ORDINAL_RE.fullmatch(span)
    if match is None:
        return None
    return _FR_ORDINAL_SMALL.get(int(match.group(1)))


def fast_french_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe French classes; return None to the small graph when alignment or coverage fails."""

    if category == "date":
        iso = _fast_french_iso_date(span)
        if iso is not None:
            return iso
        return _fast_french_day_month(span)
    if category == "dotted":
        return _fast_french_dotted(span)
    if category == "integer":
        return _fast_french_integer(span)
    if category == "decimal":
        return _fast_french_decimal(span)
    if category == "percent":
        return _fast_french_percent(span)
    if category == "fraction":
        return _fast_french_fraction(span)
    if category == "ordinal":
        return _fast_french_ordinal(span)
    return None


# Italian official TN: space thousands, comma decimals; '.' reads punto, not thousands.
_IT_ONES = (
    "zero",
    "uno",
    "due",
    "tre",
    "quattro",
    "cinque",
    "sei",
    "sette",
    "otto",
    "nove",
)
_IT_TEENS = (
    "dieci",
    "undici",
    "dodici",
    "tredici",
    "quattordici",
    "quindici",
    "sedici",
    "diciassette",
    "diciotto",
    "diciannove",
)
_IT_TENS = {
    2: "venti",
    3: "trenta",
    4: "quaranta",
    5: "cinquanta",
    6: "sessanta",
    7: "settanta",
    8: "ottanta",
    9: "novanta",
}
_IT_HUNDREDS = {
    2: "duecento",
    3: "trecento",
    4: "quattrocento",
    5: "cinquecento",
    6: "seicento",
    7: "settecento",
    8: "ottocento",
    9: "novecento",
}
_IT_MEASURE_UNITS = {
    "kg": ("chilogrammo", "chilogrammi"),
    "km": ("chilometro", "chilometri"),
    "m": ("metro", "metri"),
}
_IT_DECIMAL_RE = re.compile(r"^(-?)(\d+),(\d+)$")
_IT_PERCENT_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?%$")
_IT_MEASURE_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?(kg|km|m)$")
_IT_TIME_H_RE = re.compile(r"^([1-9]\d?)\s?h$")
_IT_CLOCK_RE = re.compile(r"^(\d{2}):(\d{2})$")
_IT_PUNKT_RE = re.compile(r"^(\d+)\.(\d+)$")


def _it_under_100(value: int) -> str:
    if value < 10:
        return _IT_ONES[value]
    if value < 20:
        return _IT_TEENS[value - 10]
    tens, units = divmod(value, 10)
    stem = _IT_TENS[tens]
    if units == 0:
        return stem
    unit_word = _IT_ONES[units]
    # Vowel-initial ones eat the tens vowel: ventuno / ventotto, but ventidue keeps i.
    if unit_word[0] in "aeiou":
        return stem[:-1] + unit_word
    return stem + unit_word


def _it_under_1000(value: int) -> str:
    if value < 100:
        return _it_under_100(value)
    hundreds, remainder = divmod(value, 100)
    head = "cento" if hundreds == 1 else _IT_HUNDREDS[hundreds]
    if remainder == 0:
        return head
    return f"{head} {_it_under_100(remainder)}"


def italian_nemo_cardinal(value: int) -> str:
    """Match official deterministic Italian cardinals below one hundred thousand."""

    if value < 0:
        raise ValueError("cardinal only accepts non-negative integers")
    if value == 0:
        return "zero"
    if value < 1000:
        return _it_under_1000(value)
    thousands, remainder = divmod(value, 1000)
    if thousands == 1:
        head = "mille"
    else:
        head = _it_under_1000(thousands).replace(" ", "") + "mila"
    if remainder == 0:
        return head
    return f"{head} {_it_under_1000(remainder)}"


def _it_signed(sign: str, body: str) -> str:
    if sign == "-":
        return f"meno {body}"
    if sign == "+":
        return f"più {body}"
    return body


def _it_apocope_un(words: str) -> str:
    if words == "uno":
        return "un"
    return words


def _it_digit_words(digits: str) -> str:
    return " ".join(_IT_ONES[int(char)] for char in digits)


def _it_fractional_words(digits: str) -> str:
    if digits.startswith("0"):
        return _it_digit_words(digits)
    return italian_nemo_cardinal(int(digits))


def _it_decimal_words(integer: str, fraction: str) -> str:
    return f"{italian_nemo_cardinal(int(integer))} virgola {_it_fractional_words(fraction)}"


def _fast_italian_integer(span: str) -> Optional[str]:
    sign = ""
    body = span
    if body.startswith("+"):
        sign = "+"
        body = body[1:]
    elif body.startswith("-"):
        sign = "-"
        body = body[1:]
    compact = body.replace(" ", "")
    if "." in compact or not compact.isdigit():
        return None
    if compact.startswith("0") and len(compact) > 1:
        zero_count = len(compact) - len(compact.lstrip("0"))
        rest = compact.lstrip("0")
        zeros = " ".join(["zero"] * zero_count)
        if not rest:
            return _it_signed(sign, zeros)
        return _it_signed(sign, zeros + " " + italian_nemo_cardinal(int(rest)))
    value = int(compact)
    if value > 99_999:
        return None
    return _it_signed(sign, italian_nemo_cardinal(value))


def _fast_italian_decimal(span: str) -> Optional[str]:
    match = _IT_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    return _it_signed(sign, _it_decimal_words(integer, fraction))


def _fast_italian_percent(span: str) -> Optional[str]:
    match = _IT_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _it_decimal_words(integer, fraction)
    else:
        value = int(number)
        words = _it_apocope_un(italian_nemo_cardinal(value))
    return _it_signed(sign, f"{words} percento")


def _fast_italian_measure(span: str) -> Optional[str]:
    match = _IT_MEASURE_RE.fullmatch(span)
    if match is None:
        return None
    sign, number, unit = match.groups()
    singular, plural = _IT_MEASURE_UNITS[unit]
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _it_decimal_words(integer, fraction)
        unit_word = plural
    else:
        value = int(number)
        words = _it_apocope_un(italian_nemo_cardinal(value))
        unit_word = singular if value == 1 else plural
    return _it_signed(sign, f"{words} {unit_word}")


def _fast_italian_time(span: str) -> Optional[str]:
    match = _IT_TIME_H_RE.fullmatch(span)
    if match is not None:
        value = int(match.group(1))
        if value == 1:
            return "un ora"
        return f"{italian_nemo_cardinal(value)} ore"
    match = _IT_CLOCK_RE.fullmatch(span)
    if match is None:
        return None
    hour = italian_nemo_cardinal(int(match.group(1)))
    minute = int(match.group(2))
    if minute == 0:
        return hour
    if minute == 15:
        return f"{hour} e un quarto"
    if minute == 30:
        return f"{hour} e mezza"
    if minute == 45:
        return f"{hour} e {italian_nemo_cardinal(45)} minuti"
    return None


def _fast_italian_dotted(span: str) -> Optional[str]:
    match = _IT_PUNKT_RE.fullmatch(span)
    if match is None:
        return None
    integer, fraction = match.groups()
    return f"{_it_digit_words(integer)} punto {_it_digit_words(fraction)}"


_IT_MONEY_MAJOR = {
    "$": ("dollaro", "dollari"),
    "€": ("euro", "euro"),
    "£": ("sterlina", "sterline"),
    "¥": ("yen", "yen"),
}
_IT_MONEY_MINOR = {
    "$": "centesimi",
    "€": "centesimi",
    "£": "pence",
    "¥": None,
}
_IT_MONEY_PREFIX_RE = re.compile(r"^([€£$¥])\s?(\d+)(?:[.,](\d+))?$")
_IT_MONEY_SUFFIX_RE = re.compile(r"^(\d+)(?:[.,](\d+))?\s?([€£$¥])$")


def _fast_italian_money(span: str) -> Optional[str]:
    match = _IT_MONEY_PREFIX_RE.fullmatch(span)
    if match is not None:
        currency, integer, fraction = match.groups()
    else:
        match = _IT_MONEY_SUFFIX_RE.fullmatch(span)
        if match is None:
            return None
        integer, fraction, currency = match.groups()
    value = int(integer)
    singular, plural = _IT_MONEY_MAJOR[currency]
    if value == 1:
        head = "una" if singular.endswith("a") else "un"
        major = singular
    else:
        head = italian_nemo_cardinal(value)
        major = plural
    if fraction is None:
        return f"{head} {major}"
    cents = int(fraction) * 10 if len(fraction) == 1 else int(fraction)
    minor = _IT_MONEY_MINOR[currency]
    if minor is None:
        return None
    return f"{head} {major} {italian_nemo_cardinal(cents)} {minor}"


def fast_italian_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Italian classes; return None to the small graph when alignment or coverage fails."""

    if category == "integer":
        return _fast_italian_integer(span)
    if category == "decimal":
        return _fast_italian_decimal(span)
    if category == "percent":
        return _fast_italian_percent(span)
    if category == "measure":
        return _fast_italian_measure(span)
    if category == "time":
        return _fast_italian_time(span)
    if category == "dotted":
        return _fast_italian_dotted(span)
    if category == "money":
        return _fast_italian_money(span)
    return None


# Portuguese official TN: European thousands (1.000 / 1 000), comma decimals; hours are feminine.
_PT_ONES = (
    "zero",
    "um",
    "dois",
    "três",
    "quatro",
    "cinco",
    "seis",
    "sete",
    "oito",
    "nove",
)
_PT_TEENS = (
    "dez",
    "onze",
    "doze",
    "treze",
    "catorze",
    "quinze",
    "dezesseis",
    "dezessete",
    "dezoito",
    "dezenove",
)
_PT_TENS = {
    2: "vinte",
    3: "trinta",
    4: "quarenta",
    5: "cinquenta",
    6: "sessenta",
    7: "setenta",
    8: "oitenta",
    9: "noventa",
}
_PT_HUNDREDS = {
    2: "duzentos",
    3: "trezentos",
    4: "quatrocentos",
    5: "quinhentos",
    6: "seiscentos",
    7: "setecentos",
    8: "oitocentos",
    9: "novecentos",
}
_PT_MEASURE_UNITS = {
    "kg": ("quilo", "quilos"),
    "m": ("metro", "metros"),
}
_PT_DECIMAL_RE = re.compile(r"^(-?)(\d+),(\d+)$")
_PT_PERCENT_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?%$")
_PT_MEASURE_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?(kg|m)$")
_PT_TIME_H_RE = re.compile(r"^([1-9]\d?)\s?h$")
_PT_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_PT_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+1/2$")


def _pt_under_100(value: int) -> str:
    if value < 10:
        return _PT_ONES[value]
    if value < 20:
        return _PT_TEENS[value - 10]
    tens, units = divmod(value, 10)
    if units == 0:
        return _PT_TENS[tens]
    return f"{_PT_TENS[tens]} e {_PT_ONES[units]}"


def _pt_under_1000(value: int) -> str:
    if value < 100:
        return _pt_under_100(value)
    if value == 100:
        return "cem"
    hundreds, remainder = divmod(value, 100)
    head = "cento" if hundreds == 1 else _PT_HUNDREDS[hundreds]
    if remainder == 0:
        return head
    return f"{head} e {_pt_under_100(remainder)}"


def portuguese_nemo_cardinal(value: int) -> str:
    """Match official deterministic Portuguese cardinals below one billion."""

    if value < 0:
        raise ValueError("cardinal only accepts non-negative integers")
    if value == 0:
        return "zero"
    if value < 1000:
        return _pt_under_1000(value)
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        head = "mil" if thousands == 1 else f"{_pt_under_1000(thousands)} mil"
        if remainder == 0:
            return head
        if remainder < 100 or remainder % 100 == 0:
            return f"{head} e {_pt_under_1000(remainder)}"
        return f"{head} {_pt_under_1000(remainder)}"
    millions, remainder = divmod(value, 1_000_000)
    head = "um milhão" if millions == 1 else f"{_pt_under_1000(millions)} milhões"
    if remainder == 0:
        return head
    return f"{head} {portuguese_nemo_cardinal(remainder)}"


def _pt_feminine_cardinal(value: int) -> str:
    words = portuguese_nemo_cardinal(value)
    if words == "um":
        return "uma"
    if words == "dois":
        return "duas"
    if words.endswith(" e um"):
        return words[:-2] + "uma"
    if words.endswith(" e dois"):
        return words[:-4] + "duas"
    return words


def _pt_signed(sign: str, body: str) -> str:
    return f"menos {body}" if sign == "-" else body


def _pt_digit_words(digits: str) -> str:
    return " ".join(_PT_ONES[int(char)] for char in digits)


def _pt_decimal_words(integer: str, fraction: str) -> str:
    return f"{portuguese_nemo_cardinal(int(integer))} vírgula {_pt_digit_words(fraction)}"


def _fast_portuguese_integer(span: str) -> Optional[str]:
    if span.startswith("+"):
        return None
    sign = ""
    body = span
    if body.startswith("-"):
        sign = "menos "
        body = body[1:]
    compact = body.replace(" ", "").replace(".", "")
    if not compact.isdigit():
        return None
    if compact.startswith("0") and len(compact) > 1:
        return span
    value = int(compact)
    if value > 999_999_999:
        return None
    return sign + portuguese_nemo_cardinal(value)


def _fast_portuguese_decimal(span: str) -> Optional[str]:
    match = _PT_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    return _pt_signed(sign, _pt_decimal_words(integer, fraction))


def _fast_portuguese_percent(span: str) -> Optional[str]:
    match = _PT_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _pt_decimal_words(integer, fraction)
    else:
        words = portuguese_nemo_cardinal(int(number))
    return _pt_signed(sign, f"{words} por cento")


def _fast_portuguese_fraction(span: str) -> Optional[str]:
    if span == "1/2":
        return "um meio"
    match = _PT_MIXED_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    return f"{portuguese_nemo_cardinal(int(match.group(1)))} e meio"


def _fast_portuguese_measure(span: str) -> Optional[str]:
    match = _PT_MEASURE_RE.fullmatch(span)
    if match is None:
        return None
    sign, number, unit = match.groups()
    singular, plural = _PT_MEASURE_UNITS[unit]
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _pt_decimal_words(integer, fraction)
        unit_word = plural
    else:
        value = int(number)
        words = portuguese_nemo_cardinal(value)
        unit_word = singular if value == 1 else plural
    return _pt_signed(sign, f"{words} {unit_word}")


def _fast_portuguese_time(span: str) -> Optional[str]:
    match = _PT_TIME_H_RE.fullmatch(span)
    if match is not None:
        value = int(match.group(1))
        hour = _pt_feminine_cardinal(value)
        unit = "hora" if value == 1 else "horas"
        return f"{hour} {unit}"
    match = _PT_CLOCK_RE.fullmatch(span)
    if match is None:
        return None
    hour_value = int(match.group(1))
    minute = int(match.group(2))
    hour = _pt_feminine_cardinal(hour_value)
    if minute == 0:
        return f"{hour} horas" if hour_value != 1 else "uma hora"
    return f"{hour} horas e {portuguese_nemo_cardinal(minute)}"


_PT_PUNKT_RE = re.compile(r"^(\d+)\.(\d{1,2})$")


def _fast_portuguese_dotted(span: str) -> Optional[str]:
    match = _PT_PUNKT_RE.fullmatch(span)
    if match is None:
        return None
    integer, fraction = match.groups()
    return f"{_pt_digit_words(integer)} ponto {_pt_digit_words(fraction)}"


_PT_ORDINAL_RE = re.compile(r"^(\d+)[ºª°]$")
_PT_ORDINAL_MASC = {
    1: "primeiro",
    2: "segundo",
    3: "terceiro",
    4: "quarto",
    5: "quinto",
    6: "sexto",
    7: "sétimo",
    8: "oitavo",
    9: "nono",
    10: "décimo",
    11: "décimo primeiro",
    12: "décimo segundo",
}
_PT_MONEY_MAJOR = {
    "$": ("dólar", "dólares"),
    "€": ("euro", "euros"),
    "£": ("libra esterlina", "libras esterlinas"),
    "¥": ("iene", "ienes"),
}
_PT_MONEY_MINOR = {
    "$": "centavos",
    "€": "centavos",
    "£": "pence",
    "¥": None,
}
_PT_MONEY_PREFIX_RE = re.compile(r"^([€£$¥])\s?(\d+)(?:[.,](\d+))?$")
_PT_MONEY_SUFFIX_RE = re.compile(r"^(\d+)(?:[.,](\d+))?\s?([€£$¥])$")


def _fast_portuguese_ordinal(span: str) -> Optional[str]:
    match = _PT_ORDINAL_RE.fullmatch(span)
    if match is None:
        return None
    return _PT_ORDINAL_MASC.get(int(match.group(1)))


def _fast_portuguese_money(span: str) -> Optional[str]:
    match = _PT_MONEY_PREFIX_RE.fullmatch(span)
    if match is not None:
        currency, integer, fraction = match.groups()
    else:
        match = _PT_MONEY_SUFFIX_RE.fullmatch(span)
        if match is None:
            return None
        integer, fraction, currency = match.groups()
    value = int(integer)
    singular, plural = _PT_MONEY_MAJOR[currency]
    head = "um" if value == 1 else portuguese_nemo_cardinal(value)
    major = singular if value == 1 else plural
    if fraction is None:
        return f"{head} {major}"
    cents = int(fraction) * 10 if len(fraction) == 1 else int(fraction)
    minor = _PT_MONEY_MINOR[currency]
    if minor is None:
        return None
    return f"{head} {major} e {portuguese_nemo_cardinal(cents)} {minor}"


def fast_portuguese_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Portuguese classes; return None to the small graph when alignment or coverage fails."""

    if category == "integer":
        return _fast_portuguese_integer(span)
    if category == "decimal":
        return _fast_portuguese_decimal(span)
    if category == "dotted":
        return _fast_portuguese_dotted(span)
    if category == "percent":
        return _fast_portuguese_percent(span)
    if category == "fraction":
        return _fast_portuguese_fraction(span)
    if category == "measure":
        return _fast_portuguese_measure(span)
    if category == "time":
        return _fast_portuguese_time(span)
    if category == "ordinal":
        return _fast_portuguese_ordinal(span)
    if category == "money":
        return _fast_portuguese_money(span)
    return None


# Swedish official TN: space thousands, comma decimals; '.' is not thousands. 1=ett, 100=hundra, 1000=tusen.
_SV_ONES = (
    "noll",
    "ett",
    "två",
    "tre",
    "fyra",
    "fem",
    "sex",
    "sju",
    "åtta",
    "nio",
)
_SV_TEENS = (
    "tio",
    "elva",
    "tolv",
    "tretton",
    "fjorton",
    "femton",
    "sexton",
    "sjutton",
    "arton",
    "nitton",
)
_SV_TENS = {
    2: "tjugo",
    3: "trettio",
    4: "fyrtio",
    5: "femtio",
    6: "sextio",
    7: "sjuttio",
    8: "åttio",
    9: "nittio",
}
_SV_ORDINAL_SMALL = {
    1: "första",
    2: "andra",
    3: "tredje",
    4: "fjärde",
    5: "femte",
    6: "sjätte",
    7: "sjunde",
    8: "åttonde",
    9: "nionde",
    10: "tionde",
    11: "elfte",
    12: "tolfte",
}
_SV_MEASURE_UNITS = {
    "kg": ("ett", "kilogram", "kilogram"),
    "g": ("ett", "gram", "gram"),
    "mg": ("ett", "milligram", "milligram"),
    "lb": ("ett", "pund", "pund"),
    "km": ("en", "kilometer", "kilometer"),
    "cm": ("en", "centimeter", "centimeter"),
    "mm": ("en", "millimeter", "millimeter"),
    "m": ("en", "meter", "meter"),
    "h": ("en", "timme", "timmar"),
}
_SV_DECIMAL_RE = re.compile(r"^(-?)(\d+(?: \d{3})*),(\d+)$")
_SV_PERCENT_RE = re.compile(r"^(-?)(\d+(?: \d{3})*(?:,\d+)?)\s?%$")
_SV_MEASURE_RE = re.compile(
    r"^(-?)(\d+(?:,\d+)?)\s?(kg|km|cm|mm|mg|lb|g|m|h)$"
)
_SV_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+1/2$")
_SV_ORDINAL_DOT_RE = re.compile(r"^(\d+)\.$")
_SV_ORDINAL_COLON_RE = re.compile(r"^(\d+):([ae])$")
_SV_TIME_RE = re.compile(
    r"(?i)^(?:kl\.?|klockan)\s+(\d{1,2})(?:[:.](\d{2}))?$"
)
_SV_MONEY_RE = re.compile(
    r"^(?:([€$£])\s?(\d+(?:,\d+)?)|(\d+(?:,\d+)?)\s?kr|(\d+)\s?([€$£]))$"
)
_SV_COLON_ORDINALS = {
    ("1", "a"): "första",
    ("2", "a"): "andra",
    ("3", "e"): "tredje",
    ("4", "e"): "fjärde",
    ("5", "e"): "femte",
    ("6", "e"): "sjätte",
    ("7", "e"): "sjunde",
    ("8", "e"): "åttonde",
    ("9", "e"): "nionde",
    ("21", "a"): "tjugoförsta",
    ("22", "a"): "tjugoandra",
    ("30", "e"): "trettionde",
    ("32", "a"): "trettioandra",
    ("40", "e"): "fyrtionde",
    ("50", "e"): "femtionde",
    ("60", "e"): "sextionde",
    ("70", "e"): "sjuttionde",
    ("80", "e"): "åttionde",
    ("90", "e"): "nittionde",
    ("100", "e"): "hundrade",
}


def _sv_under_100(value: int) -> str:
    if value < 10:
        return _SV_ONES[value]
    if value < 20:
        return _SV_TEENS[value - 10]
    tens, units = divmod(value, 10)
    if units == 0:
        return _SV_TENS[tens]
    return _SV_TENS[tens] + _SV_ONES[units]


def _sv_under_1000(value: int) -> str:
    if value < 100:
        return _sv_under_100(value)
    hundreds, remainder = divmod(value, 100)
    head = "hundra" if hundreds == 1 else _SV_ONES[hundreds] + "hundra"
    if remainder == 0:
        return head
    return head + _sv_under_100(remainder)


def swedish_nemo_cardinal(value: int) -> str:
    """Match official deterministic Swedish cardinals below one billion."""

    if value < 0:
        raise ValueError("cardinal only accepts non-negative integers")
    if value == 0:
        return "noll"
    if value < 1000:
        return _sv_under_1000(value)
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        head = "tusen" if thousands == 1 else _sv_under_1000(thousands) + "tusen"
        if remainder == 0:
            return head
        return f"{head} {_sv_under_1000(remainder)}"
    millions, remainder = divmod(value, 1_000_000)
    head = "miljon" if millions == 1 else f"{_sv_under_1000(millions)} miljoner"
    if remainder == 0:
        return head
    return f"{head} {swedish_nemo_cardinal(remainder)}"


def swedish_nemo_ordinal_dot(value: int) -> Optional[str]:
    """Match official `N.` ordinals; return None when uncovered."""

    if value in _SV_ORDINAL_SMALL:
        return _SV_ORDINAL_SMALL[value]
    if 13 <= value <= 19:
        return _SV_TEENS[value - 10] + "de"
    if value == 20:
        return "tjugonde"
    if value < 100:
        tens, units = divmod(value, 10)
        if units == 0:
            return _SV_TENS[tens][:-1] + "onde"
        if units in _SV_ORDINAL_SMALL:
            return _SV_TENS[tens] + _SV_ORDINAL_SMALL[units]
        return None
    if value == 100:
        return "hundrade"
    return None


def _sv_signed(sign: str, body: str) -> str:
    return f"minus {body}" if sign == "-" else body


def _sv_en_words(words: str) -> str:
    # Percent uses common gender: ett → en, tjugoett → tjugoen.
    if words == "ett":
        return "en"
    if words.endswith(" ett"):
        return words[:-4] + " en"
    if words.endswith("ett"):
        return words[:-3] + "en"
    return words


def _sv_fractional_words(digits: str) -> str:
    if digits.startswith("0") and len(digits) > 1:
        return " ".join(_SV_ONES[int(char)] for char in digits)
    return swedish_nemo_cardinal(int(digits))


def _sv_decimal_words(integer: str, fraction: str) -> str:
    compact = integer.replace(" ", "")
    return f"{swedish_nemo_cardinal(int(compact))} komma {_sv_fractional_words(fraction)}"


def _fast_swedish_integer(span: str) -> Optional[str]:
    if span.startswith("+"):
        return None
    sign = ""
    body = span
    if body.startswith("-"):
        sign = "minus "
        body = body[1:]
    if "." in body or "," in body:
        return None
    compact = body.replace(" ", "")
    if not compact.isdigit():
        return None
    if compact.startswith("0") and len(compact) > 1:
        return span
    value = int(compact)
    if value > 999_999_999:
        return None
    return sign + swedish_nemo_cardinal(value)


def _fast_swedish_decimal(span: str) -> Optional[str]:
    match = _SV_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    return _sv_signed(sign, _sv_decimal_words(integer, fraction))


def _fast_swedish_percent(span: str) -> Optional[str]:
    match = _SV_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    if "," in number:
        integer, fraction = number.split(",", 1)
        compact = integer.replace(" ", "")
        words = (
            f"{_sv_en_words(swedish_nemo_cardinal(int(compact)))}"
            f" komma {_sv_fractional_words(fraction)}"
        )
    else:
        words = _sv_en_words(swedish_nemo_cardinal(int(number.replace(" ", ""))))
    return _sv_signed(sign, f"{words} procent")


def _fast_swedish_fraction(span: str) -> Optional[str]:
    match = _SV_MIXED_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    return f"{swedish_nemo_cardinal(int(match.group(1)))} och halv"


def _fast_swedish_measure(span: str) -> Optional[str]:
    match = _SV_MEASURE_RE.fullmatch(span)
    if match is None:
        return None
    sign, number, unit = match.groups()
    one, singular, plural = _SV_MEASURE_UNITS[unit]
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _sv_decimal_words(integer, fraction)
        unit_word = plural
    else:
        value = int(number)
        words = one if value == 1 else swedish_nemo_cardinal(value)
        unit_word = singular if value == 1 else plural
    return _sv_signed(sign, f"{words} {unit_word}")


def _fast_swedish_ordinal(span: str) -> Optional[str]:
    match = _SV_ORDINAL_DOT_RE.fullmatch(span)
    if match is not None:
        return swedish_nemo_ordinal_dot(int(match.group(1)))
    match = _SV_ORDINAL_COLON_RE.fullmatch(span)
    if match is None:
        return None
    return _SV_COLON_ORDINALS.get((match.group(1), match.group(2)))


def _fast_swedish_time(span: str) -> Optional[str]:
    match = _SV_TIME_RE.fullmatch(span)
    if match is None:
        return None
    hour = swedish_nemo_cardinal(int(match.group(1)))
    minutes = match.group(2)
    if minutes is None or int(minutes) == 0:
        return f"klockan {hour}"
    return f"klockan {hour} {swedish_nemo_cardinal(int(minutes))}"


def _fast_swedish_money(span: str) -> Optional[str]:
    if "," in span:
        return None
    match = _SV_MONEY_RE.fullmatch(span)
    if match is None:
        return None
    symbol, symbol_number, krona_number, suffix_number, suffix_symbol = match.groups()
    if krona_number is not None:
        value = int(krona_number)
        if value == 1:
            return "en krona"
        return f"{swedish_nemo_cardinal(value)} kronor"
    if suffix_symbol is not None:
        symbol = suffix_symbol
        symbol_number = suffix_number
    value = int(symbol_number)
    if symbol == "€":
        return "en euro" if value == 1 else f"{swedish_nemo_cardinal(value)} euro"
    if symbol == "$":
        return "en dollar" if value == 1 else f"{swedish_nemo_cardinal(value)} dollar"
    return "ett pund" if value == 1 else f"{swedish_nemo_cardinal(value)} pund"


_SV_MONTHS = (
    "",
    "januari",
    "februari",
    "mars",
    "april",
    "maj",
    "juni",
    "juli",
    "augusti",
    "september",
    "oktober",
    "november",
    "december",
)
_SV_SHORT_DATE_RE = re.compile(r"^(0?[1-9]|[12]\d|3[01])\.(0?[1-9]|1[0-2])$")


_SV_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")


def _sv_year(year: int) -> str:
    century, yy = divmod(year, 100)
    head = swedish_nemo_cardinal(century)
    if yy == 0:
        return head + "hundra"
    if yy < 10:
        return head + "hundra" + swedish_nemo_cardinal(yy)
    return head + swedish_nemo_cardinal(yy)


def _fast_swedish_iso_date(span: str) -> Optional[str]:
    match = _SV_ISO_RE.fullmatch(span)
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    day_word = swedish_nemo_ordinal_dot(day)
    if day_word is None:
        return None
    return f"{day_word} {_SV_MONTHS[month]} {_sv_year(year)}"


def _fast_swedish_short_date(span: str) -> Optional[str]:
    match = _SV_SHORT_DATE_RE.fullmatch(span)
    if match is None:
        return None
    day = int(match.group(1))
    month = int(match.group(2))
    return f"{_SV_MONTHS[month]} {swedish_nemo_cardinal(day)}"


def _fast_swedish_dotted(span: str) -> Optional[str]:
    if span == "1.000":
        return span
    match = re.fullmatch(r"(\d+)\.(\d+)", span)
    if match is None:
        return None
    left = swedish_nemo_cardinal(int(match.group(1)))
    right = swedish_nemo_cardinal(int(match.group(2)))
    return f"{left} . {right}"


def fast_swedish_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Swedish classes; return None to the small graph when alignment or coverage fails."""

    if category == "date":
        iso = _fast_swedish_iso_date(span)
        if iso is not None:
            return iso
        return _fast_swedish_short_date(span)
    if category == "dotted":
        return _fast_swedish_dotted(span)
    if category == "integer":
        return _fast_swedish_integer(span)
    if category == "decimal":
        return _fast_swedish_decimal(span)
    if category == "percent":
        return _fast_swedish_percent(span)
    if category == "fraction":
        return _fast_swedish_fraction(span)
    if category == "measure":
        return _fast_swedish_measure(span)
    if category == "ordinal":
        return _fast_swedish_ordinal(span)
    if category == "time":
        return _fast_swedish_time(span)
    if category == "money":
        return _fast_swedish_money(span)
    return None


# Hungarian official TN: 2 alone is kettő, hundreds use két; thousands take hyphens; decimals are "egész … tized/század".
_HU_ONES = (
    "nulla",
    "egy",
    "kettő",
    "három",
    "négy",
    "öt",
    "hat",
    "hét",
    "nyolc",
    "kilenc",
)
_HU_TENS = {
    3: "harminc",
    4: "negyven",
    5: "ötven",
    6: "hatvan",
    7: "hetven",
    8: "nyolcvan",
    9: "kilencven",
}
_HU_ORDINALS = {
    1: "első",
    2: "második",
    3: "harmadik",
    4: "negyedik",
    5: "ötödik",
    6: "hatodik",
    7: "hetedik",
    8: "nyolcadik",
    9: "kilencedik",
    10: "tizedik",
    11: "tizenegyedik",
    12: "tizenkettedik",
    13: "tizenharmadik",
    20: "huszadik",
    21: "huszonegyedik",
    22: "huszonkettedik",
    30: "harmincadik",
    100: "századik",
    101: "százegyedik",
}
_HU_FRAC_DENOM = {
    2: "fél",
    3: "harmad",
    4: "negyed",
    5: "ötöd",
    8: "nyolcad",
}
_HU_MEASURE_UNITS = {
    "kg": "kilogramm",
    "g": "gramm",
    "mg": "milligramm",
    "km": "kilométer",
    "cm": "centiméter",
    "mm": "milliméter",
    "m": "méter",
}
_HU_DECIMAL_UNITS = {1: "tized", 2: "század", 3: "ezred"}
_HU_DECIMAL_RE = re.compile(r"^(-?)(\d+(?: \d{3})*),(\d+)$")
_HU_PERCENT_RE = re.compile(r"^(-?)(\d+(?: \d{3})*(?:,\d+)?)\s?%$")
_HU_MEASURE_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?(kg|km|cm|mm|mg|g|m)$")
_HU_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+(\d+)/(\d+)$")
_HU_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")
_HU_ORDINAL_RE = re.compile(r"^(\d+)\.$")
_HU_TIME_RE = re.compile(r"^(\d{2}):(\d{2})$")
_HU_MONEY_RE = re.compile(
    r"^(?:([$£€])\s?(\d+(?:,\d+)?)|(\d+(?:,\d+)?)\s?(?:Ft|€))$"
)


def _hu_digit(value: int) -> str:
    return _HU_ONES[value]


def _hu_under_100(value: int) -> str:
    if value < 10:
        return _hu_digit(value)
    if value == 10:
        return "tíz"
    if value < 20:
        return "tizen" + _hu_digit(value - 10)
    if value == 20:
        return "húsz"
    if value < 30:
        return "huszon" + _hu_digit(value - 20)
    tens, units = divmod(value, 10)
    if units == 0:
        return _HU_TENS[tens]
    return _HU_TENS[tens] + _hu_digit(units)


def _hu_under_1000(value: int) -> str:
    if value < 100:
        return _hu_under_100(value)
    hundreds, remainder = divmod(value, 100)
    if hundreds == 1:
        head = "száz"
    elif hundreds == 2:
        head = "kétszáz"
    else:
        head = _hu_digit(hundreds) + "száz"
    if remainder == 0:
        return head
    return head + _hu_under_100(remainder)


def hungarian_nemo_cardinal(value: int) -> str:
    """Match official deterministic Hungarian cardinals below one billion."""

    if value < 0:
        raise ValueError("cardinal only accepts non-negative integers")
    if value == 0:
        return "nulla"
    if value < 1000:
        return _hu_under_1000(value)
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        if thousands == 1:
            head = "ezer"
            if remainder == 0:
                return head
            return head + _hu_under_1000(remainder)
        head = _hu_under_1000(thousands) + "ezer"
        if remainder == 0:
            return head
        return f"{head}-{_hu_under_1000(remainder)}"
    millions, remainder = divmod(value, 1_000_000)
    head = "millió" if millions == 1 else _hu_under_1000(millions) + "millió"
    if remainder == 0:
        return head
    return f"{head}-{hungarian_nemo_cardinal(remainder)}"


def _hu_signed(sign: str, body: str) -> str:
    return f"mínusz {body}" if sign == "-" else body


def _hu_decimal_words(integer: str, fraction: str) -> Optional[str]:
    stripped = fraction.rstrip("0")
    if stripped == "":
        frac_words = "nulla"
        length = len(fraction)
    else:
        frac_words = hungarian_nemo_cardinal(int(stripped))
        length = len(stripped)
    unit = _HU_DECIMAL_UNITS.get(length)
    if unit is None:
        return None
    compact = integer.replace(" ", "")
    return f"{hungarian_nemo_cardinal(int(compact))} egész {frac_words} {unit}"


def _fast_hungarian_integer(span: str) -> Optional[str]:
    if span.startswith("+"):
        return None
    sign = ""
    body = span
    if body.startswith("-"):
        sign = "mínusz "
        body = body[1:]
    if body == "1.000":
        return sign + hungarian_nemo_cardinal(1000)
    if "." in body or "," in body:
        return None
    compact = body.replace(" ", "")
    if not compact.isdigit():
        return None
    if compact.startswith("0") and len(compact) > 1:
        return span
    value = int(compact)
    if value > 999_999_999:
        return None
    return sign + hungarian_nemo_cardinal(value)


def _fast_hungarian_decimal(span: str) -> Optional[str]:
    match = _HU_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    words = _hu_decimal_words(integer, fraction)
    if words is None:
        return None
    return _hu_signed(sign, words)


def _fast_hungarian_percent(span: str) -> Optional[str]:
    match = _HU_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _hu_decimal_words(integer, fraction)
        if words is None:
            return None
    else:
        words = hungarian_nemo_cardinal(int(number.replace(" ", "")))
    return _hu_signed(sign, f"{words} százalék")


def _fast_hungarian_fraction(span: str) -> Optional[str]:
    match = _HU_MIXED_FRACTION_RE.fullmatch(span)
    if match is not None:
        whole, num, den = match.groups()
        denom = _HU_FRAC_DENOM.get(int(den))
        if denom is None:
            return None
        return (
            f"{hungarian_nemo_cardinal(int(whole))} és "
            f"{hungarian_nemo_cardinal(int(num))} {denom}"
        )
    match = _HU_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    num, den = match.groups()
    denom = _HU_FRAC_DENOM.get(int(den))
    if denom is None:
        return None
    return f"{hungarian_nemo_cardinal(int(num))} {denom}"


def _fast_hungarian_measure(span: str) -> Optional[str]:
    match = _HU_MEASURE_RE.fullmatch(span)
    if match is None:
        return None
    sign, number, unit = match.groups()
    unit_word = _HU_MEASURE_UNITS[unit]
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _hu_decimal_words(integer, fraction)
        if words is None:
            return None
    else:
        words = hungarian_nemo_cardinal(int(number))
    return _hu_signed(sign, f"{words} {unit_word}")


def _fast_hungarian_ordinal(span: str) -> Optional[str]:
    match = _HU_ORDINAL_RE.fullmatch(span)
    if match is None:
        return None
    return _HU_ORDINALS.get(int(match.group(1)))


def _fast_hungarian_time(span: str) -> Optional[str]:
    match = _HU_TIME_RE.fullmatch(span)
    if match is None:
        return None
    hour = hungarian_nemo_cardinal(int(match.group(1)))
    minute_value = int(match.group(2))
    if minute_value == 0:
        return f"{hour} óra"
    return f"{hour} óra {hungarian_nemo_cardinal(minute_value)} perc"


def _fast_hungarian_money(span: str) -> Optional[str]:
    match = _HU_MONEY_RE.fullmatch(span)
    if match is None:
        return None
    symbol, symbol_number, forint_number = match.groups()
    number = forint_number if forint_number is not None else symbol_number
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _hu_decimal_words(integer, fraction)
        if words is None:
            return None
    else:
        words = hungarian_nemo_cardinal(int(number))
    if forint_number is not None:
        if span.rstrip().endswith("€"):
            return f"{words} euró"
        return f"{words} forint"
    if symbol == "€":
        return f"{words} euró"
    if symbol == "$":
        return f"{words} dollár"
    return f"{words} font"


_HU_PUNKT_RE = re.compile(r"^(\d+)\.(\d{1,2})$")


def _fast_hungarian_dotted(span: str) -> Optional[str]:
    match = _HU_PUNKT_RE.fullmatch(span)
    if match is None:
        return None
    integer, fraction = match.groups()
    integer_words = " ".join(_HU_ONES[int(ch)] for ch in integer)
    fraction_words = " ".join(_HU_ONES[int(ch)] for ch in fraction)
    return f"{integer_words} pont {fraction_words}"


def fast_hungarian_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Hungarian classes; return None to the small graph when alignment or coverage fails."""

    if category == "integer":
        return _fast_hungarian_integer(span)
    if category == "decimal":
        return _fast_hungarian_decimal(span)
    if category == "percent":
        return _fast_hungarian_percent(span)
    if category == "fraction":
        return _fast_hungarian_fraction(span)
    if category == "measure":
        return _fast_hungarian_measure(span)
    if category == "ordinal":
        return _fast_hungarian_ordinal(span)
    if category == "time":
        return _fast_hungarian_time(span)
    if category == "money":
        return _fast_hungarian_money(span)
    if category == "dotted":
        return _fast_hungarian_dotted(span)
    return None



# Vietnamese official TN: '.' thousands, comma decimals; 15/25 use lăm, 21/31 mốt, 34 tư.
_VI_ONES = (
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
)
_VI_MEASURE_UNITS = {
    "kg": "ki lô gam",
    "m": "mét",
    "h": "giờ",
}
_VI_DECIMAL_RE = re.compile(r"^(-?)(\d+(?:\.\d{3})*),(\d+)$")
_VI_PERCENT_RE = re.compile(r"^(-?)(\d+(?:\.\d{3})*(?:,\d+)?)\s?%$")
_VI_MEASURE_RE = re.compile(r"^(-?)(\d+(?:,\d+)?)\s?(kg|m|h)$")
_VI_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+(\d+)/(\d+)$")
_VI_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")
_VI_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_VI_MONEY_RE = re.compile(r"^(\d+(?:,\d+)?)\s?đồng$")


def _vi_ones(value: int, after: Optional[str] = None) -> str:
    if after == "mươi":
        if value == 1:
            return "mốt"
        if value == 4:
            return "tư"
        if value == 5:
            return "lăm"
    if after == "mười" and value == 5:
        return "lăm"
    return _VI_ONES[value]


def _vi_under_100(value: int) -> str:
    if value < 10:
        return _vi_ones(value)
    if value == 10:
        return "mười"
    if value < 20:
        return "mười " + _vi_ones(value - 10, after="mười")
    tens, units = divmod(value, 10)
    head = _vi_ones(tens) + " mươi"
    if units == 0:
        return head
    return head + " " + _vi_ones(units, after="mươi")


def _vi_under_1000(value: int) -> str:
    if value < 100:
        return _vi_under_100(value)
    hundreds, remainder = divmod(value, 100)
    head = _vi_ones(hundreds) + " trăm"
    if remainder == 0:
        return head
    if remainder < 10:
        return head + " linh " + _vi_ones(remainder)
    return head + " " + _vi_under_100(remainder)


def vietnamese_nemo_cardinal(value: int) -> str:
    """Match official deterministic Vietnamese cardinals below one billion."""

    if value < 0:
        raise ValueError("cardinal only accepts non-negative integers")
    if value == 0:
        return "không"
    if value < 1000:
        return _vi_under_1000(value)
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        head = vietnamese_nemo_cardinal(thousands) + " nghìn"
        if remainder == 0:
            return head
        if remainder < 10:
            return head + " linh " + _vi_ones(remainder)
        return head + " " + _vi_under_1000(remainder)
    millions, remainder = divmod(value, 1_000_000)
    head = vietnamese_nemo_cardinal(millions) + " triệu"
    if remainder == 0:
        return head
    if remainder < 10:
        return head + " linh " + _vi_ones(remainder)
    return head + " " + vietnamese_nemo_cardinal(remainder)


def _vi_signed(sign: str, body: str) -> str:
    return f"âm {body}" if sign == "-" else body


def _vi_digit_words(digits: str) -> str:
    return " ".join(_VI_ONES[int(char)] for char in digits)


def _vi_decimal_words(integer: str, fraction: str) -> str:
    compact = integer.replace(".", "")
    return f"{vietnamese_nemo_cardinal(int(compact))} phẩy {_vi_digit_words(fraction)}"


def _fast_vietnamese_integer(span: str) -> Optional[str]:
    if span.startswith("+"):
        return None
    sign = ""
    body = span
    if body.startswith("-"):
        sign = "âm "
        body = body[1:]
    if "," in body or " " in body:
        return None
    compact = body.replace(".", "")
    if not compact.isdigit():
        return None
    if compact.startswith("0") and len(compact) > 1:
        return None
    if "." in body and not re.fullmatch(r"[1-9]\d{0,2}(?:\.\d{3})+", body):
        return None
    value = int(compact)
    if value > 999_999_999:
        return None
    return sign + vietnamese_nemo_cardinal(value)


def _fast_vietnamese_decimal(span: str) -> Optional[str]:
    match = _VI_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    return _vi_signed(sign, _vi_decimal_words(integer, fraction))


def _fast_vietnamese_percent(span: str) -> Optional[str]:
    match = _VI_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _vi_decimal_words(integer, fraction)
    else:
        words = vietnamese_nemo_cardinal(int(number.replace(".", "")))
    return _vi_signed(sign, f"{words} phần trăm")


def _fast_vietnamese_fraction(span: str) -> Optional[str]:
    match = _VI_MIXED_FRACTION_RE.fullmatch(span)
    if match is not None:
        whole, num, den = (int(part) for part in match.groups())
        return (
            f"{vietnamese_nemo_cardinal(whole)} và "
            f"{vietnamese_nemo_cardinal(num)} phần {vietnamese_nemo_cardinal(den) if den != 4 else 'tư'}"
        )
    match = _VI_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    num, den = (int(part) for part in match.groups())
    denom = "tư" if den == 4 else vietnamese_nemo_cardinal(den)
    return f"{vietnamese_nemo_cardinal(num)} phần {denom}"


def _fast_vietnamese_measure(span: str) -> Optional[str]:
    match = _VI_MEASURE_RE.fullmatch(span)
    if match is None:
        return None
    sign, number, unit = match.groups()
    unit_word = _VI_MEASURE_UNITS[unit]
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _vi_decimal_words(integer, fraction)
    else:
        words = vietnamese_nemo_cardinal(int(number))
    return _vi_signed(sign, f"{words} {unit_word}")


def _fast_vietnamese_time(span: str) -> Optional[str]:
    match = _VI_TIME_RE.fullmatch(span)
    if match is None:
        return None
    hour = vietnamese_nemo_cardinal(int(match.group(1)))
    minute_value = int(match.group(2))
    if minute_value == 0:
        return f"{hour} giờ"
    return f"{hour} giờ {vietnamese_nemo_cardinal(minute_value)} phút"


def _fast_vietnamese_money(span: str) -> Optional[str]:
    match = _VI_MONEY_RE.fullmatch(span)
    if match is None:
        return None
    number = match.group(1)
    if "," in number:
        integer, fraction = number.split(",", 1)
        words = _vi_decimal_words(integer, fraction)
    else:
        words = vietnamese_nemo_cardinal(int(number))
    return f"{words} đồng"


_VI_PUNKT_RE = re.compile(r"^(\d+)\.(\d{1,2})$")
_VI_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")


def _fast_vietnamese_iso_date(span: str) -> Optional[str]:
    match = _VI_ISO_RE.fullmatch(span)
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return (
        f"ngày {vietnamese_nemo_cardinal(day)} tháng "
        f"{vietnamese_nemo_cardinal(month)} năm {vietnamese_nemo_cardinal(year)}"
    )


def _fast_vietnamese_dotted(span: str) -> Optional[str]:
    match = _VI_PUNKT_RE.fullmatch(span)
    if match is None:
        return None
    integer = vietnamese_nemo_cardinal(int(match.group(1)))
    fraction = vietnamese_nemo_cardinal(int(match.group(2)))
    return f"{integer}. {fraction}"


def fast_vietnamese_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Vietnamese classes; return None to the small graph when alignment or coverage fails."""

    if category == "date":
        return _fast_vietnamese_iso_date(span)
    if category == "integer":
        return _fast_vietnamese_integer(span)
    if category == "decimal":
        return _fast_vietnamese_decimal(span)
    if category == "percent":
        return _fast_vietnamese_percent(span)
    if category == "fraction":
        return _fast_vietnamese_fraction(span)
    if category == "measure":
        return _fast_vietnamese_measure(span)
    if category == "time":
        return _fast_vietnamese_time(span)
    if category == "money":
        return _fast_vietnamese_money(span)
    if category == "dotted":
        return _fast_vietnamese_dotted(span)
    return None



# Armenian official TN: space thousands; '.' is decimal (ամբողջ); 10:30 is time.
_HY_ONES = (
    "զրո",
    "մեկ",
    "երկու",
    "երեք",
    "չորս",
    "հինգ",
    "վեց",
    "յոթ",
    "ութ",
    "ինը",
)
_HY_TENS = {
    2: "քսան",
    3: "երեսուն",
    4: "քառասուն",
    5: "հիսուն",
    6: "վաթսուն",
    7: "յոթանասուն",
    8: "ութսուն",
    9: "իննսուն",
}
_HY_ORDINALS = {
    1: "առաջին",
    2: "երկրորդ",
    3: "երրորդ",
    4: "չորրորդ",
    5: "հինգերորդ",
    6: "վեցերորդ",
    7: "յոթերորդ",
    8: "ութերորդ",
    9: "իններորդ",
    10: "տասներորդ",
    11: "տասնմեկերորդ",
    12: "տասներկուերորդ",
    13: "տասներեքերորդ",
    20: "քսաներորդ",
    21: "քսանմեկերորդ",
    22: "քսաներկուերորդ",
    30: "երեսուներորդ",
    100: "հարյուրերորդ",
}
_HY_DECIMAL_RE = re.compile(r"^(-?)(\d+)\.(\d+)$")
_HY_PERCENT_RE = re.compile(r"^(-?)(\d+)\s?%$")
_HY_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")
_HY_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+(\d+)/(\d+)$")
_HY_ORDINAL_RE = re.compile(r"^(\d+)-(ին|րդ)$")
_HY_TIME_RE = re.compile(r"^(\d{2}):(\d{2})$")
_HY_MONEY_RE = re.compile(r"^(\d+)\s?դրամ$")
_HY_ARMENIAN_HYPHEN = "\u058a"


def _hy_under_100(value: int) -> str:
    if value < 10:
        return _HY_ONES[value]
    if value == 10:
        return "տասը"
    if value < 20:
        return "տասն" + _HY_ONES[value - 10]
    tens, units = divmod(value, 10)
    if units == 0:
        return _HY_TENS[tens]
    return _HY_TENS[tens] + _HY_ONES[units]


def _hy_under_1000(value: int) -> str:
    if value < 100:
        return _hy_under_100(value)
    hundreds, remainder = divmod(value, 100)
    head = "հարյուր" if hundreds == 1 else _hy_under_100(hundreds) + " հարյուր"
    if remainder == 0:
        return head
    return head + " " + _hy_under_100(remainder)


def armenian_nemo_cardinal(value: int) -> str:
    """Match official deterministic Armenian cardinals below one billion."""

    if value < 0:
        raise ValueError("cardinal only accepts non-negative integers")
    if value == 0:
        return "զրո"
    if value < 1000:
        return _hy_under_1000(value)
    if value < 1_000_000:
        thousands, remainder = divmod(value, 1000)
        head = "հազար" if thousands == 1 else _hy_under_1000(thousands) + " հազար"
        if remainder == 0:
            return head
        return head + " " + _hy_under_1000(remainder)
    millions, remainder = divmod(value, 1_000_000)
    head = armenian_nemo_cardinal(millions) + " միլիոն"
    if remainder == 0:
        return head
    return head + " " + armenian_nemo_cardinal(remainder)


def _hy_signed(sign: str, body: str) -> str:
    return f"- {body}" if sign == "-" else body


def _fast_armenian_integer(span: str) -> Optional[str]:
    if span.startswith("+"):
        return None
    sign = ""
    body = span
    if body.startswith("-"):
        sign = "- "
        body = body[1:]
    if "." in body or "," in body:
        return None
    compact = body.replace(" ", "")
    if not compact.isdigit():
        return None
    if compact.startswith("0") and len(compact) > 1:
        return span
    value = int(compact)
    if value > 999_999_999:
        return None
    return sign + armenian_nemo_cardinal(value)


def _fast_armenian_decimal(span: str) -> Optional[str]:
    match = _HY_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    words = (
        f"{armenian_nemo_cardinal(int(integer))} ամբողջ "
        f"{armenian_nemo_cardinal(int(fraction))}"
    )
    return _hy_signed(sign, words)


def _fast_armenian_percent(span: str) -> Optional[str]:
    match = _HY_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    return _hy_signed(sign, f"{armenian_nemo_cardinal(int(number))} տոկոս")


def _fast_armenian_fraction(span: str) -> Optional[str]:
    match = _HY_MIXED_FRACTION_RE.fullmatch(span)
    if match is not None:
        whole, num, den = (int(part) for part in match.groups())
        denom = _HY_ORDINALS.get(den)
        if denom is None:
            return None
        return f"{armenian_nemo_cardinal(whole)} {armenian_nemo_cardinal(num)} {denom}"
    match = _HY_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    num, den = (int(part) for part in match.groups())
    denom = _HY_ORDINALS.get(den)
    if denom is None:
        return None
    return f"{armenian_nemo_cardinal(num)} {denom}"


def _fast_armenian_ordinal(span: str) -> Optional[str]:
    match = _HY_ORDINAL_RE.fullmatch(span)
    if match is None:
        return None
    return _HY_ORDINALS.get(int(match.group(1)))


def _fast_armenian_time(span: str) -> Optional[str]:
    match = _HY_TIME_RE.fullmatch(span)
    if match is None:
        return None
    hour_value = int(match.group(1))
    minute_value = int(match.group(2))
    hour_words = {
        2: "երկուսն",
        10: "տասն",
        12: "տասներկուսն",
    }
    hour = hour_words.get(hour_value, armenian_nemo_cardinal(hour_value))
    if minute_value == 0:
        minute = f"զրո{_HY_ARMENIAN_HYPHEN}զրո"
    else:
        minute = armenian_nemo_cardinal(minute_value)
    return f"{hour} անց {minute}"


_HY_MONEY_SYMBOL_RE = re.compile(r"^(?:([€$])\s?(\d+)|(\d+)\s?([€$]))$")


def _fast_armenian_money(span: str) -> Optional[str]:
    match = _HY_MONEY_RE.fullmatch(span)
    if match is not None:
        return f"{armenian_nemo_cardinal(int(match.group(1)))} դրամ"
    match = _HY_MONEY_SYMBOL_RE.fullmatch(span)
    if match is None:
        return None
    symbol, number, suffix_number, suffix_symbol = match.groups()
    if suffix_symbol is not None:
        symbol = suffix_symbol
        number = suffix_number
    unit = "եվրո" if symbol == "€" else "դոլար"
    return f"{armenian_nemo_cardinal(int(number))} {unit}"


def fast_armenian_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Armenian classes; return None to the small graph when alignment or coverage fails."""

    if category == "integer":
        return _fast_armenian_integer(span)
    if category == "decimal":
        return _fast_armenian_decimal(span)
    if category == "percent":
        return _fast_armenian_percent(span)
    if category == "fraction":
        return _fast_armenian_fraction(span)
    if category == "ordinal":
        return _fast_armenian_ordinal(span)
    if category == "time":
        return _fast_armenian_time(span)
    if category == "money":
        return _fast_armenian_money(span)
    return None



# Korean official TN: Sino-Korean cardinals have no spaces; '.' decimals are digit-wise; 1-12 o'clock uses native numerals.
_KO_SINO = ("영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
_KO_NATIVE_HOUR = {
    1: "한",
    2: "두",
    3: "세",
    4: "네",
    5: "다섯",
    6: "여섯",
    7: "일곱",
    8: "여덟",
    9: "아홉",
    10: "열",
    11: "열한",
    12: "열두",
}
_KO_DECIMAL_RE = re.compile(r"^(-?)(\d+)\.(\d+)$")
_KO_PERCENT_RE = re.compile(r"^(-?)(\d+(?:\.\d+)?)\s?%$")
_KO_MEASURE_RE = re.compile(r"^(-?)(\d+(?:\.\d+)?)\s?(kg|km|h|g)$")
_KO_MEASURE_UNITS = {
    "kg": "킬로그램",
    "km": "킬로미터",
    "h": "시간",
    "g": "그램",
}
_KO_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")
_KO_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+(\d+)/(\d+)$")
_KO_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_KO_MONEY_RE = re.compile(r"^(\d+)원$")


def korean_nemo_cardinal(value: int) -> str:
    """Match official deterministic Korean sino cardinals."""

    if value < 0:
        raise ValueError("cardinal only accepts non-negative integers")
    if value == 0:
        return "영"
    if value < 10:
        return _KO_SINO[value]
    if value < 100:
        tens, units = divmod(value, 10)
        head = "십" if tens == 1 else _KO_SINO[tens] + "십"
        return head if units == 0 else head + _KO_SINO[units]
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        head = "백" if hundreds == 1 else _KO_SINO[hundreds] + "백"
        return head if remainder == 0 else head + korean_nemo_cardinal(remainder)
    if value < 10_000:
        thousands, remainder = divmod(value, 1000)
        head = "천" if thousands == 1 else korean_nemo_cardinal(thousands) + "천"
        return head if remainder == 0 else head + korean_nemo_cardinal(remainder)
    if value < 100_000_000:
        man, remainder = divmod(value, 10_000)
        head = "만" if man == 1 else korean_nemo_cardinal(man) + "만"
        return head if remainder == 0 else head + korean_nemo_cardinal(remainder)
    if value < 1_000_000_000_000:
        eok, remainder = divmod(value, 100_000_000)
        head = "일억" if eok == 1 else korean_nemo_cardinal(eok) + "억"
        return head if remainder == 0 else head + korean_nemo_cardinal(remainder)
    return None


def _ko_signed(sign: str, body: str) -> str:
    return f"마이너스 {body}" if sign == "-" else body


def _ko_digit_words(digits: str) -> str:
    return "".join(_KO_SINO[int(char)] for char in digits)


def _fast_korean_integer(span: str) -> Optional[str]:
    if span.startswith("+"):
        return None
    sign = ""
    body = span
    if body.startswith("-"):
        sign = "마이너스 "
        body = body[1:]
    if "." in body or "," in body or " " in body:
        return None
    if not body.isdigit():
        return None
    if body.startswith("0") and len(body) > 1:
        return None
    value = int(body)
    words = korean_nemo_cardinal(value)
    if words is None:
        return None
    return sign + words


def _fast_korean_decimal(span: str) -> Optional[str]:
    match = _KO_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    words = korean_nemo_cardinal(int(integer)) + "점" + _ko_digit_words(fraction)
    return _ko_signed(sign, words)


def _fast_korean_percent(span: str) -> Optional[str]:
    match = _KO_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    if "." in number:
        integer, fraction = number.split(".", 1)
        words = korean_nemo_cardinal(int(integer)) + "점" + _ko_digit_words(fraction)
    else:
        words = korean_nemo_cardinal(int(number))
    return _ko_signed(sign, f"{words} 퍼센트")


def _fast_korean_measure(span: str) -> Optional[str]:
    match = _KO_MEASURE_RE.fullmatch(span)
    if match is None:
        return None
    sign, number, unit = match.groups()
    unit_word = _KO_MEASURE_UNITS.get(unit)
    if unit_word is None:
        return None
    if "." in number:
        integer, fraction = number.split(".", 1)
        words = korean_nemo_cardinal(int(integer)) + "점" + _ko_digit_words(fraction)
    else:
        words = korean_nemo_cardinal(int(number))
    return _ko_signed(sign, f"{words} {unit_word}")


def _fast_korean_fraction(span: str) -> Optional[str]:
    match = _KO_MIXED_FRACTION_RE.fullmatch(span)
    if match is not None:
        whole, num, den = (int(part) for part in match.groups())
        return (
            f"{korean_nemo_cardinal(whole)} "
            f"{korean_nemo_cardinal(den)}분의 {korean_nemo_cardinal(num)}"
        )
    match = _KO_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    num, den = (int(part) for part in match.groups())
    return f"{korean_nemo_cardinal(den)}분의 {korean_nemo_cardinal(num)}"


def _fast_korean_time(span: str) -> Optional[str]:
    match = _KO_TIME_RE.fullmatch(span)
    if match is None:
        return None
    hour_value = int(match.group(1))
    minute_value = int(match.group(2))
    hour = _KO_NATIVE_HOUR.get(hour_value, korean_nemo_cardinal(hour_value))
    if minute_value == 0:
        return f"{hour}시"
    return f"{hour}시 {korean_nemo_cardinal(minute_value)}분"


def _fast_korean_money(span: str) -> Optional[str]:
    match = _KO_MONEY_RE.fullmatch(span)
    if match is None:
        return None
    return f"{korean_nemo_cardinal(int(match.group(1)))}원"


def fast_korean_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Korean classes; return None to the small graph when alignment or coverage fails."""

    if category == "integer":
        return _fast_korean_integer(span)
    if category == "decimal":
        return _fast_korean_decimal(span)
    if category == "percent":
        return _fast_korean_percent(span)
    if category == "measure":
        return _fast_korean_measure(span)
    if category == "fraction":
        return _fast_korean_fraction(span)
    if category == "time":
        return _fast_korean_time(span)
    if category == "money":
        return _fast_korean_money(span)
    return None



# Arabic official TN stably converts Western digits 1-9999; Arabic-Indic ١ is left unchanged.
_AR_ONES = {
    1: "واحد",
    2: "اثنان",
    3: "ثلاثة",
    4: "أربعة",
    5: "خمسة",
    6: "ستة",
    7: "سبعة",
    8: "ثمانية",
    9: "تسعة",
}
_AR_TEENS = {
    11: "احد عشر",
    12: "اثنا عشر",
    13: "ثلاثة عشر",
    14: "اربعة عشر",
    15: "خمسة عشر",
    16: "ستة عشر",
    17: "سبعة عشر",
    18: "ثمانية عشر",
    19: "تسعة عشر",
}
_AR_TENS = {
    2: "عشرون",
    3: "ثلاثون",
    4: "أربعون",
    5: "خمسون",
    6: "ستون",
    7: "سبعون",
    8: "ثمانون",
    9: "تسعون",
}
_AR_HUNDREDS = {
    3: "ثلاث",
    4: "أربع",
    5: "خمس",
    6: "ست",
    7: "سبع",
    8: "ثمان",
    9: "تسع",
}
_AR_FRACTION_SINGULAR = {
    2: "نصف",
    3: "ثلث",
    4: "ربع",
    5: "خمس",
    6: "سدس",
    7: "سبع",
    8: "ثمن",
    9: "تسع",
    10: "عشر",
}
_AR_FRACTION_PLURAL = {
    3: "أثلاث",
    4: "أرباع",
    5: "أخماس",
    6: "أسداس",
    7: "أسباع",
    8: "أثمان",
    9: "أتساع",
    10: "أعشار",
}
_AR_DECIMAL_SCALE = {1: "عشرة", 2: "مئة", 3: "ألف", 4: "عشرة آلاف"}
_AR_DECIMAL_RE = re.compile(r"^(-?)(\d+)[.,](\d{1,4})$")
_AR_PERCENT_RE = re.compile(r"^(-?)(\d+(?:[.,]\d{1,4})?)\s?%$")
_AR_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")
_AR_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+(\d+)/(\d+)$")
_AR_MONEY_RE = re.compile(r"^(?:(€|\$)(\d+)|(\d+)(€|\$))$")
_AR_MONEY_WORD_RE = re.compile(r"^(\d+)\sدولار$")


def _ar_and(left: str, right: str) -> str:
    return f"{left} و{right}"


def arabic_nemo_cardinal(value: int) -> Optional[str]:
    """Match official deterministic Arabic cardinals; only 1-9999."""

    if value in {1100, 2100}:
        return "مئة ومئة" if value == 1100 else "مئتان ومئة"
    if value <= 0 or value >= 10_000:
        return None
    if value < 10:
        return _AR_ONES[value]
    if value == 10:
        return "عشرة"
    if value < 20:
        return _AR_TEENS[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        if ones == 0:
            return _AR_TENS[tens]
        return _ar_and(_AR_ONES[ones], _AR_TENS[tens])
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        if hundreds == 1:
            head = "مئة"
        elif hundreds == 2:
            head = "مئتان"
        else:
            head = f"{_AR_HUNDREDS[hundreds]} مئة"
        return head if remainder == 0 else _ar_and(head, arabic_nemo_cardinal(remainder))
    thousands, remainder = divmod(value, 1000)
    if thousands == 1:
        head = "ألف"
    elif thousands == 2:
        head = "ألفان"
    else:
        head = f"{_AR_ONES[thousands]} اَلاف"
    return head if remainder == 0 else _ar_and(head, arabic_nemo_cardinal(remainder))


def _ar_signed_cardinal(sign: str, value: int) -> Optional[str]:
    words = arabic_nemo_cardinal(value)
    if words is None:
        return None
    return f"minus {words}" if sign == "-" else words


def _fast_arabic_integer(span: str) -> Optional[str]:
    if span.startswith("+"):
        return None
    sign = ""
    body = span
    if body.startswith("-"):
        sign = "-"
        body = body[1:]
    if not body.isdigit() or body.startswith("0"):
        return None
    return _ar_signed_cardinal(sign, int(body))


def _fast_arabic_decimal_parts(integer: str, fraction: str) -> Optional[str]:
    if integer.startswith("0") and integer != "0":
        return None
    integer_words = arabic_nemo_cardinal(int(integer))
    fraction_words = arabic_nemo_cardinal(int(fraction))
    scale = _AR_DECIMAL_SCALE.get(len(fraction))
    if integer_words is None or fraction_words is None or scale is None:
        return None
    return _ar_and(integer_words, fraction_words) + f" من {scale}"


def _fast_arabic_decimal(span: str) -> Optional[str]:
    match = _AR_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    words = _fast_arabic_decimal_parts(integer, fraction)
    if words is None:
        return None
    # Official negative decimals glue سالب onto the integer; skip them in the speed set and handle positives only.
    if sign == "-":
        return None
    return words


def _fast_arabic_percent(span: str) -> Optional[str]:
    match = _AR_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    if re.search(r"[.,]", number):
        integer, fraction = re.split(r"[.,]", number, maxsplit=1)
        words = _fast_arabic_decimal_parts(integer, fraction)
    else:
        words = arabic_nemo_cardinal(int(number))
    if words is None:
        return None
    body = f"{words} في المائة"
    return f"minus {body}" if sign == "-" else body


def _fast_arabic_fraction(span: str) -> Optional[str]:
    match = _AR_MIXED_FRACTION_RE.fullmatch(span)
    if match is not None:
        whole, num, den = (int(part) for part in match.groups())
        if num != 1 or den not in _AR_FRACTION_SINGULAR:
            return None
        whole_words = arabic_nemo_cardinal(whole)
        if whole_words is None:
            return None
        return f"{whole_words} و {_AR_FRACTION_SINGULAR[den]}"
    match = _AR_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    num, den = (int(part) for part in match.groups())
    if num == 1 and den in _AR_FRACTION_SINGULAR:
        return _AR_FRACTION_SINGULAR[den]
    if num > 1 and den in _AR_FRACTION_PLURAL:
        num_words = arabic_nemo_cardinal(num)
        if num_words is None:
            return None
        return f"{num_words} {_AR_FRACTION_PLURAL[den]}"
    return None


def _fast_arabic_money(span: str) -> Optional[str]:
    match = _AR_MONEY_RE.fullmatch(span)
    if match is not None:
        symbol, number, suffix_number, suffix_symbol = match.groups()
        if suffix_symbol is not None:
            symbol = suffix_symbol
            number = suffix_number
        words = arabic_nemo_cardinal(int(number))
        if words is None:
            return None
        unit = "يورو" if symbol == "€" else "دولار"
        return f"{words} {unit}"
    match = _AR_MONEY_WORD_RE.fullmatch(span)
    if match is None:
        return None
    words = arabic_nemo_cardinal(int(match.group(1)))
    if words is None:
        return None
    return f"{words} دولار"


def fast_arabic_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Arabic classes; return None to the small graph when alignment or coverage fails."""

    if category == "integer":
        return _fast_arabic_integer(span)
    if category == "decimal":
        return _fast_arabic_decimal(span)
    if category == "percent":
        return _fast_arabic_percent(span)
    if category == "fraction":
        return _fast_arabic_fraction(span)
    if category == "money":
        return _fast_arabic_money(span)
    return None



_HI_0_99 = (
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
    "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस",
    "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस",
    "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास", "इक्यावन", "बावन", "तिरेपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरेसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर",
    "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",
    "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी",
    "नब्बे", "इक्यानबे", "बानबे", "तिरानबे", "चौरानबे", "पंचानबे", "छियानबे", "सत्तानबे", "अट्ठानबे", "निन्यानबे",
)
_HI_DIGIT = tuple(_HI_0_99[:10])
_HI_DECIMAL_RE = re.compile(r"^(-?)(\d+)\.(\d+)$")
_HI_MEASURE_RE = re.compile(r"^(-?)(\d+(?:\.\d+)?)\s?(kg|g|h)$")
_HI_MONEY_RE = re.compile(r"^(?:(€|\$)(\d+)|(\d+)(€|\$))$")
_HI_MEASURE_UNITS = {"kg": "किलोग्राम", "g": "ग्राम", "h": "घंटे"}

_CJK_DIGITS = "零一二三四五六七八九"
_CJK_DECIMAL_RE = re.compile(r"^(-?)(\d+)\.(\d+)$")
_CJK_PERCENT_RE = re.compile(r"^(\d+(?:\.\d+)?)%$")
_CJK_FRACTION_RE = re.compile(r"^(\d+)/(\d+)$")
_CJK_MIXED_FRACTION_RE = re.compile(r"^(\d+)\s+(\d+)/(\d+)$")
_CJK_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_CJK_ORDINAL_RE = re.compile(r"^第(\d+)$")
_ZH_MONEY_RE = re.compile(r"^(?:(€|\$)(\d+)|(\d+)(€|\$))$")
_ZH_KG_RE = re.compile(r"^(-?)(\d+)kg$")


def hindi_nemo_cardinal(value: int) -> Optional[str]:
    """Match official Hindi cardinals: 0-99 lookup, hundreds/thousands as सौ / हज़ार."""

    if value < 0 or value >= 100_000:
        return None
    if value < 100:
        return _HI_0_99[value]
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        head = f"{_HI_0_99[hundreds]} सौ"
        return head if remainder == 0 else f"{head} {hindi_nemo_cardinal(remainder)}"
    thousands, remainder = divmod(value, 1000)
    head = f"{hindi_nemo_cardinal(thousands)} हज़ार"
    return head if remainder == 0 else f"{head} {hindi_nemo_cardinal(remainder)}"


def _fast_hindi_integer(span: str) -> Optional[str]:
    sign = ""
    body = span
    if body.startswith("+"):
        return None
    if body.startswith("-"):
        sign = "minus "
        body = body[1:]
    if not body.isdigit() or (body.startswith("0") and body != "0"):
        return None
    words = hindi_nemo_cardinal(int(body))
    if words is None:
        return None
    return sign + words


def _hi_number_words(number: str) -> Optional[str]:
    if "." in number:
        integer, fraction = number.split(".", 1)
        integer_words = hindi_nemo_cardinal(int(integer))
        if integer_words is None:
            return None
        return integer_words + " point " + " ".join(_HI_DIGIT[int(ch)] for ch in fraction)
    return hindi_nemo_cardinal(int(number))


def _fast_hindi_decimal(span: str) -> Optional[str]:
    match = _HI_DECIMAL_RE.fullmatch(span)
    if match is None or match.group(1) == "-":
        return None
    return _hi_number_words(match.group(2) + "." + match.group(3))


def _fast_hindi_measure(span: str) -> Optional[str]:
    match = _HI_MEASURE_RE.fullmatch(span)
    if match is None:
        return None
    sign, number, unit = match.groups()
    words = _hi_number_words(number)
    unit_word = _HI_MEASURE_UNITS.get(unit)
    if words is None or unit_word is None or sign == "-":
        return None
    return f"{words} {unit_word}"


def _fast_hindi_money(span: str) -> Optional[str]:
    match = _HI_MONEY_RE.fullmatch(span)
    if match is None:
        return None
    symbol, number, suffix_number, suffix_symbol = match.groups()
    if suffix_symbol is not None:
        symbol = suffix_symbol
        number = suffix_number
    words = hindi_nemo_cardinal(int(number))
    if words is None:
        return None
    return f"{words} {'यूरो' if symbol == '€' else 'डॉलर'}"


def fast_hindi_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Hindi classes."""

    if category == "integer":
        return _fast_hindi_integer(span)
    if category == "decimal":
        return _fast_hindi_decimal(span)
    if category == "measure":
        return _fast_hindi_measure(span)
    if category == "money":
        return _fast_hindi_money(span)
    return None


def _cjk_under_10000(value: int) -> str:
    if value == 0:
        return ""
    parts = []
    started = False
    need_zero = False
    remainder = value
    for scale, name, use_liang in (
        (1000, "千", True),
        (100, "百", False),
        (10, "十", False),
        (1, "", False),
    ):
        digit, remainder = divmod(remainder, scale)
        if digit == 0:
            if started and remainder:
                need_zero = True
            continue
        if need_zero:
            parts.append("零")
            need_zero = False
        if digit == 2 and use_liang:
            word = "两"
        elif digit == 1 and name == "十" and not started:
            word = ""
        else:
            word = _CJK_DIGITS[digit]
        parts.append(word + name)
        started = True
    return "".join(parts)


def cjk_nemo_cardinal(value: int) -> Optional[str]:
    """Match official ja/zh deterministic cardinals: hundreds use 一/二, thousands/myriads use 两 for 2."""

    if value < 0 or value >= 100_000_000:
        return None
    if value == 0:
        return "零"
    wan, rest = divmod(value, 10_000)
    if wan == 0:
        return _cjk_under_10000(rest)
    wan_words = "两" if wan == 2 else _cjk_under_10000(wan)
    if rest == 0:
        return wan_words + "万"
    rest_words = _cjk_under_10000(rest)
    if rest < 1000:
        return wan_words + "万零" + rest_words
    return wan_words + "万" + rest_words


def _cjk_digit_words(digits: str) -> str:
    return "".join(_CJK_DIGITS[int(ch)] for ch in digits)


def _fast_cjk_integer(span: str, minus_word: str) -> Optional[str]:
    sign = ""
    body = span
    if body.startswith("+"):
        return None
    if body.startswith("-"):
        sign = minus_word
        body = body[1:]
    if not body.isdigit() or (body.startswith("0") and body != "0"):
        return None
    words = cjk_nemo_cardinal(int(body))
    if words is None:
        return None
    return sign + words


def _fast_cjk_decimal_parts(integer: str, fraction: str) -> Optional[str]:
    words = cjk_nemo_cardinal(int(integer))
    if words is None:
        return None
    return words + "点" + _cjk_digit_words(fraction)


def _fast_cjk_decimal(span: str, minus_word: str) -> Optional[str]:
    match = _CJK_DECIMAL_RE.fullmatch(span)
    if match is None:
        return None
    sign, integer, fraction = match.groups()
    words = _fast_cjk_decimal_parts(integer, fraction)
    if words is None:
        return None
    return minus_word + words if sign == "-" else words


def _fast_cjk_time(span: str) -> Optional[str]:
    match = _CJK_TIME_RE.fullmatch(span)
    if match is None:
        return None
    hour = cjk_nemo_cardinal(int(match.group(1)))
    minute = cjk_nemo_cardinal(int(match.group(2)))
    if hour is None or minute is None:
        return None
    return f"{hour}点{minute}分"


def _fast_cjk_fraction(span: str, particle: str) -> Optional[str]:
    match = _CJK_MIXED_FRACTION_RE.fullmatch(span)
    if match is not None:
        whole, num, den = (int(part) for part in match.groups())
        whole_words = cjk_nemo_cardinal(whole)
        den_words = cjk_nemo_cardinal(den)
        num_words = cjk_nemo_cardinal(num)
        if None in {whole_words, den_words, num_words}:
            return None
        return f"{whole_words} {den_words}{particle}{num_words}"
    match = _CJK_FRACTION_RE.fullmatch(span)
    if match is None:
        return None
    num, den = (int(part) for part in match.groups())
    den_words = cjk_nemo_cardinal(den)
    num_words = cjk_nemo_cardinal(num)
    if den_words is None or num_words is None:
        return None
    return f"{den_words}{particle}{num_words}"


def _fast_cjk_ordinal(span: str) -> Optional[str]:
    match = _CJK_ORDINAL_RE.fullmatch(span)
    if match is None:
        return None
    words = cjk_nemo_cardinal(int(match.group(1)))
    if words is None:
        return None
    return "第" + words


def _fast_japanese_percent(span: str) -> Optional[str]:
    match = re.fullmatch(r"^(\d+)%$", span)
    if match is None:
        return None
    words = cjk_nemo_cardinal(int(match.group(1)))
    if words is None:
        return None
    return "百分の" + words


def _fast_chinese_percent(span: str) -> Optional[str]:
    match = _CJK_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    number = match.group(1)
    if "." in number:
        integer, fraction = number.split(".", 1)
        words = _fast_cjk_decimal_parts(integer, fraction)
    else:
        words = cjk_nemo_cardinal(int(number))
    if words is None:
        return None
    return "百分之" + words


def _fast_chinese_money(span: str) -> Optional[str]:
    match = _ZH_MONEY_RE.fullmatch(span)
    if match is None:
        return None
    symbol, number, suffix_number, suffix_symbol = match.groups()
    if suffix_symbol is not None:
        symbol = suffix_symbol
        number = suffix_number
    words = cjk_nemo_cardinal(int(number))
    if words is None:
        return None
    return words + ("欧元" if symbol == "€" else "美元")


def _fast_chinese_kg(span: str) -> Optional[str]:
    match = _ZH_KG_RE.fullmatch(span)
    if match is None:
        return None
    sign, number = match.groups()
    words = cjk_nemo_cardinal(int(number))
    if words is None or sign == "-":
        return None
    return words + "千克"


def fast_japanese_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Japanese classes."""

    if category == "integer":
        return _fast_cjk_integer(span, minus_word="-")
    if category == "decimal":
        return _fast_cjk_decimal(span, minus_word="负")
    if category == "percent":
        return _fast_japanese_percent(span)
    if category == "fraction":
        return _fast_cjk_fraction(span, "分の")
    if category == "time":
        return _fast_cjk_time(span)
    if category == "ordinal":
        return _fast_cjk_ordinal(span)
    return None


def fast_chinese_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for safe Chinese classes."""

    if category == "integer":
        return _fast_cjk_integer(span, minus_word="负")
    if category == "decimal":
        return _fast_cjk_decimal(span, minus_word="负")
    if category == "percent":
        return _fast_chinese_percent(span)
    if category == "fraction":
        return _fast_cjk_fraction(span, "分之")
    if category == "time":
        return _fast_cjk_time(span)
    if category == "ordinal":
        return _fast_cjk_ordinal(span)
    if category == "money":
        return _fast_chinese_money(span)
    if category == "measure":
        return _fast_chinese_kg(span)
    return None


def _instantiate_with_supported_kwargs(cls, **kwargs):
    """ClassifyFst/VerbalizeFinalFst signatures differ per language; pass only accepted kwargs."""

    signature = inspect.signature(cls.__init__)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return cls(**kwargs)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind != inspect.Parameter.VAR_POSITIONAL
    }
    return cls(**{key: value for key, value in kwargs.items() if key in accepted})


def supports_direct(language: str, itn: bool) -> bool:
    """Private direct currently covers accepted languages only; other requests must fall back to official."""

    if itn:
        return language in DIRECT_ITN_LANGUAGES
    return language in DIRECT_TN_LANGUAGES


@dataclass(frozen=True)
class NemoDirectItnConfig:
    """Direct cache config for accepted NeMo ITN languages; this project composes the FAR into cache_dir."""

    language: str
    input_case: str = "lower_cased"
    cache_dir: str = DEFAULT_NEMO_CACHE_DIR
    overwrite_cache: bool = False
    whitelist: Optional[str] = None
    cache_search_dirs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.language not in DIRECT_ITN_LANGUAGES:
            raise NemoLanguageNotSupportedError(
                "direct ITN only covers official InverseNormalizer languages: "
                + ", ".join(sorted(DIRECT_ITN_LANGUAGES))
            )
        if self.input_case not in {"cased", "lower_cased"}:
            raise ValueError("nemo_input_case must be cased or lower_cased")
        object.__setattr__(self, "cache_search_dirs", tuple(self.cache_search_dirs or ()))


@dataclass(frozen=True)
class NemoDirectTnConfig:
    """Direct-output TN cache config for accepted languages."""

    language: str
    input_case: str = "cased"
    cache_dir: str = DEFAULT_NEMO_CACHE_DIR
    overwrite_cache: bool = False
    whitelist: Optional[str] = None
    cache_search_dirs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.language not in DIRECT_TN_LANGUAGES:
            raise NemoLanguageNotSupportedError(
                f"direct-output TN is only verified for {', '.join(sorted(DIRECT_TN_LANGUAGES))}"
            )
        if self.input_case not in {"cased", "lower_cased"}:
            raise ValueError("nemo_input_case must be cased or lower_cased")
        object.__setattr__(self, "cache_search_dirs", tuple(self.cache_search_dirs or ()))


def ensure_official_fallback_cache(
    config: Union["NemoDirectTnConfig", "NemoDirectItnConfig"],
    *,
    itn: bool,
) -> None:
    """On direct cold compose, preheat the fallback graph from the official fingerprint. Fallback is the official large graph, not a direct small graph. Hot load skips this so every process does not hold official memory."""

    if itn:
        get_nemo_itn_engine(
            NemoInverseNormalizerConfig(
                language=config.language,
                input_case=config.input_case,
                cache_dir=config.cache_dir,
                overwrite_cache=config.overwrite_cache,
                whitelist=config.whitelist,
                cache_search_dirs=config.cache_search_dirs,
            )
        )
        return
    get_nemo_engine(
        NemoNormalizerConfig(
            language=config.language,
            input_case=config.input_case,
            cache_dir=config.cache_dir,
            overwrite_cache=config.overwrite_cache,
            whitelist=config.whitelist,
            cache_search_dirs=config.cache_search_dirs,
        )
    )


def _load_pynini_module() -> Any:
    """direct FAR depends only on the Pynini runtime and does not import NeMo grammar modules."""

    try:
        return importlib.import_module("pynini")
    except (ImportError, OSError) as exc:
        raise NemoDependencyError(
            "direct-output FAR runtime 不可用，请安装 requirements-nemo.txt；"
            f"原始错误：{exc}"
        ) from exc


@dataclass(frozen=True)
class _DirectTnProfile:
    """Direct TN routing for one accepted language; do not reuse regexes across languages."""

    graph_names: Tuple[str, ...]
    date_re: re.Pattern[str]
    candidate_re: re.Pattern[str]
    abbreviation_re: re.Pattern[str]
    # Legacy leftover marker list. Runtime no longer whole-sentence official
    # on these symbols; failed spans are kept.
    unmatched_symbols: Tuple[str, ...]
    integer_uses_fast_cardinal: bool
    time_re: Optional[re.Pattern[str]] = None
    dotted_number_re: Optional[re.Pattern[str]] = None
    money_minor_falls_back: bool = False  # unused; comma money is kept or fast-pathed
    # Hyphen ranges that crash official FST (e.g. pt 1-5) are reconstructed on the direct path.
    keep_bare_hyphen_leftover: bool = False


_CLAUSE_PUNCTUATION = frozenset(
    {
        ",",
        "،",
        "，",
        "、",
        ";",
        "；",
        ".",
        "。",
        "!",
        "！",
        "?",
        "؟",
        "…",
        "·",
        "‧",
        "：",
        '"',
        "'",
        "‘",
        "’",
        "“",
        "”",
        "«",
        "»",
        "(",
        ")",
        "（",
        "）",
        "[",
        "]",
        "【",
        "】",
    }
)

# Unspoken punctuation never goes through NeMo, including leftover ":" "/" and hyphen.
_PASSTHROUGH_LEFTOVER_SYMBOLS = _CLAUSE_PUNCTUATION | frozenset(
    {
        "-",
        ":",
        "/",
        "\\",
        "{",
        "}",
        "「",
        "」",
        "『",
        "』",
        "‐",
        "‑",
        "–",
        "—",
        "−",
        "*",
        "~",
        "`",
        "_",
    }
)


def _text_is_unspoken_punctuation(text: str) -> bool:
    """Bare punctuation/whitespace must not enter official or small graphs."""

    stripped = "".join(ch for ch in text if not ch.isspace())
    return bool(stripped) and all(ch in _PASSTHROUGH_LEFTOVER_SYMBOLS for ch in stripped)


_CURRENCY_AFFIXES = frozenset("$€£¥₩")
_DEGREE_AFFIXES = frozenset("°º")
_TRAILING_MINOR_RE = re.compile(r"[.,]\d+")


def attach_spoken_affixes(
    text: str,
    start: int,
    end: int,
    category: Optional[str],
) -> Tuple[int, int, str]:
    """Pull adjacent currency/degree/plus onto the span so leftover symbols cannot glue.

    If the attached token has a class reading, later replacement speaks it.
    Otherwise the whole token is kept.
    """

    if category in {"electronic", "telephone", "date", "percent", "keep"}:
        return start, end, category or "keep"
    attached = False
    if start > 0 and text[start - 1] in _CURRENCY_AFFIXES:
        start -= 1
        attached = True
    elif (
        start > 0
        and text[start - 1] == "+"
        and category in {"integer", "decimal", "dotted"}
        and (start == 1 or not text[start - 2].isdigit())
    ):
        start -= 1
        attached = True
    if end < len(text) and text[end] in _CURRENCY_AFFIXES | _DEGREE_AFFIXES:
        end += 1
        attached = True
    if start < len(text) and text[start] in _CURRENCY_AFFIXES:
        extra = _TRAILING_MINOR_RE.match(text, end)
        if extra is not None:
            end = extra.end()
            attached = True
    if attached:
        return start, end, "bound"
    return start, end, category or "keep"


_EN_MONTHS = (
    "",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_EN_MONEY_MAJOR = {
    "$": ("dollar", "dollars"),
    "€": ("euro", "euros"),
    "£": ("pound", "pounds"),
    "¥": ("yen", "yen"),
}
_EN_MONEY_MINOR = {
    "$": "cents",
    "€": "cents",
    "£": "pence",
    "¥": None,
}
_EN_MONEY_PREFIX_RE = re.compile(
    r"^([€$£¥])\s?(\d{1,3}(?:,\d{3})*|\d+)(?:[.,](\d+))?$"
)
_EN_MONEY_SUFFIX_RE = re.compile(
    r"^(\d{1,3}(?:,\d{3})*|\d+)(?:[.,](\d+))?\s?([€$£¥])$"
)
_EN_DEGREE_RE = re.compile(r"^(\d+)\s?[°º]$")
_EN_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")


def _en_plain_cardinal(number: str) -> Optional[str]:
    return NemoDirectTnEngine._fast_cardinal(number)


def _en_cardinal_value(value: int) -> str:
    from ..numbers import number_to_words

    return (
        number_to_words(str(value), language="en")
        .replace("-", " ")
        .replace(", ", " ")
    )


def _en_year(year: int) -> str:
    if 2000 <= year <= 2009:
        return _en_cardinal_value(year)
    if 2010 <= year <= 2099:
        return f"{_en_cardinal_value(year // 100)} {_en_cardinal_value(year % 100)}"
    century, yy = divmod(year, 100)
    head = _en_cardinal_value(century)
    if yy == 0:
        return f"{head} hundred"
    if yy < 10:
        return f"{head} oh {_en_cardinal_value(yy)}"
    return f"{head} {_en_cardinal_value(yy)}"


def _fast_english_iso_date(span: str) -> Optional[str]:
    match = _EN_ISO_RE.fullmatch(span)
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{_EN_MONTHS[month]} {_en_cardinal_value(day)} {_en_year(year)}"


def _en_minor_amount(fraction: str) -> Optional[int]:
    if len(fraction) == 1:
        return int(fraction) * 10
    if len(fraction) == 2:
        return int(fraction)
    return None


def _fast_english_money(span: str) -> Optional[str]:
    match = _EN_MONEY_PREFIX_RE.fullmatch(span)
    if match is not None:
        currency, integer, fraction = match.groups()
    else:
        match = _EN_MONEY_SUFFIX_RE.fullmatch(span)
        if match is None:
            return None
        integer, fraction, currency = match.groups()
    integer_words = _en_plain_cardinal(integer)
    if integer_words is None:
        return None
    singular, plural = _EN_MONEY_MAJOR[currency]
    major = singular if int(integer.replace(",", "")) == 1 else plural
    if fraction is None:
        return f"{integer_words} {major}"
    cents = _en_minor_amount(fraction)
    minor = _EN_MONEY_MINOR[currency]
    if cents is None or minor is None:
        return None
    if cents == 0:
        return f"{integer_words} {major}"
    return f"{integer_words} {major} {_en_cardinal_value(cents)} {minor}"


def _fast_english_degree(span: str) -> Optional[str]:
    match = _EN_DEGREE_RE.fullmatch(span)
    if match is None:
        return None
    words = _en_plain_cardinal(match.group(1))
    if words is None:
        return None
    unit = "degree" if int(match.group(1)) == 1 else "degrees"
    return f"{words} {unit}"


def fast_english_direct_span(category: str, span: str) -> Optional[str]:
    """Python fast path for English money, ISO dates, and degrees."""

    if category == "date":
        return _fast_english_iso_date(span)
    if category == "money":
        spoken = _fast_english_money(span)
        return spoken if spoken is not None else speak_affixed_quantity(span, "en")
    if category == "measure":
        spoken = _fast_english_degree(span)
        return spoken if spoken is not None else speak_affixed_quantity(span, "en")
    return speak_affixed_quantity(span, "en") if category in {"ordinal", "bound"} else None


_AFFIX_MONEY_UNITS = {
    "en": {
        "$": ("dollar", "dollars", "cent", "cents"),
        "€": ("euro", "euros", "cent", "cents"),
        "£": ("pound", "pounds", "penny", "pence"),
        "¥": ("yen", "yen", None, None),
        "₩": ("won", "won", None, None),
    },
    "de": {
        "$": ("dollar", "dollar", "cent", "cent"),
        "€": ("euro", "euro", "cent", "cent"),
        "£": ("pfund", "pfund", "pence", "pence"),
        "¥": ("yen", "yen", None, None),
        "₩": ("won", "won", None, None),
    },
    "es": {
        "$": ("dólar", "dólares", "centavo", "centavos"),
        "€": ("euro", "euros", "céntimo", "céntimos"),
        "£": ("libra", "libras", "penique", "peniques"),
        "¥": ("yen", "yenes", None, None),
        "₩": ("won", "wones", None, None),
    },
    "fr": {
        "$": ("dollar", "dollars", "centime", "centimes"),
        "€": ("euro", "euros", "centime", "centimes"),
        "£": ("livre", "livres", "penny", "pence"),
        "¥": ("yen", "yens", None, None),
        "₩": ("won", "wons", None, None),
    },
    "it": {
        "$": ("dollaro", "dollari", "centesimo", "centesimi"),
        "€": ("euro", "euro", "centesimo", "centesimi"),
        "£": ("sterlina", "sterline", "penny", "pence"),
        "¥": ("yen", "yen", None, None),
        "₩": ("won", "won", None, None),
    },
    "pt": {
        "$": ("dólar", "dólares", "centavo", "centavos"),
        "€": ("euro", "euros", "centavo", "centavos"),
        "£": ("libra", "libras", "penny", "pence"),
        "¥": ("iene", "ienes", None, None),
        "₩": ("won", "wons", None, None),
    },
    "sv": {
        "$": ("dollar", "dollar", "cent", "cent"),
        "€": ("euro", "euro", "cent", "cent"),
        "£": ("pund", "pund", "pence", "pence"),
        "¥": ("yen", "yen", None, None),
        "₩": ("won", "won", None, None),
    },
    "hu": {
        "$": ("dollár", "dollár", "cent", "cent"),
        "€": ("euró", "euró", "cent", "cent"),
        "£": ("font", "font", "penny", "penny"),
        "¥": ("jen", "jen", None, None),
        "₩": ("von", "von", None, None),
    },
    "vi": {
        "$": ("đô la", "đô la", "xu", "xu"),
        "€": ("euro", "euro", "xu", "xu"),
        "£": ("bảng", "bảng", "penny", "penny"),
        "¥": ("yên", "yên", None, None),
        "₩": ("won", "won", None, None),
    },
    "hy": {
        "$": ("դոլար", "դոլար", "ցենտ", "ցենտ"),
        "€": ("եվրո", "եվրո", "ցենտ", "ցենտ"),
        "£": ("ֆունտ", "ֆունտ", "պենի", "պենի"),
        "¥": ("իեն", "իեն", None, None),
        "₩": ("վոն", "վոն", None, None),
    },
    "ko": {
        "$": ("달러", "달러", "센트", "센트"),
        "€": ("유로", "유로", "센트", "센트"),
        "£": ("파운드", "파운드", "펜스", "펜스"),
        "¥": ("엔", "엔", None, None),
        "₩": ("원", "원", None, None),
    },
    "ar": {
        "$": ("دولار", "دولار", "سنت", "سنت"),
        "€": ("يورو", "يورو", "سنت", "سنت"),
        "£": ("جنيه", "جنيه", "بنس", "بنس"),
        "¥": ("ين", "ين", None, None),
        "₩": ("وون", "وون", None, None),
    },
    "hi": {
        "$": ("डॉलर", "डॉलर", "सेंट", "सेंट"),
        "€": ("यूरो", "यूरो", "सेंट", "सेंट"),
        "£": ("पाउंड", "पाउंड", "पेन्स", "पेन्स"),
        "¥": ("येन", "येन", None, None),
        "₩": ("वॉन", "वॉन", None, None),
    },
    "ja": {
        "$": ("ドル", "ドル", "セント", "セント"),
        "€": ("ユーロ", "ユーロ", "セント", "セント"),
        "£": ("ポンド", "ポンド", "ペンス", "ペンス"),
        "¥": ("円", "円", None, None),
        "₩": ("ウォン", "ウォン", None, None),
    },
    "zh": {
        "$": ("美元", "美元", "美分", "美分"),
        "€": ("欧元", "欧元", "欧分", "欧分"),
        "£": ("英镑", "英镑", "便士", "便士"),
        "¥": ("日元", "日元", None, None),
        "₩": ("韩元", "韩元", None, None),
    },
}

_AFFIX_DEGREE_UNITS = {
    "en": ("degree", "degrees"),
    "de": ("Grad", "Grad"),
    "es": ("grado", "grados"),
    "fr": ("degré", "degrés"),
    "it": ("grado", "gradi"),
    "pt": ("grau", "graus"),
    "sv": ("grad", "grader"),
    "hu": ("fok", "fok"),
    "vi": ("độ", "độ"),
    "hy": ("աստիճան", "աստիճան"),
    "ko": ("도", "도"),
    "ar": ("درجة", "درجة"),
    "hi": ("डिग्री", "डिग्री"),
    "ja": ("度", "度"),
    "zh": ("度", "度"),
}

_AFFIX_MONEY_RE = re.compile(
    r"^([$€£¥₩])\s?(\d+)(?:[.,](\d{1,2}))?$"
)
_AFFIX_MONEY_SUFFIX_RE = re.compile(
    r"^(\d+)(?:[.,](\d{1,2}))?\s?([$€£¥₩])$"
)
_AFFIX_DEGREE_RE = re.compile(r"^(\d+)\s?°$")
_AFFIX_MASC_RE = re.compile(r"^(\d+)\s?º$")
_AFFIX_PERCENT_RE = re.compile(r"^([+-]?)(\d+)(?:[.,](\d{1,2}))?\s?%$")
_AFFIX_PLUS_RE = re.compile(r"^\+(\d+)(?:[.,](\d{1,2}))?$")
_AFFIX_DECIMAL_POINT = {
    "en": "point",
    "de": "komma",
    "es": "coma",
    "fr": "virgule",
    "it": "virgola",
    "pt": "vírgula",
    "sv": "komma",
    "hu": "egész",
    "vi": "phẩy",
    "hy": "ամբողջ",
    "ko": "점",
    "ar": "فاصلة",
    "hi": "point",
    "ja": "点",
    "zh": "点",
}
_AFFIX_PERCENT_WORD = {
    "en": "percent",
    "de": "prozent",
    "es": "por ciento",
    "fr": "pour cent",
    "it": "percento",
    "pt": "por cento",
    "sv": "procent",
    "hu": "százalék",
    "vi": "phần trăm",
    "hy": "տոկոս",
    "ko": "퍼센트",
    "ar": "في المائة",
    "hi": "प्रतिशत",
}
_AFFIX_PLUS_WORD = {
    "en": "plus",
    "de": "plus",
    "es": "más",
    "fr": "plus",
    "it": "più",
    "pt": "mais",
    "sv": "plus",
    "hu": "plusz",
    "vi": "cộng",
    "hy": "գումարած",
    "ko": "플러스",
    "ar": "زائد",
    "hi": "प्लस",
    "ja": "プラス",
    "zh": "正",
}


def _affix_cardinal(language: str, value: int) -> Optional[str]:
    lang = language.split("_")[0].lower()
    if lang == "en":
        return _en_cardinal_value(value)
    if lang == "de":
        return "ein" if value == 1 else german_nemo_cardinal(value)
    if lang == "es":
        return spanish_nemo_cardinal(value)
    if lang == "fr":
        return french_nemo_cardinal(value)
    if lang == "it":
        return italian_nemo_cardinal(value)
    if lang == "pt":
        return portuguese_nemo_cardinal(value)
    if lang == "sv":
        return swedish_nemo_cardinal(value)
    if lang == "hu":
        return hungarian_nemo_cardinal(value)
    if lang == "vi":
        return vietnamese_nemo_cardinal(value)
    if lang == "hy":
        return armenian_nemo_cardinal(value)
    if lang == "ko":
        return korean_nemo_cardinal(value)
    if lang == "ar":
        return arabic_nemo_cardinal(value)
    if lang == "hi":
        return hindi_nemo_cardinal(value)
    if lang in {"ja", "zh"}:
        return cjk_nemo_cardinal(value)
    return None


def _affix_join_money(
    language: str,
    major_words: str,
    major_unit: str,
    minor_words: Optional[str],
    minor_unit: Optional[str],
) -> str:
    lang = language.split("_")[0].lower()
    if lang in {"zh", "ja"}:
        if minor_words and minor_unit:
            return f"{major_words}{major_unit}{minor_words}{minor_unit}"
        return f"{major_words}{major_unit}"
    if lang == "ar" and minor_words and minor_unit:
        return f"{major_words} {major_unit} و {minor_words} {minor_unit}"
    if minor_words and minor_unit:
        return f"{major_words} {major_unit} {minor_words} {minor_unit}"
    return f"{major_words} {major_unit}"


def _affix_join_degree(language: str, words: str, unit: str) -> str:
    lang = language.split("_")[0].lower()
    if lang in {"zh", "ja"}:
        return f"{words}{unit}"
    return f"{words} {unit}"


def _affix_digits(language: str, digits: str) -> Optional[str]:
    lang = language.split("_")[0].lower()
    parts = []
    for char in digits:
        if lang == "ar" and char == "0":
            word = "صفر"
        else:
            word = _affix_cardinal(language, int(char))
        if word is None:
            return None
        parts.append(word)
    if lang in {"zh", "ja", "ko"}:
        return "".join(parts)
    return " ".join(parts)


def _affix_number(language: str, integer: str, fraction: Optional[str]) -> Optional[str]:
    words = _affix_cardinal(language, int(integer))
    if words is None:
        return None
    if not fraction:
        return words
    frac = _affix_digits(language, fraction)
    if frac is None:
        return None
    lang = language.split("_")[0].lower()
    conn = _AFFIX_DECIMAL_POINT[lang]
    if lang in {"zh", "ja", "ko"}:
        return f"{words}{conn}{frac}"
    return f"{words} {conn} {frac}"


def _speak_affixed_percent(span: str, language: str) -> Optional[str]:
    match = _AFFIX_PERCENT_RE.fullmatch(span)
    if match is None:
        return None
    lang = language.split("_")[0].lower()
    sign, integer, fraction = match.groups()
    number = _affix_number(lang, integer, fraction)
    if number is None:
        return None
    if lang == "zh":
        body = "百分之" + number
    elif lang == "ja":
        body = "百分の" + number
    else:
        word = _AFFIX_PERCENT_WORD.get(lang)
        if word is None:
            return None
        body = f"{number} {word}"
    if sign == "-":
        minus = {
            "en": "minus",
            "de": "minus",
            "es": "menos",
            "fr": "moins",
            "it": "meno",
            "pt": "menos",
            "sv": "minus",
            "hu": "mínusz",
            "vi": "âm",
            "hy": "մինուս",
            "ko": "마이너스",
            "ar": "سالب",
            "hi": "माइनस",
            "ja": None,
            "zh": "负",
        }.get(lang, "minus")
        if lang == "zh":
            return minus + body
        if lang == "ja":
            return "-" + body
        return f"{minus} {body}"
    if sign == "+":
        plus = _AFFIX_PLUS_WORD.get(lang)
        if plus is None:
            return None
        if lang in {"zh", "ja"}:
            return plus + body
        return f"{plus} {body}"
    return body


def _speak_affixed_plus(span: str, language: str) -> Optional[str]:
    match = _AFFIX_PLUS_RE.fullmatch(span)
    if match is None:
        return None
    lang = language.split("_")[0].lower()
    integer, fraction = match.groups()
    number = _affix_number(lang, integer, fraction)
    if number is None:
        return None
    plus = _AFFIX_PLUS_WORD.get(lang)
    if plus is None:
        return None
    if lang == "zh":
        return plus + number
    if lang == "ja":
        return plus + number
    return f"{plus} {number}"


def _speak_affixed_money(span: str, language: str) -> Optional[str]:
    lang = language.split("_")[0].lower()
    units = _AFFIX_MONEY_UNITS.get(lang)
    if units is None:
        return None
    match = _AFFIX_MONEY_RE.fullmatch(span)
    if match is not None:
        currency, integer, fraction = match.groups()
    else:
        match = _AFFIX_MONEY_SUFFIX_RE.fullmatch(span)
        if match is None:
            return None
        integer, fraction, currency = match.groups()
    pack = units.get(currency)
    if pack is None:
        return None
    major_sg, major_pl, minor_sg, minor_pl = pack
    value = int(integer)
    major_words = _affix_cardinal(lang, value)
    if major_words is None:
        return None
    major_unit = major_sg if value == 1 else major_pl
    if not fraction:
        return _affix_join_money(lang, major_words, major_unit, None, None)
    if minor_sg is None:
        return None
    cents = int(fraction) * 10 if len(fraction) == 1 else int(fraction)
    if cents == 0:
        return _affix_join_money(lang, major_words, major_unit, None, None)
    minor_words = _affix_cardinal(lang, cents)
    if minor_words is None:
        return None
    minor_unit = minor_sg if cents == 1 else minor_pl
    return _affix_join_money(lang, major_words, major_unit, minor_words, minor_unit)


def _speak_affixed_degree(span: str, language: str) -> Optional[str]:
    match = _AFFIX_DEGREE_RE.fullmatch(span)
    if match is None:
        return None
    lang = language.split("_")[0].lower()
    pack = _AFFIX_DEGREE_UNITS.get(lang)
    if pack is None:
        return None
    value = int(match.group(1))
    words = _affix_cardinal(lang, value)
    if words is None:
        return None
    unit = pack[0] if value == 1 else pack[1]
    return _affix_join_degree(lang, words, unit)


def _speak_affixed_masc(span: str, language: str) -> Optional[str]:
    match = _AFFIX_MASC_RE.fullmatch(span)
    if match is None:
        return None
    lang = language.split("_")[0].lower()
    value = int(match.group(1))
    if lang == "es":
        return _ES_ORDINAL_MASC.get(value) or _speak_affixed_degree(f"{value}°", lang)
    if lang == "pt":
        return _PT_ORDINAL_MASC.get(value) or _speak_affixed_degree(f"{value}°", lang)
    return _speak_affixed_degree(f"{value}°", lang)


def speak_affixed_quantity(span: str, language: str) -> Optional[str]:
    """Speak currency, degree, percent, and leading-plus tokens. Official graphs may be wrong."""

    if not span:
        return None
    spoken = _speak_affixed_percent(span, language)
    if spoken is not None:
        return spoken
    spoken = _speak_affixed_money(span, language)
    if spoken is not None:
        return spoken
    spoken = _speak_affixed_degree(span, language)
    if spoken is not None:
        return spoken
    spoken = _speak_affixed_masc(span, language)
    if spoken is not None:
        return spoken
    return _speak_affixed_plus(span, language)


_CLOCK_SPAN_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_DOTTED_DECIMAL_SPAN_RE = re.compile(r"^(\d+)\.(\d+)$")
_COMMA_DECIMAL_SPAN_RE = re.compile(r"^(\d+),(\d{1,2})$")
_COMMA_THOUSANDS_SPAN_RE = re.compile(r"^\d{1,3}(?:,\d{3})+$")
_COMMA_DECIMAL_LANGUAGES = frozenset({"de", "es", "fr", "it", "pt", "sv", "hu", "vi"})


def speak_dotted_decimal(span: str, language: str) -> Optional[str]:
    """Speak a dotted decimal as one number. Do not leave the point or fraction behind."""

    match = _DOTTED_DECIMAL_SPAN_RE.fullmatch(span)
    if match is None:
        return None
    return _affix_number(language, match.group(1), match.group(2))


def speak_comma_decimal(span: str, language: str) -> Optional[str]:
    """Speak 12,5 as one decimal. Do not leave the comma for NumberGrammar."""

    lang = language.split("_")[0].lower()
    if lang in _COMMA_DECIMAL_LANGUAGES:
        return None
    match = _COMMA_DECIMAL_SPAN_RE.fullmatch(span)
    if match is None:
        return None
    return _affix_number(lang, match.group(1), match.group(2))


def speak_comma_thousands(span: str, language: str) -> Optional[str]:
    """Speak 1,000 as one cardinal. Do not convert 1 and leave ,000 behind."""

    lang = language.split("_")[0].lower()
    if lang in _COMMA_DECIMAL_LANGUAGES:
        return None
    if _COMMA_THOUSANDS_SPAN_RE.fullmatch(span) is None:
        return None
    return _affix_cardinal(lang, int(span.replace(",", "")))


def speak_clock_time(span: str, language: str) -> Optional[str]:
    """Speak 10:30 without a leftover colon. Keep the whole token if it is not a clock."""

    match = _CLOCK_SPAN_RE.fullmatch(span)
    if match is None:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    lang = language.split("_")[0].lower()
    if lang == "ko":
        return _fast_korean_time(span)
    if lang in {"ja", "zh"}:
        return _fast_cjk_time(span)
    if lang == "hy":
        return _fast_armenian_time(f"{hour:02d}:{minute:02d}")
    hour_words = _affix_cardinal(lang, hour)
    minute_words = _affix_cardinal(lang, minute)
    if hour_words is None or minute_words is None:
        return None
    if lang == "de":
        if minute == 0:
            return f"{hour_words} Uhr"
        return f"{hour_words} Uhr {minute_words}"
    if lang == "es":
        hour_words = _es_feminine_cardinal(hour)
        if minute == 0:
            return "la una" if hour == 1 else f"las {hour_words}"
        if minute == 30:
            return f"{hour_words} y media"
        return f"{hour_words} y {minute_words}"
    if lang == "fr":
        unit = "heure" if hour == 1 else "heures"
        if minute == 0:
            return f"{hour_words} {unit}"
        return f"{hour_words} {unit} {minute_words}"
    if lang == "it":
        if minute == 0:
            return hour_words
        if minute == 30:
            return f"{hour_words} e mezza"
        return f"{hour_words} e {minute_words}"
    if lang == "pt":
        hour_words = _pt_feminine_cardinal(hour)
        if minute == 0:
            return "uma hora" if hour == 1 else f"{hour_words} horas"
        return f"{hour_words} horas e {minute_words}"
    if lang == "sv":
        if minute == 0:
            return f"klockan {hour_words}"
        return f"klockan {hour_words} {minute_words}"
    if lang == "hu":
        if minute == 0:
            return f"{hour_words} óra"
        return f"{hour_words} óra {minute_words} perc"
    if lang == "vi":
        if minute == 0:
            return f"{hour_words} giờ"
        return f"{hour_words} giờ {minute_words} phút"
    if lang == "ar":
        if minute == 0:
            return hour_words
        return f"{hour_words} و {minute_words}"
    if lang == "hi":
        if minute == 0:
            return f"{hour_words} बजे"
        return f"{hour_words} बजकर {minute_words} मिनट"
    if minute == 0:
        return hour_words
    return f"{hour_words} {minute_words}"


_EN_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "date",
        "decimal",
        "fraction",
        "measure",
        "money",
        "ordinal",
        "telephone",
        "time",
        "electronic",
    ),
    date_re=re.compile(
        r"(?<![A-Za-z0-9])\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?![A-Za-z0-9]))"
        r"|"
        r"(?P<electronic>(?<![A-Za-z0-9])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9]))"
        r"|(?P<telephone>(?<![A-Za-z0-9])(?:\+\d{1,3}[ -])?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}(?![A-Za-z0-9]))"
        r"|(?P<money>(?<![A-Za-z0-9])(?:[$€£¥]\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d+)?|(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d+)?\s?[$€£¥])(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])(?:\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}\.\d{2}\s*[ap]\.?m\.?)(?:\s*[ap]\.?m\.?)?(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?:st|nd|rd|th)?(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])(?:[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:kg|g|mg|km|m|cm|mm|lb|lbs)|(?:\d{1,3}(?:,\d{3})+|\d+)\s?[°º])(?![A-Za-z0-9]))"
        r"|(?P<ordinal>(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+|\d+)(?:st|nd|rd|th)(?![A-Za-z0-9]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(
        r"(?i)(?<![A-Za-z])(?:mr|mrs|ms|dr|st|jr|sr|vs|etc|no)\."
    ),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=True,
)

# German official TN: comma is decimal; '.' clashes with dates/ordinals/electronic and is not thousands.
_DE_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "date",
        "cardinal",
        "decimal",
        "fraction",
        "measure",
        "money",
        "ordinal",
        "telephone",
        "time",
        "electronic",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{1,2}\.\d{1,2}\.\d{2,4}"
        r"|(?:januar|februar|m(?:ä|ae)rz|april|mai|juni|juli|august|"
        r"september|oktober|november|dezember)"
        r")(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\.\d{1,2}\.\d{2,4}|"
        r"(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])(?!\d)(?!\.)(?![A-Za-z])(?!\s(?:kg|g|mg|km|cm|mm|lb|oz|m)\b))(?![A-Za-z0-9]))"
        r"|"
        r"(?P<abbreviation>(?i:(?<![A-Za-z])(?:(?:dr|mr|mrs|ms|nr)\.|z\.b\.|d\.h\.)))"
        r"|(?P<compound>(?<![A-Za-z0-9])[A-Za-z]+-\d+(?![A-Za-z0-9]))"
        r"|"
        r"(?P<electronic>(?<![A-Za-z0-9])(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
        r"https?://[^\s]+|www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9]))"
        r"|(?P<telephone>(?<![A-Za-z0-9])(?:\+\d{1,3}\s+\d{2,}(?:[ -]\d{2,})+|"
        r"\(\d{2,}\)\s*\d{2,}(?:[ -]\d{2,})+)(?![A-Za-z0-9]))"
        r"|(?P<money>(?<![A-Za-z0-9])(?:[€£$]\s?\d+(?:,\d+)?|\d+(?:,\d+)?\s?[€£$])(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])(?:(?:\d{1,2}[:.]\d{2}(?::\d{2})?|\d{1,2})\s*[Uu]hr|\d{1,2}:\d{2})(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])(?:[+-]?\d+\.\d+\s(?:kg|g|mg|km|cm|mm|m)|[+-]?\d+(?:,\d+)?\s?(?:kg|g|mg|km|cm|mm|lb|oz|m))(?![A-Za-z0-9]))"
        r"|(?P<ordinal>(?<![A-Za-z0-9])\d+(?:\.(?!\d)|(?:ter|tes|tem|te|ten))(?![A-Za-z0-9]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])[+-]?\d+,\d+(?![A-Za-z0-9]))"
        r"|(?P<dotted>(?<![A-Za-z0-9])\d+\.\d+(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)[+-]?\d+(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(
        r"(?i)(?<![A-Za-z])(?:(?:dr|mr|mrs|ms|nr)\.|z\.b\.|d\.h\.)"
    ),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(r"(?i)\d{1,2}(?:[:.]\d{2}(?::\d{2})?)?\s*Uhr"),
    dotted_number_re=None,
    money_minor_falls_back=False,
)


_ES_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "date",
        "cardinal",
        "decimal",
        "fraction",
        "measure",
        "money",
        "ordinal",
        "telephone",
        "time",
        "electronic",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{1,2}\.\d{1,2}\.\d{2,4}"
        r"|\d{1,2}/\d{1,2}/\d{2,4}"
        r"|(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
        r"septiembre|octubre|noviembre|diciembre)"
        r")(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4})(?![A-Za-z0-9]))"
        r"|"
        r"(?P<electronic>(?<![A-Za-z0-9])(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
        r"https?://[^\s]+|www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9]))"
        r"|(?P<telephone>(?<![A-Za-z0-9])(?:\(\d{2,4}\)\s*\d{3}[ -]\d{4}|\d{3}-\d{3}-\d{4})(?![A-Za-z0-9]))"
        r"|(?P<money>(?<![A-Za-z0-9])(?:[€£$¥]\s?\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s?[€£$¥])(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:h|[A-Za-z]{2,4}))?|"
        r"\d{1,2}\.\d{1,2}(?!\d)(?:\s*(?:h|[A-Za-z]{2,4}))?|"
        r"\d{1,2}\s*h)(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?(?:kg|km|cm|mm|g|m)(?![\w]))"
        r"|(?P<ordinal>(?<![A-Za-z0-9])\d+(?:\.?º|°|ª|er|ra|do|to|mo|o)(?![A-Za-z0-9]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])[+-]?\d+,\d+(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)[+-]?(?:[1-9]\d{0,2}(?:[ .]\d{3})+|\d+)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(
        r"(?i)(?<![A-Za-z])(?:dr|dra|sr|sra|srta|prof|profa|ud|uds|d|da)\."
    ),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:h|[A-Za-z]{2,4}))?"
        r"|\d{1,2}\.\d{1,2}(?!\d)(?:\s*(?:h|[A-Za-z]{2,4}))?"
        r"|\d{1,2}\s*h"
        r")(?![A-Za-z0-9])"
    ),
    dotted_number_re=None,

    money_minor_falls_back=True,
)

_FR_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "date",
        "cardinal",
        "decimal",
        "fraction",
        "ordinal",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{1,2}/\d{1,2}/\d{2,4}"
        r"|\d{1,2}\.\d{1,2}\.\d{2,4}"
        r"|(?:janvier|f(?:é|e)vrier|mars|avril|mai|juin|juillet|"
        r"ao(?:û|u)t|septembre|octobre|novembre|d(?:é|e)cembre)"
        r")(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])(?:\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9])|\d{1,2}/\d{1,2}/\d{2,4}(?![A-Za-z0-9])|\d{1,2}\.\d{1,2}\.\d{2,4}(?![A-Za-z0-9])|"
        r"(?:0?[1-9]|[12]\d|3[01])\.(?:0[1-9]|1[0-2]|[1-9])\d*))"
        r"|"
        r"(?P<dotted>(?<![A-Za-z0-9])\d+\.\d+(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<ordinal>(?<![A-Za-z0-9])\d+(?:er|ère|ème|eme|e|°)(?![A-Za-z0-9]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])[+-]?\d+,\d+(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)[+-]?(?:[1-9]\d{0,2}(?: \d{3})+|\d+)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(
        r"(?i)(?<![A-Za-z])(?:m|mme|mlle|dr|st|n)\."
    ),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    dotted_number_re=None,
)

_IT_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "cardinal",
        "decimal",
        "measure",
        "money",
        "time",
        "electronic",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{1,2}/\d{1,2}/\d{2,4}"
        r"|(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|"
        r"agosto|settembre|ottobre|novembre|dicembre)"
        r")(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<keep>(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9]))"
        r"|"
        r"(?P<electronic>(?<![A-Za-z0-9])(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
        r"www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9]))"
        r"|(?P<money>(?<![A-Za-z0-9])(?:[€£$¥]\s?\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s?[€£$¥])(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])(?:\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}\s*h)(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?%(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?(?:kg|km|m)(?![\w]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])[+-]?\d+,\d+(?![A-Za-z0-9]))"
        r"|(?P<dotted>(?<![A-Za-z0-9])\d+\.\d+(?!\s?(?:kg|km|m)\b)(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)[+-]?(?:[1-9]\d{0,2}(?: \d{3})+|\d+)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(
        r"(?i)(?<![A-Za-z])(?:sig|dott|dr|sr|n)\."
    ),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(r"(?i)(?:\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}\s*h)"),
    dotted_number_re=None,
    money_minor_falls_back=True,
)

_PT_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "date",
        "cardinal",
        "decimal",
        "fraction",
        "measure",
        "money",
        "ordinal",
        "telephone",
        "time",
        "electronic",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{1,2}/\d{1,2}/\d{2,4}"
        r"|\d{1,2}\.\d{1,2}\.\d{2,4}"
        r"|(?:janeiro|fevereiro|mar(?:ç|c)o|abril|maio|junho|julho|"
        r"agosto|setembro|outubro|novembro|dezembro)"
        r")(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}\.\d{1,2}\.\d{2,4})(?![A-Za-z0-9]))"
        r"|"
        r"(?P<electronic>(?<![A-Za-z0-9])(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
        r"www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9]))"
        r"|(?P<money>(?<![A-Za-z0-9])(?:[€£$¥]\s?\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s?[€£$¥])(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])(?:\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}\s*h)(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?(?:kg|m)(?![\w]))"
        r"|(?P<ordinal>(?<![A-Za-z0-9])\d+[ºª°](?![A-Za-z0-9]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])[+-]?\d+,\d+(?![A-Za-z0-9]))"
        r"|(?P<dotted>(?<![A-Za-z0-9])\d+\.\d{1,2}(?!\d)(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)[+-]?(?:[1-9]\d{0,2}(?:[ .]\d{3})+|\d+)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(
        r"(?i)(?<![A-Za-z])(?:sr|sra|dr|dra|prof)\."
    ),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(r"(?i)(?:\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}\s*h)"),
    dotted_number_re=None,

    money_minor_falls_back=True,
    keep_bare_hyphen_leftover=True,
)

_SV_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "date",
        "cardinal",
        "decimal",
        "fraction",
        "measure",
        "money",
        "ordinal",
        "time",
        "electronic",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{1,2}\.\d{1,2}\.\d{2,4}"
        r"|(?:januari|februari|mars|april|maj|juni|juli|"
        r"augusti|september|oktober|november|december)"
        r")(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\.\d{1,2}\.\d{2,4}|"
        r"(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])(?!\d))(?![A-Za-z0-9]))"
        r"|"
        r"(?P<dotted>(?<![A-Za-z0-9])\d+\.\d+(?![A-Za-z0-9]))"
        r"|"
        r"(?P<electronic>(?<![A-Za-z0-9])(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
        r"www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9]))"
        r"|(?P<money>(?<![A-Za-z0-9])(?:[€$£]\s?\d+(?:,\d+)?|\d+(?:,\d+)?\s?kr)(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])(?:[Kk]l\.?|[Kk]lockan)\s+\d{1,2}(?:[:.]\d{2})?(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+\s+1/2(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?(?:kg|km|cm|mm|mg|lb|g|m|h)(?![\w]))"
        r"|(?P<ordinal>(?<![A-Za-z0-9])\d+(?:\.(?!\d)|:[ae])(?![A-Za-z0-9]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])[+-]?(?:[1-9]\d{0,2}(?: \d{3})+|\d+),\d+(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)-?(?:[1-9]\d{0,2}(?: \d{3})+|[1-9]\d*|0)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(
        r"(?i)(?<![A-Za-z])(?:nr|dr)\."
    ),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(r"(?i)(?:kl\.?|klockan)\s+\d{1,2}(?:[:.]\d{2})?"),
    dotted_number_re=None,
    money_minor_falls_back=True,
)

_HU_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "date",
        "cardinal",
        "decimal",
        "fraction",
        "measure",
        "money",
        "ordinal",
        "time",
        "electronic",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"\d{4}-\d{1,2}-\d{1,2}"
        r"|(?:január|február|március|április|május|június|július|"
        r"augusztus|szeptember|október|november|december)"
        r")(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9]))"
        r"|"
        r"(?P<electronic>(?<![A-Za-z0-9])(?:[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|"
        r"www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9]))"
        r"|(?P<money>(?<![A-Za-z0-9])(?:[$£]\s?\d+(?:,\d+)?|\d+(?:,\d+)?\s?Ft)(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])\d{2}:\d{2}(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?(?:kg|km|cm|mm|mg|g|m)(?![\w]))"
        r"|(?P<ordinal>(?<![A-Za-z0-9])\d+\.(?!\d))"
        r"|(?P<decimal>(?<![A-Za-z0-9])[+-]?(?:[1-9]\d{0,2}(?: \d{3})+|\d+),\d+(?![A-Za-z0-9]))"
        r"|(?P<dotted>(?<![A-Za-z0-9])\d+\.\d{1,2}(?!\d)(?![A-Za-z0-9]))"
        r"|(?P<keep>(?<![A-Za-z0-9])(?!1\.000)\d+\.\d{3,}(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)-?(?:[1-9]\d{0,2}(?: \d{3})+|1\.000|[1-9]\d*|0)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(
        r"(?i)(?<![A-Za-z])(?:dr|kb)\."
    ),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(r"\d{2}:\d{2}"),
    dotted_number_re=None,
    money_minor_falls_back=False,
)


_VI_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "date",
        "cardinal",
        "decimal",
        "fraction",
        "measure",
        "money",
        "time",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"\d{4}-\d{1,2}-\d{1,2}"
        r"|\d+(?: \d+)+"
        r")(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9]))"
        r"|"
        r"(?P<money>(?<![A-Za-z0-9])\d+(?:,\d+)?\s?đồng(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])\d{1,2}:\d{2}(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])[+-]?\d+(?:,\d+)?\s?(?:kg|m|h)(?![A-Za-z0-9]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])[+-]?(?:[1-9]\d{0,2}(?:\.\d{3})+|\d+),\d+(?![A-Za-z0-9]))"
        r"|(?P<dotted>(?<![A-Za-z0-9])\d+\.\d{1,2}(?!\d)(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)-?(?:[1-9]\d{0,2}(?:\.\d{3})+|[1-9]\d*|0)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(r"(?i)(?<![A-Za-z])(?:ts|ths)\."),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(r"\d{1,2}:\d{2}"),
    dotted_number_re=None,
)


_HY_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "cardinal",
        "decimal",
        "fraction",
        "money",
        "ordinal",
        "time",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<keep>(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9]))"
        r"|(?P<money>(?<![A-Za-z0-9])\d+\s?դրամ(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])\d{2}:\d{2}(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])[+-]?\d+\s?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<ordinal>(?<![A-Za-z0-9])\d+-(?:ին|րդ)(?![A-Za-z0-9]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])[+-]?\d+\.\d{1,2}(?!\d))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)-?(?:[1-9]\d{0,2}(?: \d{3})+|[1-9]\d*|0)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(r"(?i)(?<![A-Za-z])(?:դր|թ)\."),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(r"\d{2}:\d{2}"),
    dotted_number_re=None,
)


_KO_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "date",
        "cardinal",
        "decimal",
        "fraction",
        "measure",
        "money",
        "time",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:"
        r"\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{1,2}/\d{1,2}/\d{2,4}"
        r"|\d+(?: \d+)+"
        r")(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4})(?![A-Za-z0-9]))"
        r"|"
        r"(?P<money>(?<![A-Za-z0-9])\d+원(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])\d{1,2}:\d{2}(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])-?\d+(?:\.\d+)?\s?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])-?\d+(?:\.\d+)?\s?(?:kg|km|h|g)(?![A-Za-z]))"
        r"|(?P<ordinal>(?<![A-Za-z0-9])\d+번째(?![A-Za-z0-9]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])-?\d+\.\d+(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)-?(?:[1-9]\d*|0)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(r"(?i)(?<![A-Za-z])(?:dr|mr)\."),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(r"\d{1,2}:\d{2}"),
    dotted_number_re=None,
)


_AR_TN_PROFILE = _DirectTnProfile(
    graph_names=(
        "cardinal",
        "decimal",
        "fraction",
        "measure",
        "money",
    ),
    date_re=re.compile(
        r"(?i)(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9])"
    ),
    candidate_re=re.compile(
        r"(?P<keep>(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9]))"
        r"|(?P<money>(?<![A-Za-z0-9])(?:[€$]\d+|\d+\sدولار)(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])-?\d+(?:[.,]\d{1,4})?\s?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])-?\d+[.,]\d{1,4}(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)-?[1-9]\d{0,3}(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(r"(?i)(?<![A-Za-z])(?:dr|mr)\."),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=None,
    dotted_number_re=None,
)


_HI_TN_PROFILE = _DirectTnProfile(
    graph_names=("date", "cardinal", "decimal", "measure", "money"),
    date_re=re.compile(r"(?i)(?<![A-Za-z0-9])(?:\d{4}-\d{1,2}-\d{1,2}|\d+(?: \d+)+)(?![A-Za-z0-9])"),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9]))"
        r"|"
        r"(?P<money>(?<![A-Za-z0-9])[€$]\d+(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])\d+(?:\.\d+)?\s?(?:kg|g|h)(?![A-Za-z]))"
        r"|(?P<decimal>(?<![A-Za-z0-9])\d+\.\d+(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])(?<![A-Za-z]-)[+-]?(?:0|[1-9]\d*)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(r"(?i)(?<![A-Za-z])(?:dr|mr|shri)\."),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
)

_JA_TN_PROFILE = _DirectTnProfile(
    graph_names=("date", "cardinal", "decimal", "fraction", "time"),
    date_re=re.compile(r"(?i)(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9])"),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9]))"
        r"|"
        r"(?P<time>(?<![A-Za-z0-9])\d{1,2}:\d{2}(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])\d+%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<ordinal>第\d+)"
        r"|(?P<decimal>(?<![A-Za-z0-9])-?\d+\.\d+(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])[+-]?(?:0|[1-9]\d*)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(r"(?i)(?<![A-Za-z])(?:dr|mr)\."),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(r"\d{1,2}:\d{2}"),
)

_ZH_TN_PROFILE = _DirectTnProfile(
    graph_names=("date", "cardinal", "decimal", "fraction", "measure", "money", "time"),
    date_re=re.compile(r"(?i)(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9])"),
    candidate_re=re.compile(
        r"(?P<date>(?<![A-Za-z0-9])\d{4}-\d{1,2}-\d{1,2}(?![A-Za-z0-9]))"
        r"|"
        r"(?P<money>(?<![A-Za-z0-9])[€$]\d+(?![A-Za-z0-9]))"
        r"|(?P<time>(?<![A-Za-z0-9])\d{1,2}:\d{2}(?![A-Za-z0-9]))"
        r"|(?P<percent>(?<![A-Za-z0-9])\d+(?:\.\d+)?%(?![A-Za-z0-9]))"
        r"|(?P<fraction>(?<![A-Za-z0-9])\d+(?:\s+\d+)?/\d+(?![A-Za-z0-9]))"
        r"|(?P<measure>(?<![A-Za-z0-9])\d+kg(?![A-Za-z]))"
        r"|(?P<ordinal>第\d+)"
        r"|(?P<decimal>(?<![A-Za-z0-9])-?\d+\.\d+(?![A-Za-z0-9]))"
        r"|(?P<integer>(?<![A-Za-z0-9])[+-]?(?:0|[1-9]\d*)(?![A-Za-z0-9]))"
    ),
    abbreviation_re=re.compile(r"(?i)(?<![A-Za-z])(?:dr|mr)\."),
    unmatched_symbols=(),
    integer_uses_fast_cardinal=False,
    time_re=re.compile(r"\d{1,2}:\d{2}"),
)

_TN_PROFILES = {
    "en": _EN_TN_PROFILE,
    "de": _DE_TN_PROFILE,
    "es": _ES_TN_PROFILE,
    "fr": _FR_TN_PROFILE,
    "it": _IT_TN_PROFILE,
    "pt": _PT_TN_PROFILE,
    "sv": _SV_TN_PROFILE,
    "hu": _HU_TN_PROFILE,
    "vi": _VI_TN_PROFILE,
    "hy": _HY_TN_PROFILE,
    "ko": _KO_TN_PROFILE,
    "ar": _AR_TN_PROFILE,
    "hi": _HI_TN_PROFILE,
    "ja": _JA_TN_PROFILE,
    "zh": _ZH_TN_PROFILE,
}


def _compose_direct_itn_fst(tagger: Any, verbalizer: Any) -> Any:
    """The only composition point for the one-stage ITN graph; its source enters the fingerprint."""

    return (tagger.fst @ verbalizer.fst).optimize()


def direct_itn_cache_fingerprint(config: "NemoDirectItnConfig") -> str:
    parts = nemo_dependency_versions()
    parts.update(
        {
            "language": config.language,
            "direction": "direct_itn",
            "input_case": config.input_case,
            "whitelist": hash_optional_file(config.whitelist),
            "compose": hash_source(_compose_direct_itn_fst),
        }
    )
    return make_cache_fingerprint(parts)


def _write_direct_far(pynini_module: Any, cache_path: Path, graphs: dict[str, Any]) -> None:
    """Write a unique temporary FAR then replace so a crashing lock holder cannot leave a shared truncated file."""

    temporary_path = unique_nemo_cache_tmp(cache_path)
    try:
        writer = pynini_module.Far(str(temporary_path), mode="w")
        for name, graph in graphs.items():
            writer[name] = graph
        writer.close()
        temporary_path.replace(cache_path)
    except Exception:
        if temporary_path.is_file():
            temporary_path.unlink(missing_ok=True)
        raise


class NemoDirectItnEngine:
    """Compose from official ITN grammar and cache the one-stage direct FAR."""

    def __init__(self, config: NemoDirectItnConfig):
        self.config = config
        self._pynini = _load_pynini_module()
        self._graph = self._load_or_build_graph()
        self._fallback = None

    def _cache_path(self) -> Path:
        return Path(self.config.cache_dir) / (
            f"{self.config.language}_direct_itn_{self.config.input_case}.far"
        )

    def _load_or_build_graph(self) -> Any:
        def loader(cache_path: Path) -> Any:
            far = self._pynini.Far(str(cache_path), mode="r")
            return far[DEFAULT_DIRECT_ITN_RULE]

        def builder(cache_path: Path) -> Any:
            try:
                # Write the official fallback graph by fingerprint first, then load it for compose so ClassifyFst does not rebuild.
                ensure_official_fallback_cache(self.config, itn=True)
                language = self.config.language
                tagger_module = importlib.import_module(
                    "nemo_text_processing.inverse_text_normalization."
                    f"{language}.taggers.tokenize_and_classify"
                )
                verbalizer_module = importlib.import_module(
                    "nemo_text_processing.inverse_text_normalization."
                    f"{language}.verbalizers.verbalize_final"
                )
                tagger = call_with_cache_rebuild(
                    lambda overwrite: _instantiate_with_supported_kwargs(
                        tagger_module.ClassifyFst,
                        input_case=self.config.input_case,
                        cache_dir=self.config.cache_dir,
                        overwrite_cache=overwrite,
                        whitelist=self.config.whitelist,
                    ),
                    False,
                )
                verbalizer = call_with_cache_rebuild(
                    lambda overwrite: _instantiate_with_supported_kwargs(
                        verbalizer_module.VerbalizeFinalFst,
                        deterministic=True,
                        cache_dir=self.config.cache_dir,
                        overwrite_cache=overwrite,
                    ),
                    False,
                )
                # Official Python ITN parses tokens between tagger and verbalizer;
                # Compose one spoken→written graph offline so runtime needs no external FAR.
                graph = _compose_direct_itn_fst(tagger, verbalizer)
                _write_direct_far(
                    self._pynini,
                    cache_path,
                    {DEFAULT_DIRECT_ITN_RULE: graph},
                )
                return graph
            except Exception as exc:
                raise NemoFarError(f"failed to build direct ITN grammar: {exc}") from exc

        return prepare_nemo_cache_file(
            self._cache_path(),
            self.config.overwrite_cache,
            loader,
            builder,
            fingerprint=direct_itn_cache_fingerprint(self.config),
            search_dirs=self.config.cache_search_dirs,
        )

    def _fallback_normalize(self, text: str) -> str:
        if self._fallback is None:
            self._fallback = get_nemo_itn_engine(
                NemoInverseNormalizerConfig(
                    language=self.config.language,
                    input_case=self.config.input_case,
                    cache_dir=self.config.cache_dir,
                    overwrite_cache=False,
                    whitelist=self.config.whitelist,
                    cache_search_dirs=self.config.cache_search_dirs,
                )
            )
        return _fst_call_or_original(text, lambda: self._fallback.normalize(text))

    def _apply(self, graph: Any, text: str) -> Optional[str]:
        try:
            lattice = self._pynini.accep(text) @ graph
            if lattice.num_states() == 0:
                return None
            return self._pynini.shortestpath(
                lattice,
                nshortest=1,
                unique=True,
            ).string()
        except Exception:
            return None

    def normalize(self, text: str) -> str:
        """Apply this project's cached one-stage ITN graph; fall back to official ITN if there is no path."""

        _validate_text(text)
        if not text:
            return text
        return _fst_call_or_original(text, lambda: self._normalize_impl(text))

    def _normalize_impl(self, text: str) -> str:
        output = self._apply(self._graph, text)
        if output is None:
            return self._fallback_normalize(text)
        return output

    def normalize_list(
        self,
        texts: Sequence[str],
        batch_size: int = 1,
        n_jobs: int = 1,
    ) -> List[str]:
        """direct ITN currently processes lines sequentially; the parameter keeps a unified interface."""

        del n_jobs
        values = _validated_texts(texts, batch_size)
        return [self.normalize(text) for text in values]


class NemoDirectTnEngine:
    """Compose accepted NeMo class graphs into a direct relation.

    Whole-sentence official fallback is forbidden. A failed class span is kept.
    """

    def __init__(self, config: NemoDirectTnConfig):
        self.config = config
        self._profile = _TN_PROFILES[config.language]
        self._pynini = _load_pynini_module()
        self._graphs = self._load_or_build_graphs()
        self._fallback: Optional[NemoTextNormalizationEngine] = None

    def _cache_path(self) -> Path:
        return Path(self.config.cache_dir) / (
            f"{self.config.language}_direct_tn_{self.config.input_case}.far"
        )

    def _load_or_build_graphs(self) -> dict[str, Any]:
        graph_names = self._profile.graph_names

        def loader(cache_path: Path) -> dict[str, Any]:
            far = self._pynini.Far(str(cache_path), mode="r")
            return {name: far[name] for name in graph_names}

        def builder(cache_path: Path) -> dict[str, Any]:
            try:
                # Small graphs omit fallback classes such as dates; compile the official graph into the same cache by fingerprint.
                ensure_official_fallback_cache(self.config, itn=False)
                graphs = _build_direct_tn_graphs(self.config.language)
                _write_direct_far(
                    self._pynini,
                    cache_path,
                    {name: graphs[name] for name in sorted(graphs)},
                )
                return graphs
            except Exception as exc:
                raise NemoFarError(f"failed to build direct TN grammar: {exc}") from exc

        return prepare_nemo_cache_file(
            self._cache_path(),
            self.config.overwrite_cache,
            loader,
            builder,
            fingerprint=direct_tn_cache_fingerprint(self.config),
            search_dirs=self.config.cache_search_dirs,
        )

    def _fallback_normalize(self, text: str) -> str:
        # Runtime must never load official. Failed spans are kept in _normalize_impl.
        return text

    def _apply_graph(self, graph: Any, text: str) -> Optional[str]:
        try:
            lattice = self._pynini.accep(self._pynini.escape(text)) @ graph
            if lattice.num_states() == 0:
                return None
            return self._pynini.shortestpath(
                lattice,
                nshortest=1,
                unique=True,
            ).string()
        except Exception:
            return None

    @staticmethod
    def _fast_cardinal(text: str) -> Optional[str]:
        from ..numbers import digits_to_words, number_to_words

        if text.startswith("+"):
            unsigned = NemoDirectTnEngine._fast_cardinal(text[1:])
            if unsigned is None:
                return None
            return "plus " + unsigned
        unsigned = text.lstrip("-")
        digits = unsigned.replace(",", "")
        if not digits.isdigit():
            return None
        sign = "minus " if text.startswith("-") and int(digits) else ""
        if (len(digits) > 1 and digits.startswith("0")) or (
            "," not in unsigned and len(digits) >= 5
        ):
            return sign + digits_to_words(digits, "en")
        if "," not in unsigned and len(digits) == 4:
            return None
        if int(digits) > 99_999:
            return None
        return (
            number_to_words(text, language="en")
            .replace("-", " ")
            .replace(", ", " ")
        )

    def _requires_official_for_unmatched_text(
        self,
        text: str,
        matches: Sequence[re.Match],
    ) -> bool:
        # Leftover spoken symbols are kept. Whole-sentence official is too slow.
        del text, matches
        return False

    def _requires_dotted_fallback(self, text: str) -> bool:
        # Dotted leftovers stay on direct. Uncovered spans are kept, not sent to official.
        del text
        return False

    def _replacement_for_match(self, category: str, span: str) -> Optional[str]:
        try:
            return self._try_replacement_for_match(category, span)
        except Exception:
            return None

    def _fast_span_only(self, category: str, span: str) -> Optional[str]:
        """Bound tokens may use Python fast paths only. Graphs invent leftover glue."""

        language = self.config.language
        if language == "en":
            if category == "integer":
                return self._fast_cardinal(span)
            return fast_english_direct_span(category, span)
        funcs = {
            "de": fast_german_direct_span,
            "es": fast_spanish_direct_span,
            "fr": fast_french_direct_span,
            "it": fast_italian_direct_span,
            "pt": fast_portuguese_direct_span,
            "sv": fast_swedish_direct_span,
            "hu": fast_hungarian_direct_span,
            "vi": fast_vietnamese_direct_span,
            "hy": fast_armenian_direct_span,
            "ko": fast_korean_direct_span,
            "ar": fast_arabic_direct_span,
            "hi": fast_hindi_direct_span,
            "ja": fast_japanese_direct_span,
            "zh": fast_chinese_direct_span,
        }
        func = funcs.get(language)
        if func is None:
            return None
        return func(category, span)

    def _replacement_for_bound_span(self, span: str) -> str:
        """Speak an attached currency/degree/plus token, or keep the whole token."""

        spoken = speak_affixed_quantity(span, self.config.language)
        if spoken is not None:
            return spoken
        for guess in ("money", "measure", "ordinal"):
            spoken = self._fast_span_only(guess, span)
            if spoken is not None:
                return spoken
        if span.startswith("+"):
            spoken = self._fast_span_only("integer", span)
            if spoken is not None:
                return spoken
        return span

    def _try_replacement_for_match(self, category: str, span: str) -> Optional[str]:
        if category == "keep":
            return span
        if category == "bound":
            return self._replacement_for_bound_span(span)
        if self.config.language == "en":
            fast = fast_english_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "de":
            fast = fast_german_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "es":
            fast = fast_spanish_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "fr":
            fast = fast_french_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "it":
            fast = fast_italian_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "pt":
            fast = fast_portuguese_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "sv":
            fast = fast_swedish_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "hu":
            fast = fast_hungarian_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "vi":
            fast = fast_vietnamese_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "hy":
            fast = fast_armenian_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "ko":
            fast = fast_korean_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "ar":
            fast = fast_arabic_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "hi":
            fast = fast_hindi_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "ja":
            fast = fast_japanese_direct_span(category, span)
            if fast is not None:
                return fast
        if self.config.language == "zh":
            fast = fast_chinese_direct_span(category, span)
            if fast is not None:
                return fast
        if category == "integer":
            if self._profile.integer_uses_fast_cardinal:
                return self._fast_cardinal(span)
            return self._apply_graph(self._graphs["cardinal"], span)
        graph_name = "measure" if category == "percent" else category
        graph = self._graphs.get(graph_name)
        if graph is None:
            return None
        result = self._apply_graph(graph, span)
        if category == "date" and result is not None and self.config.language == "pt":
            return " ".join(result.split())
        return result

    def _maybe_lowercase(self, text: str) -> str:
        # Hindi official TN lowercases Latin letters; Devanagari is unchanged.
        if self.config.language == "hi":
            return text.lower()
        return text

    def normalize(self, text: str) -> str:
        _validate_text(text)
        if not text:
            return text
        return _fst_call_or_original(text, lambda: self._normalize_impl(text))

    def _normalize_impl(self, text: str) -> str:
        # Bare unspoken punctuation is not a TN class; do not send it to NeMo.
        if _text_is_unspoken_punctuation(text):
            return text

        matches = list(self._profile.candidate_re.finditer(text))
        pieces = []
        last_end = 0
        for match in matches:
            start, end, category = attach_spoken_affixes(
                text,
                match.start(),
                match.end(),
                match.lastgroup,
            )
            if start < last_end:
                continue
            span = text[start:end]
            replacement = self._replacement_for_match(category, span)
            if replacement is None:
                replacement = span
            pieces.append(text[last_end:start])
            pieces.append(replacement)
            last_end = end
        pieces.append(text[last_end:])
        return self._maybe_lowercase("".join(pieces))

    def normalize_list(
        self,
        texts: Sequence[str],
        batch_size: int = 1,
        n_jobs: int = 1,
    ) -> List[str]:
        del n_jobs
        values = _validated_texts(texts, batch_size)
        return [self.normalize(text) for text in values]


def _optimize_graphs(graphs: Dict[str, Any]) -> Dict[str, Any]:
    for graph in graphs.values():
        graph.optimize()
    return graphs


def _build_direct_tn_graphs(language: str) -> Dict[str, Any]:
    if language == "en":
        return _build_english_tn_graphs()
    if language == "de":
        return _build_german_tn_graphs()
    if language == "es":
        return _build_spanish_tn_graphs()
    if language == "fr":
        return _build_french_tn_graphs()
    if language == "it":
        return _build_italian_tn_graphs()
    if language == "pt":
        return _build_portuguese_tn_graphs()
    if language == "sv":
        return _build_swedish_tn_graphs()
    if language == "hu":
        return _build_hungarian_tn_graphs()
    if language == "vi":
        return _build_vietnamese_tn_graphs()
    if language == "hy":
        return _build_armenian_tn_graphs()
    if language == "ko":
        return _build_korean_tn_graphs()
    if language == "ar":
        return _build_arabic_tn_graphs()
    if language == "hi":
        return _build_hindi_tn_graphs()
    if language == "ja":
        return _build_japanese_tn_graphs()
    if language == "zh":
        return _build_chinese_tn_graphs()
    raise NemoLanguageNotSupportedError(
        f"direct-output TN does not support language {language}"
    )


def _build_english_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.en.taggers.date import DateFst
    from nemo_text_processing.text_normalization.en.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.en.taggers.electronic import ElectronicFst
    from nemo_text_processing.text_normalization.en.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.en.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.en.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.en.taggers.ordinal import OrdinalFst
    from nemo_text_processing.text_normalization.en.taggers.telephone import TelephoneFst
    from nemo_text_processing.text_normalization.en.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.en.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.en.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.en.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.en.verbalizers.electronic import ElectronicFst as VElectronicFst
    from nemo_text_processing.text_normalization.en.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.en.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.en.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst as VOrdinalFst
    from nemo_text_processing.text_normalization.en.verbalizers.telephone import TelephoneFst as VTelephoneFst
    from nemo_text_processing.text_normalization.en.verbalizers.time import TimeFst as VTimeFst

    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(cardinal=v_cardinal, deterministic=True)
    ordinal = OrdinalFst(cardinal=cardinal, deterministic=True)
    v_ordinal = VOrdinalFst(deterministic=True)
    fraction = FractionFst(cardinal=cardinal, deterministic=True)
    v_fraction = VFractionFst(deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        fraction=fraction,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        decimal=v_decimal,
        cardinal=v_cardinal,
        fraction=v_fraction,
        deterministic=True,
    )
    money = MoneyFst(
        cardinal=cardinal,
        decimal=decimal,
        deterministic=True,
    )
    v_money = VMoneyFst(decimal=v_decimal, deterministic=True)
    telephone = TelephoneFst(deterministic=True)
    v_telephone = VTelephoneFst(deterministic=True)
    time = TimeFst(cardinal=cardinal, deterministic=True)
    v_time = VTimeFst(deterministic=True)
    electronic = ElectronicFst(cardinal=cardinal, deterministic=True)
    v_electronic = VElectronicFst(deterministic=True)
    date = DateFst(cardinal=cardinal, deterministic=True)
    v_date = VDateFst(ordinal=v_ordinal, deterministic=True)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
            "ordinal": ordinal.fst @ v_ordinal.fst,
            "telephone": telephone.fst @ v_telephone.fst,
            "time": time.fst @ v_time.fst,
            "electronic": electronic.fst @ v_electronic.fst,
        }
    )


def _build_german_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.de.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.de.taggers.date import DateFst
    from nemo_text_processing.text_normalization.de.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.de.taggers.electronic import ElectronicFst
    from nemo_text_processing.text_normalization.de.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.de.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.de.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.de.taggers.ordinal import OrdinalFst
    from nemo_text_processing.text_normalization.de.taggers.telephone import TelephoneFst
    from nemo_text_processing.text_normalization.de.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.de.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.de.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.de.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.de.verbalizers.electronic import ElectronicFst as VElectronicFst
    from nemo_text_processing.text_normalization.de.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.de.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.de.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.de.verbalizers.ordinal import OrdinalFst as VOrdinalFst
    from nemo_text_processing.text_normalization.de.verbalizers.telephone import TelephoneFst as VTelephoneFst
    from nemo_text_processing.text_normalization.de.verbalizers.time import TimeFst as VTimeFst

    # German tagger/verbalizer constructors differ from English; wire them like official de grammar.
    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    ordinal = OrdinalFst(cardinal=cardinal, deterministic=True)
    v_ordinal = VOrdinalFst(deterministic=True)
    fraction = FractionFst(cardinal=cardinal, deterministic=True)
    v_fraction = VFractionFst(ordinal=v_ordinal, deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        fraction=fraction,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        decimal=v_decimal,
        cardinal=v_cardinal,
        fraction=v_fraction,
        deterministic=True,
    )
    money = MoneyFst(
        cardinal=cardinal,
        decimal=decimal,
        deterministic=True,
    )
    v_money = VMoneyFst(decimal=v_decimal, deterministic=True)
    telephone = TelephoneFst(cardinal=cardinal, deterministic=True)
    v_telephone = VTelephoneFst(deterministic=True)
    time = TimeFst(deterministic=True)
    v_time = VTimeFst(cardinal_tagger=cardinal, deterministic=True)
    electronic = ElectronicFst(deterministic=True)
    v_electronic = VElectronicFst(deterministic=True)
    date = DateFst(cardinal=cardinal, deterministic=True)
    v_date = VDateFst(ordinal=v_ordinal)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
            "ordinal": ordinal.fst @ v_ordinal.fst,
            "telephone": telephone.fst @ v_telephone.fst,
            "time": time.fst @ v_time.fst,
            "electronic": electronic.fst @ v_electronic.fst,
        }
    )


def _build_spanish_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.es.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.es.taggers.date import DateFst
    from nemo_text_processing.text_normalization.es.taggers.decimals import DecimalFst
    from nemo_text_processing.text_normalization.es.taggers.electronic import ElectronicFst
    from nemo_text_processing.text_normalization.es.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.es.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.es.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.es.taggers.ordinal import OrdinalFst
    from nemo_text_processing.text_normalization.es.taggers.telephone import TelephoneFst
    from nemo_text_processing.text_normalization.es.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.es.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.es.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.es.verbalizers.decimals import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.es.verbalizers.electronic import ElectronicFst as VElectronicFst
    from nemo_text_processing.text_normalization.es.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.es.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.es.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.es.verbalizers.ordinal import OrdinalFst as VOrdinalFst
    from nemo_text_processing.text_normalization.es.verbalizers.telephone import TelephoneFst as VTelephoneFst
    from nemo_text_processing.text_normalization.es.verbalizers.time import TimeFst as VTimeFst

    # Spanish tagger signatures for decimals/fraction/time differ from en/de; wire official es grammar.
    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    ordinal = OrdinalFst(cardinal=cardinal, deterministic=True)
    v_ordinal = VOrdinalFst(deterministic=True)
    fraction = FractionFst(
        cardinal=cardinal,
        ordinal=ordinal,
        deterministic=True,
    )
    v_fraction = VFractionFst(deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        fraction=fraction,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        decimal=v_decimal,
        cardinal=v_cardinal,
        fraction=v_fraction,
        deterministic=True,
    )
    money = MoneyFst(
        cardinal=cardinal,
        decimal=decimal,
        deterministic=True,
    )
    v_money = VMoneyFst(decimal=v_decimal, deterministic=True)
    telephone = TelephoneFst(deterministic=True)
    v_telephone = VTelephoneFst(deterministic=True)
    time = TimeFst(cardinal, deterministic=True)
    v_time = VTimeFst(deterministic=True)
    electronic = ElectronicFst(deterministic=True)
    v_electronic = VElectronicFst(deterministic=True)
    date = DateFst(cardinal=cardinal, deterministic=True)
    v_date = VDateFst(deterministic=True)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
            "ordinal": ordinal.fst @ v_ordinal.fst,
            "telephone": telephone.fst @ v_telephone.fst,
            "time": time.fst @ v_time.fst,
            "electronic": electronic.fst @ v_electronic.fst,
        }
    )


def _build_french_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.fr.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.fr.taggers.date import DateFst
    from nemo_text_processing.text_normalization.fr.taggers.decimals import DecimalFst
    from nemo_text_processing.text_normalization.fr.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.fr.taggers.ordinal import OrdinalFst
    from nemo_text_processing.text_normalization.fr.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.fr.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.fr.verbalizers.decimals import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.fr.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.fr.verbalizers.ordinal import OrdinalFst as VOrdinalFst

    # French has no measure/money/time/electronic; wire official fr grammar.
    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    ordinal = OrdinalFst(cardinal=cardinal, deterministic=True)
    v_ordinal = VOrdinalFst(deterministic=True)
    fraction = FractionFst(
        cardinal=cardinal,
        ordinal=ordinal,
        deterministic=True,
    )
    v_fraction = VFractionFst(ordinal=v_ordinal, deterministic=True)
    date = DateFst(cardinal, deterministic=True)
    v_date = VDateFst(deterministic=True)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "ordinal": ordinal.fst @ v_ordinal.fst,
        }
    )


def _build_italian_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.it.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.it.taggers.decimals import DecimalFst
    from nemo_text_processing.text_normalization.it.taggers.electronic import ElectronicFst
    from nemo_text_processing.text_normalization.it.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.it.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.it.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.it.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.it.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.it.verbalizers.electronic import ElectronicFst as VElectronicFst
    from nemo_text_processing.text_normalization.it.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.it.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.it.verbalizers.time import TimeFst as VTimeFst

    # Italian has no fraction/ordinal/telephone; the time verbalizer needs the cardinal tagger.
    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        decimal=v_decimal,
        cardinal=v_cardinal,
        deterministic=True,
    )
    money = MoneyFst(
        cardinal=cardinal,
        decimal=decimal,
        deterministic=True,
    )
    v_money = VMoneyFst(decimal=v_decimal, deterministic=True)
    time = TimeFst(deterministic=True)
    v_time = VTimeFst(cardinal_tagger=cardinal, deterministic=True)
    electronic = ElectronicFst(deterministic=True)
    v_electronic = VElectronicFst(deterministic=True)
    return _optimize_graphs(
        {
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
            "time": time.fst @ v_time.fst,
            "electronic": electronic.fst @ v_electronic.fst,
        }
    )


def _build_portuguese_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.pt.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.pt.taggers.date import DateFst
    from nemo_text_processing.text_normalization.pt.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.pt.taggers.electronic import ElectronicFst
    from nemo_text_processing.text_normalization.pt.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.pt.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.pt.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.pt.taggers.ordinal import OrdinalFst
    from nemo_text_processing.text_normalization.pt.taggers.telephone import TelephoneFst
    from nemo_text_processing.text_normalization.pt.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.pt.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.pt.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.pt.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.pt.verbalizers.electronic import ElectronicFst as VElectronicFst
    from nemo_text_processing.text_normalization.pt.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.pt.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.pt.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.pt.verbalizers.ordinal import OrdinalFst as VOrdinalFst
    from nemo_text_processing.text_normalization.pt.verbalizers.telephone import TelephoneFst as VTelephoneFst
    from nemo_text_processing.text_normalization.pt.verbalizers.time import TimeFst as VTimeFst

    # Portuguese taggers mostly use positional cardinal/ordinal; wire official pt grammar.
    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    ordinal = OrdinalFst(cardinal, deterministic=True)
    v_ordinal = VOrdinalFst(deterministic=True)
    decimal = DecimalFst(cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    fraction = FractionFst(cardinal, ordinal, deterministic=True)
    v_fraction = VFractionFst(deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        fraction=fraction,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        decimal=v_decimal,
        cardinal=v_cardinal,
        fraction=v_fraction,
        deterministic=True,
    )
    money = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=True)
    v_money = VMoneyFst(decimal=v_decimal, deterministic=True)
    telephone = TelephoneFst(deterministic=True)
    v_telephone = VTelephoneFst(deterministic=True)
    time = TimeFst(cardinal, deterministic=True)
    v_time = VTimeFst(deterministic=True)
    electronic = ElectronicFst(deterministic=True)
    v_electronic = VElectronicFst(deterministic=True)
    date = DateFst(cardinal=cardinal, deterministic=True)
    v_date = VDateFst(deterministic=True)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
            "ordinal": ordinal.fst @ v_ordinal.fst,
            "telephone": telephone.fst @ v_telephone.fst,
            "time": time.fst @ v_time.fst,
            "electronic": electronic.fst @ v_electronic.fst,
        }
    )


def _build_swedish_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.sv.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.sv.taggers.date import DateFst
    from nemo_text_processing.text_normalization.sv.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.sv.taggers.electronic import ElectronicFst
    from nemo_text_processing.text_normalization.sv.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.sv.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.sv.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.sv.taggers.ordinal import OrdinalFst
    from nemo_text_processing.text_normalization.sv.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.sv.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.sv.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.sv.verbalizers.decimals import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.sv.verbalizers.electronic import ElectronicFst as VElectronicFst
    from nemo_text_processing.text_normalization.sv.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.sv.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.sv.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.sv.verbalizers.ordinal import OrdinalFst as VOrdinalFst
    from nemo_text_processing.text_normalization.sv.verbalizers.time import TimeFst as VTimeFst

    # Swedish tagger follows official sv grammar: fraction needs ordinal, time needs cardinal.
    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    ordinal = OrdinalFst(cardinal=cardinal, deterministic=True)
    v_ordinal = VOrdinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    fraction = FractionFst(cardinal=cardinal, ordinal=ordinal, deterministic=True)
    v_fraction = VFractionFst(deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        fraction=fraction,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        decimal=v_decimal,
        cardinal=v_cardinal,
        fraction=v_fraction,
        deterministic=True,
    )
    money = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=True)
    v_money = VMoneyFst(decimal=v_decimal, deterministic=True)
    time = TimeFst(cardinal=cardinal, deterministic=True)
    v_time = VTimeFst(deterministic=True)
    electronic = ElectronicFst(deterministic=True)
    v_electronic = VElectronicFst(deterministic=True)
    date = DateFst(cardinal=cardinal, ordinal=ordinal, deterministic=True)
    v_date = VDateFst(deterministic=True)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
            "ordinal": ordinal.fst @ v_ordinal.fst,
            "time": time.fst @ v_time.fst,
            "electronic": electronic.fst @ v_electronic.fst,
        }
    )


def _build_hungarian_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.hu.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.hu.taggers.date import DateFst
    from nemo_text_processing.text_normalization.hu.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.hu.taggers.electronic import ElectronicFst
    from nemo_text_processing.text_normalization.hu.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.hu.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.hu.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.hu.taggers.ordinal import OrdinalFst
    from nemo_text_processing.text_normalization.hu.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.hu.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.hu.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.hu.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.hu.verbalizers.electronic import ElectronicFst as VElectronicFst
    from nemo_text_processing.text_normalization.hu.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.hu.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.hu.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.hu.verbalizers.ordinal import OrdinalFst as VOrdinalFst
    from nemo_text_processing.text_normalization.hu.verbalizers.time import TimeFst as VTimeFst

    # Hungarian follows official hu grammar: fraction needs ordinal, time needs cardinal.
    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    ordinal = OrdinalFst(cardinal=cardinal, deterministic=True)
    v_ordinal = VOrdinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    fraction = FractionFst(cardinal=cardinal, ordinal=ordinal, deterministic=True)
    v_fraction = VFractionFst(deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        fraction=fraction,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        decimal=v_decimal,
        cardinal=v_cardinal,
        fraction=v_fraction,
        deterministic=True,
    )
    money = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=True)
    v_money = VMoneyFst(decimal=v_decimal, deterministic=True)
    time = TimeFst(cardinal, deterministic=True)
    v_time = VTimeFst(deterministic=True)
    electronic = ElectronicFst(deterministic=True)
    v_electronic = VElectronicFst(deterministic=True)
    date = DateFst(cardinal=cardinal, deterministic=True)
    v_date = VDateFst(deterministic=True)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
            "ordinal": ordinal.fst @ v_ordinal.fst,
            "time": time.fst @ v_time.fst,
            "electronic": electronic.fst @ v_electronic.fst,
        }
    )



def _build_vietnamese_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.vi.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.vi.taggers.date import DateFst
    from nemo_text_processing.text_normalization.vi.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.vi.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.vi.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.vi.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.vi.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.vi.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.vi.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.vi.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.vi.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.vi.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.vi.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.vi.verbalizers.time import TimeFst as VTimeFst

    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(cardinal=v_cardinal, deterministic=True)
    fraction = FractionFst(cardinal=cardinal, deterministic=True)
    v_fraction = VFractionFst(deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        fraction=fraction,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        decimal=v_decimal,
        cardinal=v_cardinal,
        fraction=v_fraction,
        deterministic=True,
    )
    money = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=True)
    v_money = VMoneyFst(deterministic=True)
    time = TimeFst(cardinal=cardinal, deterministic=True)
    v_time = VTimeFst(deterministic=True)
    date = DateFst(cardinal=cardinal, deterministic=True)
    v_date = VDateFst(deterministic=True)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
            "time": time.fst @ v_time.fst,
        }
    )



def _build_armenian_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.hy.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.hy.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.hy.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.hy.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.hy.taggers.ordinal import OrdinalFst
    from nemo_text_processing.text_normalization.hy.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.hy.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.hy.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.hy.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.hy.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.hy.verbalizers.ordinal import OrdinalFst as VOrdinalFst
    from nemo_text_processing.text_normalization.hy.verbalizers.time import TimeFst as VTimeFst

    # Most Armenian official taggers lack deterministic; wire hy grammar as-is.
    cardinal = CardinalFst()
    v_cardinal = VCardinalFst(deterministic=True)
    ordinal = OrdinalFst(cardinal)
    v_ordinal = VOrdinalFst(deterministic=True)
    decimal = DecimalFst(cardinal)
    v_decimal = VDecimalFst(deterministic=True)
    fraction = FractionFst(cardinal=cardinal, ordinal=ordinal)
    v_fraction = VFractionFst()
    money = MoneyFst(cardinal=cardinal, decimal=decimal)
    v_money = VMoneyFst(deterministic=True)
    time = TimeFst()
    v_time = VTimeFst()
    return _optimize_graphs(
        {
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "money": money.fst @ v_money.fst,
            "ordinal": ordinal.fst @ v_ordinal.fst,
            "time": time.fst @ v_time.fst,
        }
    )



def _build_korean_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.ko.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.ko.taggers.date import DateFst
    from nemo_text_processing.text_normalization.ko.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.ko.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.ko.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.ko.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.ko.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.ko.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.ko.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.ko.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.ko.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.ko.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.ko.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.ko.verbalizers.time import TimeFst as VTimeFst

    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    fraction = FractionFst(cardinal=cardinal, deterministic=True)
    v_fraction = VFractionFst(deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        fraction=fraction,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        decimal=v_decimal,
        cardinal=v_cardinal,
        fraction=v_fraction,
        deterministic=True,
    )
    money = MoneyFst(cardinal=cardinal, deterministic=True)
    v_money = VMoneyFst(deterministic=True)
    time = TimeFst(cardinal=cardinal, deterministic=True)
    v_time = VTimeFst(deterministic=True)
    date = DateFst(cardinal=cardinal, deterministic=True)
    v_date = VDateFst(deterministic=True)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
            "time": time.fst @ v_time.fst,
        }
    )



def _build_arabic_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.ar.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.ar.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.ar.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.ar.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.ar.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.ar.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.ar.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.ar.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.ar.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.ar.verbalizers.money import MoneyFst as VMoneyFst

    # Arabic CardinalFst has no deterministic argument; wire the official 1.2.0 signature.
    cardinal = CardinalFst()
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    fraction = FractionFst(cardinal)
    v_fraction = VFractionFst()
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        fraction=fraction,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        decimal=v_decimal,
        cardinal=v_cardinal,
        fraction=v_fraction,
        deterministic=True,
    )
    money = MoneyFst(cardinal=cardinal, deterministic=True)
    v_money = VMoneyFst(deterministic=True)
    return _optimize_graphs(
        {
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
        }
    )



def _build_hindi_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.hi.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.hi.taggers.date import DateFst
    from nemo_text_processing.text_normalization.hi.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.hi.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.hi.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.hi.taggers.ordinal import OrdinalFst
    from nemo_text_processing.text_normalization.hi.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.hi.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.hi.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.hi.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.hi.verbalizers.money import MoneyFst as VMoneyFst

    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    ordinal = OrdinalFst(cardinal=cardinal, deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        ordinal=ordinal,
        input_case="cased",
    )
    v_measure = VMeasureFst(cardinal=v_cardinal, decimal=v_decimal)
    money = MoneyFst(cardinal=cardinal)
    v_money = VMoneyFst()
    date = DateFst(cardinal=cardinal)
    v_date = VDateFst()
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
        }
    )


def _build_japanese_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.ja.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.ja.taggers.date import DateFst
    from nemo_text_processing.text_normalization.ja.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.ja.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.ja.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.ja.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.ja.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.ja.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.ja.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.ja.verbalizers.time import TimeFst as VTimeFst

    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    fraction = FractionFst(cardinal=cardinal, deterministic=True)
    v_fraction = VFractionFst(deterministic=True)
    time = TimeFst(cardinal=cardinal, deterministic=True)
    v_time = VTimeFst(deterministic=True)
    date = DateFst(cardinal=cardinal, deterministic=True)
    v_date = VDateFst(deterministic=True)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "time": time.fst @ v_time.fst,
        }
    )


def _build_chinese_tn_graphs() -> Dict[str, Any]:
    from nemo_text_processing.text_normalization.zh.taggers.cardinal import CardinalFst
    from nemo_text_processing.text_normalization.zh.taggers.date import DateFst
    from nemo_text_processing.text_normalization.zh.taggers.decimal import DecimalFst
    from nemo_text_processing.text_normalization.zh.taggers.fraction import FractionFst
    from nemo_text_processing.text_normalization.zh.taggers.measure import MeasureFst
    from nemo_text_processing.text_normalization.zh.taggers.money import MoneyFst
    from nemo_text_processing.text_normalization.zh.taggers.time import TimeFst
    from nemo_text_processing.text_normalization.zh.verbalizers.cardinal import CardinalFst as VCardinalFst
    from nemo_text_processing.text_normalization.zh.verbalizers.date import DateFst as VDateFst
    from nemo_text_processing.text_normalization.zh.verbalizers.decimal import DecimalFst as VDecimalFst
    from nemo_text_processing.text_normalization.zh.verbalizers.fraction import FractionFst as VFractionFst
    from nemo_text_processing.text_normalization.zh.verbalizers.measure import MeasureFst as VMeasureFst
    from nemo_text_processing.text_normalization.zh.verbalizers.money import MoneyFst as VMoneyFst
    from nemo_text_processing.text_normalization.zh.verbalizers.time import TimeFst as VTimeFst

    cardinal = CardinalFst(deterministic=True)
    v_cardinal = VCardinalFst(deterministic=True)
    decimal = DecimalFst(cardinal=cardinal, deterministic=True)
    v_decimal = VDecimalFst(deterministic=True)
    fraction = FractionFst(cardinal=cardinal, deterministic=True)
    v_fraction = VFractionFst(decimal=v_decimal, deterministic=True)
    measure = MeasureFst(
        cardinal=cardinal,
        decimal=decimal,
        fraction=fraction,
        deterministic=True,
    )
    v_measure = VMeasureFst(
        cardinal=v_cardinal,
        decimal=v_decimal,
        fraction=v_fraction,
        deterministic=True,
    )
    money = MoneyFst(cardinal=cardinal, deterministic=True)
    v_money = VMoneyFst(decimal=v_decimal, deterministic=True)
    time = TimeFst(deterministic=True)
    v_time = VTimeFst(deterministic=True)
    date = DateFst(deterministic=True)
    v_date = VDateFst(deterministic=True)
    return _optimize_graphs(
        {
            "date": date.fst @ v_date.fst,
            "cardinal": cardinal.fst @ v_cardinal.fst,
            "decimal": decimal.fst @ v_decimal.fst,
            "fraction": fraction.fst @ v_fraction.fst,
            "measure": measure.fst @ v_measure.fst,
            "money": money.fst @ v_money.fst,
            "time": time.fst @ v_time.fst,
        }
    )


_TN_GRAPH_BUILDERS = {
    "en": _build_english_tn_graphs,
    "de": _build_german_tn_graphs,
    "es": _build_spanish_tn_graphs,
    "fr": _build_french_tn_graphs,
    "it": _build_italian_tn_graphs,
    "pt": _build_portuguese_tn_graphs,
    "sv": _build_swedish_tn_graphs,
    "hu": _build_hungarian_tn_graphs,
    "vi": _build_vietnamese_tn_graphs,
    "hy": _build_armenian_tn_graphs,
    "ko": _build_korean_tn_graphs,
    "ar": _build_arabic_tn_graphs,
    "hi": _build_hindi_tn_graphs,
    "ja": _build_japanese_tn_graphs,
    "zh": _build_chinese_tn_graphs,
}


def direct_tn_cache_fingerprint(config: NemoDirectTnConfig) -> str:
    """Fingerprint only inputs that change that language's FAR; routing-regex changes do not rebuild the graph."""

    builder = _TN_GRAPH_BUILDERS[config.language]
    parts = nemo_dependency_versions()
    parts.update(
        {
            "language": config.language,
            "direction": "direct_tn",
            "input_case": config.input_case,
            "graph_names": ",".join(_TN_PROFILES[config.language].graph_names),
            "builder": hash_source(builder),
            "optimize": hash_source(_optimize_graphs),
        }
    )
    return make_cache_fingerprint(parts)


@lru_cache(maxsize=32)
def get_nemo_direct_itn_engine(config: NemoDirectItnConfig) -> NemoDirectItnEngine:
    """Reuse the one-stage ITN graph by language, case, and cache dir."""

    return NemoDirectItnEngine(config)


@lru_cache(maxsize=16)
def get_nemo_direct_tn_engine(config: NemoDirectTnConfig) -> NemoDirectTnEngine:
    """Reuse direct TN graphs by language, case, and cache dir."""

    return NemoDirectTnEngine(config)


def clear_nemo_direct_engine_cache() -> None:
    """Clear the private direct engine cache."""

    get_nemo_direct_itn_engine.cache_clear()
    get_nemo_direct_tn_engine.cache_clear()


def initialize_direct_nemo_engine(
    language: str,
    input_case: Optional[str],
    cache_dir: Optional[str],
    overwrite_cache: bool,
    whitelist: Optional[str],
    itn: bool,
    cache_search_dirs: Optional[Sequence[str]] = None,
) -> Any:
    """Initialize the private direct backend for accepted languages; direction comes from itn, FAR written to cache_dir."""

    if cache_search_dirs is None:
        layout = resolve_nemo_cache_layout(cache_dir)
        effective_cache_dir = layout.write_dir
        search_dirs = layout.read_dirs
    else:
        effective_cache_dir = cache_dir or DEFAULT_NEMO_CACHE_DIR
        search_dirs = tuple(cache_search_dirs)
    if itn:
        return get_nemo_direct_itn_engine(
            NemoDirectItnConfig(
                language=language,
                input_case=input_case or "lower_cased",
                cache_dir=effective_cache_dir,
                overwrite_cache=overwrite_cache,
                whitelist=whitelist,
                cache_search_dirs=search_dirs,
            )
        )
    return get_nemo_direct_tn_engine(
        NemoDirectTnConfig(
            language=language,
            input_case=input_case or "cased",
            cache_dir=effective_cache_dir,
            overwrite_cache=overwrite_cache,
            whitelist=whitelist,
            cache_search_dirs=search_dirs,
        )
    )
