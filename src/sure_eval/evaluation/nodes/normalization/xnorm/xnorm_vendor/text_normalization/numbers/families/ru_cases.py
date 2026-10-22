# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Russian cardinal/ordinal inflection. Callers pass masculine nominative without context. Only forms needed for local agreement; higher thousands stay nominative, last three digits inflect."""

from __future__ import annotations

CASES = ("nom", "gen", "dat", "acc", "ins", "prep")
_CASE_INDEX = {name: index for index, name in enumerate(CASES)}

# 1/2 inflects by gender; inanimate accusative defaults to nominative.
_ONE = {
    "m": ("один", "одного", "одному", "один", "одним", "одном"),
    "f": ("одна", "одной", "одной", "одну", "одной", "одной"),
    "n": ("одно", "одного", "одному", "одно", "одним", "одном"),
}
_TWO = {
    "m": ("два", "двух", "двум", "два", "двумя", "двух"),
    "f": ("две", "двух", "двум", "две", "двумя", "двух"),
    "n": ("два", "двух", "двум", "два", "двумя", "двух"),
}
_THREE = ("три", "трёх", "трём", "три", "тремя", "трёх")
_FOUR = ("четыре", "четырёх", "четырём", "четыре", "четырьмя", "четырёх")
_FIVE_TO_TWENTY = {
    5: ("пять", "пяти", "пяти", "пять", "пятью", "пяти"),
    6: ("шесть", "шести", "шести", "шесть", "шестью", "шести"),
    7: ("семь", "семи", "семи", "семь", "семью", "семи"),
    8: ("восемь", "восьми", "восьми", "восемь", "восемью", "восьми"),
    9: ("девять", "девяти", "девяти", "девять", "девятью", "девяти"),
    10: ("десять", "десяти", "десяти", "десять", "десятью", "десяти"),
    11: ("одиннадцать", "одиннадцати", "одиннадцати", "одиннадцать", "одиннадцатью", "одиннадцати"),
    12: ("двенадцать", "двенадцати", "двенадцати", "двенадцать", "двенадцатью", "двенадцати"),
    13: ("тринадцать", "тринадцати", "тринадцати", "тринадцать", "тринадцатью", "тринадцати"),
    14: ("четырнадцать", "четырнадцати", "четырнадцати", "четырнадцать", "четырнадцатью", "четырнадцати"),
    15: ("пятнадцать", "пятнадцати", "пятнадцати", "пятнадцать", "пятнадцатью", "пятнадцати"),
    16: ("шестнадцать", "шестнадцати", "шестнадцати", "шестнадцать", "шестнадцатью", "шестнадцати"),
    17: ("семнадцать", "семнадцати", "семнадцати", "семнадцать", "семнадцатью", "семнадцати"),
    18: ("восемнадцать", "восемнадцати", "восемнадцати", "восемнадцать", "восемнадцатью", "восемнадцати"),
    19: ("девятнадцать", "девятнадцати", "девятнадцати", "девятнадцать", "девятнадцатью", "девятнадцати"),
}
_TENS = {
    2: ("двадцать", "двадцати", "двадцати", "двадцать", "двадцатью", "двадцати"),
    3: ("тридцать", "тридцати", "тридцати", "тридцать", "тридцатью", "тридцати"),
    4: ("сорок", "сорока", "сорока", "сорок", "сорока", "сорока"),
    5: ("пятьдесят", "пятидесяти", "пятидесяти", "пятьдесят", "пятьюдесятью", "пятидесяти"),
    6: ("шестьдесят", "шестидесяти", "шестидесяти", "шестьдесят", "шестьюдесятью", "шестидесяти"),
    7: ("семьдесят", "семидесяти", "семидесяти", "семьдесят", "семьюдесятью", "семидесяти"),
    8: ("восемьдесят", "восьмидесяти", "восьмидесяти", "восемьдесят", "восемьюдесятью", "восьмидесяти"),
    9: ("девяносто", "девяноста", "девяноста", "девяносто", "девяноста", "девяноста"),
}
_HUNDREDS = {
    1: ("сто", "ста", "ста", "сто", "ста", "ста"),
    2: ("двести", "двухсот", "двумстам", "двести", "двумястами", "двухстах"),
    3: ("триста", "трёхсот", "трёмстам", "триста", "тремястами", "трёхстах"),
    4: ("четыреста", "четырёхсот", "четырёмстам", "четыреста", "четырьмястами", "четырёхстах"),
    5: ("пятьсот", "пятисот", "пятистам", "пятьсот", "пятьюстами", "пятистах"),
    6: ("шестьсот", "шестисот", "шестистам", "шестьсот", "шестьюстами", "шестистах"),
    7: ("семьсот", "семисот", "семистам", "семьсот", "семьюстами", "семистах"),
    8: ("восемьсот", "восьмисот", "восьмистам", "восемьсот", "восемьюстами", "восьмистах"),
    9: ("девятьсот", "девятисот", "девятистам", "девятьсот", "девятьюстами", "девятистах"),
}
_ZERO = ("ноль", "ноля", "нолю", "ноль", "нолём", "ноле")

# Ordinal stems 1-19; третий is special-cased.
_ORDINAL_19 = {
    1: "перв",
    2: "втор",
    3: None,
    4: "четвёрт",
    5: "пят",
    6: "шест",
    7: "седьм",
    8: "восьм",
    9: "девят",
    10: "десят",
    11: "одиннадцат",
    12: "двенадцат",
    13: "тринадцат",
    14: "четырнадцат",
    15: "пятнадцат",
    16: "шестнадцат",
    17: "семнадцат",
    18: "восемнадцат",
    19: "девятнадцат",
}
_ORDINAL_TENS = {
    2: "двадцат",
    3: "тридцат",
    4: "сороков",
    5: "пятидесят",
    6: "шестидесят",
    7: "семидесят",
    8: "восьмидесят",
    9: "девяност",
}

# Strong-adjective endings: m/f/n × six cases.
_HARD_ENDINGS = {
    "m": ("ый", "ого", "ому", "ый", "ым", "ом"),
    "f": ("ая", "ой", "ой", "ую", "ой", "ой"),
    "n": ("ое", "ого", "ому", "ое", "ым", "ом"),
}
_THIRD = {
    "m": ("третий", "третьего", "третьему", "третий", "третьим", "третьем"),
    "f": ("третья", "третьей", "третьей", "третью", "третьей", "третьей"),
    "n": ("третье", "третьего", "третьему", "третье", "третьим", "третьем"),
}
_THOUSAND = {
    "nom": ("тысяча", "тысячи", "тысяч"),
    "gen": ("тысячи", "тысяч", "тысяч"),
    "dat": ("тысяче", "тысячам", "тысячам"),
    "acc": ("тысячу", "тысячи", "тысяч"),
    "ins": ("тысячей", "тысячами", "тысячами"),
    "prep": ("тысяче", "тысячах", "тысячах"),
}


def _plural_form(value: int) -> int:
    if 11 <= value % 100 <= 19:
        return 2
    if value % 10 == 1:
        return 0
    if value % 10 in (2, 3, 4):
        return 1
    return 2


def _index(case: str) -> int:
    try:
        return _CASE_INDEX[case]
    except KeyError as exc:
        raise ValueError(f"unknown Russian case: {case}") from exc


def _under_hundred(value: int, case: str, gender: str) -> str:
    slot = _index(case)
    if value == 0:
        return ""
    if value == 1:
        return _ONE[gender][slot]
    if value == 2:
        return _TWO[gender][slot]
    if value == 3:
        return _THREE[slot]
    if value == 4:
        return _FOUR[slot]
    if value < 20:
        return _FIVE_TO_TWENTY[value][slot]
    tens, ones = divmod(value, 10)
    tens_word = _TENS[tens][slot]
    if ones == 0:
        return tens_word
    return tens_word + " " + _under_hundred(ones, case, gender)


def _under_thousand(value: int, case: str, gender: str) -> str:
    slot = _index(case)
    hundreds, rest = divmod(value, 100)
    parts = []
    if hundreds:
        parts.append(_HUNDREDS[hundreds][slot])
    if rest:
        parts.append(_under_hundred(rest, case, gender))
    return " ".join(parts)


def inflect_integer(value: int, case: str = "nom", gender: str = "m") -> str:
    """Read an integer in a given case/gender. Higher thousands stay nominative; the last group inflects."""

    if gender not in ("m", "f", "n"):
        raise ValueError(f"unknown Russian gender: {gender}")
    if value < 0:
        return "минус " + inflect_integer(-value, case, gender)
    if value == 0:
        return _ZERO[_index(case)]
    if value < 1000:
        return _under_thousand(value, case, gender)
    thousands, rest = divmod(value, 1000)
    if thousands >= 1000:
        millions, thousands = divmod(thousands, 1000)
        rest = thousands * 1000 + rest
        million_words = inflect_integer(millions, "nom", "m")
        form = 0 if millions % 10 == 1 and millions % 100 != 11 else (
            1 if millions % 10 in (2, 3, 4) and not (11 <= millions % 100 <= 19) else 2
        )
        million_scale = ("миллион", "миллиона", "миллионов")[form]
        tail = inflect_integer(rest, case, gender) if rest else ""
        return " ".join(part for part in (million_words, million_scale, tail) if part)
    parts = []
    if thousands:
        if thousands == 1 and rest == 0:
            parts.append(_ONE["f"][_index(case)])
            parts.append(_THOUSAND[case][0])
        else:
            # Thousands stay nominative so «к двух тысячам» cannot run away.
            parts.append(_under_thousand(thousands, "nom", "f"))
            parts.append(_THOUSAND["nom"][_plural_form(thousands)])
    if rest:
        parts.append(_under_thousand(rest, case, gender))
    return " ".join(parts)


def _ordinal_stem(stem: str, case: str, gender: str) -> str:
    return stem + _HARD_ENDINGS[gender][_index(case)]


def inflect_ordinal(value: int, case: str = "nom", gender: str = "m") -> str:
    """Inflect ordinals 1-99; larger numbers ordinalize only the last two digits."""

    if value <= 0:
        raise ValueError(f"ordinal must be positive: {value}")
    if value >= 100:
        high, last = divmod(value, 100)
        head = _under_thousand(high * 100, "nom", "m")
        if last == 0:
            return head
        return (head + " " + inflect_ordinal(last, case, gender)).strip()
    if value == 3:
        return _THIRD[gender][_index(case)]
    if value < 20:
        return _ordinal_stem(_ORDINAL_19[value], case, gender)
    tens, ones = divmod(value, 10)
    if ones == 0:
        return _ordinal_stem(_ORDINAL_TENS[tens], case, gender)
    # Compound ordinals: tens stay nominative cardinal, ones are ordinal.
    tens_word = _TENS[tens][_index("nom")]
    return tens_word + " " + inflect_ordinal(ones, case, gender)


def year_to_words(value: int, case: str = "nom") -> str:
    """Years: thousands as cardinals, last two digits as masculine ordinals."""

    if value < 1000 or value > 9999:
        return inflect_ordinal(value, case, "m")
    thousands, rest = divmod(value, 1000)
    parts = []
    if thousands == 1:
        parts.append("тысяча")
    else:
        parts.append(inflect_integer(thousands, "nom", "f"))
        parts.append(_THOUSAND["nom"][_plural_form(thousands)])
    if rest:
        parts.append(inflect_ordinal(rest, case, "m"))
    return " ".join(parts)
