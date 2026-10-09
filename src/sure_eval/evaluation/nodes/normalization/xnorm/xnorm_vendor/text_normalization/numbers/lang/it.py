# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Italian cardinal grammar keeping the current provider's spelling and grouping."""

from ..protocol import NumberToken


CARDINAL_WORDS = (
    "zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette",
    "otto", "nove", "dieci", "undici", "dodici", "tredici",
    "quattordici", "quindici", "sedici", "diciassette", "diciotto",
    "diciannove",
)
TENS = {2: "venti", 3: "trenta", 4: "quaranta", 6: "sessanta"}
EXPONENT_PREFIXES = (
    "zero", "m", "b", "tr", "quadr", "quint", "sest", "sett", "ott",
    "nov", "dec",
)


def _phonetic_contraction(text: str) -> str:
    return (
        text.replace("oo", "o")
        .replace("ao", "o")
        .replace("io", "o")
        .replace("au", "u")
        .replace("iu", "u")
    )


def _accentuate(text: str) -> str:
    words = []
    for word in text.split():
        if word.endswith("tre") and len(word) > 3:
            words.append(word.replace("tré", "tre")[:-3] + "tré")
        else:
            words.append(word.replace("tré", "tre"))
    return " ".join(words)


def _exponent_word(exponent_length: int) -> str:
    prefix = EXPONENT_PREFIXES[exponent_length // 6]
    return prefix + ("ilione" if exponent_length % 6 == 0 else "iliardo")


class ItalianNumberGrammar:
    language = "it"
    minus_word = "meno"
    decimal_word = "virgola"
    digit_words = CARDINAL_WORDS[:10]
    oracle_random_upper_bound = 10**12

    def _under_hundred(self, value: int) -> str:
        if value < 20:
            return CARDINAL_WORDS[value]
        tens, units = divmod(value, 10)
        prefix = TENS.get(tens, CARDINAL_WORDS[tens][:-1] + "anta")
        postfix = "" if units == 0 else CARDINAL_WORDS[units]
        return _phonetic_contraction(prefix + postfix)

    def _under_thousand(self, value: int) -> str:
        if value < 100:
            return self._under_hundred(value)
        hundreds, remainder = divmod(value, 100)
        prefix = "cento" if hundreds == 1 else CARDINAL_WORDS[hundreds] + "cento"
        postfix = "" if remainder == 0 else self._cardinal_integer(remainder)
        return _phonetic_contraction(prefix + postfix)

    def _under_million(self, value: int) -> str:
        if value < 1000:
            return self._under_thousand(value)
        thousands, remainder = divmod(value, 1000)
        prefix = (
            "mille"
            if thousands == 1
            else self._cardinal_integer(thousands) + "mila"
        )
        postfix = "" if remainder == 0 else self._cardinal_integer(remainder)
        return prefix + postfix

    def _big_number(self, value: int) -> str:
        digits = str(value)
        if len(digits) >= 66:
            raise NotImplementedError("The given number is too large.")
        leading_length = len(digits) % 3 or 3
        multiplier_digits = digits[:leading_length]
        remainder_digits = digits[leading_length:]
        unit = _exponent_word(len(remainder_digits))
        if multiplier_digits == "1":
            prefix = "un "
        else:
            prefix = self._cardinal_integer(int(multiplier_digits))
            unit = " " + unit[:-1] + "i"
        if set(remainder_digits) != {"0"}:
            remainder = self._cardinal_integer(int(remainder_digits))
            unit += ", " if " e " in remainder else " e "
        else:
            remainder = ""
        return prefix + unit + remainder

    def _integer_to_words(self, value: int) -> str:
        if value < 1_000_000:
            return self._under_million(value)
        return self._big_number(value)

    def _cardinal_integer(self, value: int) -> str:
        return _accentuate(self._integer_to_words(value))

    def cardinal(self, token: NumberToken) -> str:
        result = self._cardinal_integer(token.integer_value)
        fraction = token.significant_fraction_digits
        if fraction:
            result += f" {self.decimal_word} " + self.digit_sequence(fraction)
        if token.negative and not token.is_zero:
            return f"{self.minus_word} {result}"
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return " ".join(self.digit_words[int(digit)] for digit in digits)
