# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""French cardinal grammar。"""

from ..protocol import NumberToken


class FrenchNumberGrammar:
    language = "fr"
    minus_word = "moins"
    decimal_word = "virgule"
    digit_words = (
        "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept",
        "huit", "neuf",
    )
    _small = (
        "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept",
        "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze",
        "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf",
    )
    _tens = {
        20: "vingt",
        30: "trente",
        40: "quarante",
        50: "cinquante",
        60: "soixante",
    }
    _large_scales = (
        (10**18, "trillion", "trillions"),
        (10**15, "billiard", "billiards"),
        (10**12, "billion", "billions"),
        (10**9, "milliard", "milliards"),
        (10**6, "million", "millions"),
    )
    oracle_random_upper_bound = 10**12

    def _under_hundred(self, value: int) -> str:
        if value < 20:
            return self._small[value]
        if value < 70:
            tens, units = divmod(value, 10)
            tens_word = self._tens[tens * 10]
            if units == 0:
                return tens_word
            connector = " et " if units == 1 else "-"
            return tens_word + connector + self._small[units]
        if value < 80:
            remainder = value - 60
            connector = " et " if remainder == 11 else "-"
            return "soixante" + connector + self._small[remainder]
        remainder = value - 80
        if remainder == 0:
            return "quatre-vingts"
        return "quatre-vingt-" + self._under_hundred(remainder)

    def _under_thousand(self, value: int) -> str:
        if value < 100:
            return self._under_hundred(value)
        hundreds, remainder = divmod(value, 100)
        if hundreds == 1:
            prefix = "cent"
        else:
            prefix = self._small[hundreds] + " cent"
            if not remainder:
                prefix += "s"
        if remainder:
            prefix += " " + self._under_hundred(remainder)
        return prefix

    @staticmethod
    def _before_thousand(text: str) -> str:
        if text.endswith("cents") or text.endswith("vingts"):
            return text[:-1]
        return text

    def _under_million(self, value: int) -> str:
        if value < 1000:
            return self._under_thousand(value)
        thousands, remainder = divmod(value, 1000)
        prefix = (
            "mille"
            if thousands == 1
            else self._before_thousand(self._integer_to_words(thousands))
            + " mille"
        )
        if remainder:
            prefix += " " + self._under_thousand(remainder)
        return prefix

    def _integer_to_words(self, value: int) -> str:
        if value < 1_000_000:
            return self._under_million(value)
        for scale, singular, plural in self._large_scales:
            if value < scale:
                continue
            coefficient, remainder = divmod(value, scale)
            unit = singular if coefficient == 1 else plural
            result = f"{self._integer_to_words(coefficient)} {unit}"
            if remainder:
                result += " " + self._integer_to_words(remainder)
            return result
        raise ValueError(f"fr cannot split number {value}")

    def cardinal(self, token: NumberToken) -> str:
        result = self._integer_to_words(token.integer_value)
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
