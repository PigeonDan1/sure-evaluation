# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""German cardinal grammar。"""

from ..protocol import NumberToken


class GermanNumberGrammar:
    language = "de"
    minus_word = "minus"
    decimal_word = "Komma"
    digit_words = (
        "null", "eins", "zwei", "drei", "vier", "fünf", "sechs",
        "sieben", "acht", "neun",
    )
    _small = (
        "null", "eins", "zwei", "drei", "vier", "fünf", "sechs",
        "sieben", "acht", "neun", "zehn", "elf", "zwölf", "dreizehn",
        "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn",
        "neunzehn",
    )
    _tens = {
        20: "zwanzig",
        30: "dreißig",
        40: "vierzig",
        50: "fünfzig",
        60: "sechzig",
        70: "siebzig",
        80: "achtzig",
        90: "neunzig",
    }
    _large_scales = (
        (10**18, "Trillion", "Trillionen"),
        (10**15, "Billiarde", "Billiarden"),
        (10**12, "Billion", "Billionen"),
        (10**9, "Milliarde", "Milliarden"),
        (10**6, "Million", "Millionen"),
    )
    oracle_random_upper_bound = 10**12

    def _under_hundred(self, value: int) -> str:
        if value < 20:
            return self._small[value]
        tens, units = divmod(value, 10)
        tens_word = self._tens[tens * 10]
        if not units:
            return tens_word
        unit_word = "ein" if units == 1 else self._small[units]
        return unit_word + "und" + tens_word

    def _under_thousand(self, value: int) -> str:
        if value < 100:
            return self._under_hundred(value)
        hundreds, remainder = divmod(value, 100)
        coefficient = "ein" if hundreds == 1 else self._small[hundreds]
        return coefficient + "hundert" + (
            self._under_hundred(remainder) if remainder else ""
        )

    def _under_million(self, value: int) -> str:
        if value < 1000:
            return self._under_thousand(value)
        thousands, remainder = divmod(value, 1000)
        coefficient = (
            "ein" if thousands == 1 else self._integer_to_words(thousands)
        )
        return coefficient + "tausend" + (
            self._under_thousand(remainder) if remainder else ""
        )

    def _integer_to_words(self, value: int) -> str:
        if value < 1_000_000:
            return self._under_million(value)
        for scale, singular, plural in self._large_scales:
            if value < scale:
                continue
            coefficient, remainder = divmod(value, scale)
            if coefficient == 1:
                result = f"eine {singular}"
            else:
                result = f"{self._integer_to_words(coefficient)} {plural}"
            if remainder:
                result += " " + self._integer_to_words(remainder)
            return result
        raise ValueError(f"de cannot split number {value}")

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
