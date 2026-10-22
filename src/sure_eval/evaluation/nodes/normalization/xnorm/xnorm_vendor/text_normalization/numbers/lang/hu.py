# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Hungarian cardinal grammar。"""

from ..protocol import NumberToken


class HungarianNumberGrammar:
    language = "hu"
    minus_word = "mínusz"
    decimal_word = "egész"
    digit_words = (
        "nulla", "egy", "kettő", "három", "négy", "öt", "hat", "hét",
        "nyolc", "kilenc",
    )
    _tens = {
        3: "harminc", 4: "negyven", 5: "ötven", 6: "hatvan", 7: "hetven",
        8: "nyolcvan", 9: "kilencven",
    }
    _scales = {
        1: "tized",
        2: "század",
        3: "ezred",
        6: "millió",
        9: "milliárd",
        12: "billió",
        15: "billiárd",
        18: "trillió",
    }
    oracle_random_upper_bound = 10**12

    def _under_thirty(self, value: int) -> str:
        if value < 10:
            return self.digit_words[value]
        if value == 10:
            return "tíz"
        if value < 20:
            return "tizen" + self.digit_words[value - 10]
        if value == 20:
            return "húsz"
        return "huszon" + self.digit_words[value - 20]

    def _under_hundred(self, value: int) -> str:
        if value < 30:
            return self._under_thirty(value)
        tens, ones = divmod(value, 10)
        return self._tens[tens] + (self.digit_words[ones] if ones else "")

    def _coefficient_words(self, value: int) -> str:
        return "két" if value == 2 else self._integer_to_words(value, zero="")

    def _compound_remainder(self, value: int) -> str:
        return "két" if value == 2 else self._integer_to_words(value, zero="")

    def _under_thousand(self, value: int) -> str:
        if value < 100:
            return self._under_hundred(value)
        hundreds, remainder = divmod(value, 100)
        prefix = "száz" if hundreds == 1 else self._coefficient_words(hundreds) + "száz"
        return prefix + (self._compound_remainder(remainder) if remainder else "")

    def _under_million(self, value: int) -> str:
        if value < 1000:
            return self._under_thousand(value)
        thousands, remainder = divmod(value, 1000)
        prefix = "ezer" if thousands == 1 else self._coefficient_words(thousands) + "ezer"
        if not remainder:
            return prefix
        separator = "" if value <= 2000 else "-"
        return prefix + separator + self._compound_remainder(remainder)

    def _integer_to_words(self, value: int, zero: str = "nulla") -> str:
        if value == 0:
            return zero
        if value < 1_000_000:
            return self._under_million(value)
        digit_count = len(str(value))
        adjusted_digits = digit_count if digit_count % 3 else digit_count - 2
        exponent = adjusted_digits // 3 * 3
        scale = 10**exponent
        coefficient, remainder = divmod(value, scale)
        result = self._coefficient_words(coefficient) + self._scales[exponent]
        if remainder:
            result += "-" + self._compound_remainder(remainder)
        return result

    def cardinal(self, token: NumberToken) -> str:
        result = self._integer_to_words(token.integer_value)
        if token.fraction_digits:
            fraction_value = int(token.fraction_digits)
            suffix = self._scales[len(token.fraction_digits)]
            result += (
                f" {self.decimal_word} "
                + self._integer_to_words(fraction_value)
                + " "
                + suffix
            )
        if token.negative and not token.is_zero:
            return f"{self.minus_word} {result}"
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return " ".join(self.digit_words[int(digit)] for digit in digits)
