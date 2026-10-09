# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Spanish cardinal grammar。"""

from ..protocol import NumberToken


class SpanishNumberGrammar:
    language = "es"
    minus_word = "menos"
    decimal_word = "punto"
    digit_words = (
        "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete",
        "ocho", "nueve",
    )
    _small = (
        "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete",
        "ocho", "nueve", "diez", "once", "doce", "trece", "catorce",
        "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve",
        "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro",
        "veinticinco", "veintiséis", "veintisiete", "veintiocho",
        "veintinueve",
    )
    _tens = {
        30: "treinta",
        40: "cuarenta",
        50: "cincuenta",
        60: "sesenta",
        70: "setenta",
        80: "ochenta",
        90: "noventa",
    }
    _hundreds = {
        2: "doscientos",
        3: "trescientos",
        4: "cuatrocientos",
        5: "quinientos",
        6: "seiscientos",
        7: "setecientos",
        8: "ochocientos",
        9: "novecientos",
    }
    _large_scales = (
        (10**24, "cuatrillón", "cuatrillones"),
        (10**18, "trillón", "trillones"),
        (10**12, "billón", "billones"),
        (10**6, "millón", "millones"),
    )
    oracle_random_upper_bound = 10**12

    def _under_hundred(self, value: int) -> str:
        if value < 30:
            return self._small[value]
        tens, units = divmod(value, 10)
        result = self._tens[tens * 10]
        if units:
            result += " y " + self._small[units]
        return result

    def _under_thousand(self, value: int) -> str:
        if value < 100:
            return self._under_hundred(value)
        hundreds, remainder = divmod(value, 100)
        if hundreds == 1:
            result = "cien" if not remainder else "ciento"
        else:
            result = self._hundreds[hundreds]
        if remainder:
            result += " " + self._under_hundred(remainder)
        return result

    def _under_million(self, value: int) -> str:
        if value < 1000:
            return self._under_thousand(value)
        thousands, remainder = divmod(value, 1000)
        result = (
            "mil"
            if thousands == 1
            else self._integer_to_words(thousands) + " mil"
        )
        if remainder:
            result += " " + self._under_thousand(remainder)
        return result

    def _integer_to_words(self, value: int) -> str:
        if value < 1_000_000:
            return self._under_million(value)
        for scale, singular, plural in self._large_scales:
            if value < scale:
                continue
            coefficient, remainder = divmod(value, scale)
            if coefficient == 1:
                result = f"un {singular}"
            else:
                result = f"{self._integer_to_words(coefficient)} {plural}"
            if remainder:
                result += " " + self._integer_to_words(remainder)
            return result
        raise ValueError(f"es cannot split number {value}")

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
