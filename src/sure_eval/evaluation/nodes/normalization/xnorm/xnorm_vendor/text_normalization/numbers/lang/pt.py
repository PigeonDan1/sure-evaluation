# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""European Portuguese cardinal grammar。"""

import re

from ..protocol import NumberToken


class PortugueseNumberGrammar:
    language = "pt"
    minus_word = "menos"
    decimal_word = "vírgula"
    digit_words = (
        "zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete",
        "oito", "nove",
    )
    _small = (
        "zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete",
        "oito", "nove", "dez", "onze", "doze", "treze", "catorze",
        "quinze", "dezasseis", "dezassete", "dezoito", "dezanove", "vinte",
    )
    _tens = {
        20: "vinte",
        30: "trinta",
        40: "quarenta",
        50: "cinquenta",
        60: "sessenta",
        70: "setenta",
        80: "oitenta",
        90: "noventa",
    }
    _hundreds = {
        2: "duzentos",
        3: "trezentos",
        4: "quatrocentos",
        5: "quinhentos",
        6: "seiscentos",
        7: "setecentos",
        8: "oitocentos",
        9: "novecentos",
    }
    _large_scales = (
        (10**24, "quatrilião", "quatriliões"),
        (10**18, "trilião", "triliões"),
        (10**12, "bilião", "biliões"),
        (10**6, "milhão", "milhões"),
    )
    oracle_random_upper_bound = 10**12

    def _under_hundred(self, value: int) -> str:
        if value <= 20:
            return self._small[value]
        tens, units = divmod(value, 10)
        result = self._tens[tens * 10]
        if units:
            result += " e " + self._small[units]
        return result

    def _under_thousand(self, value: int) -> str:
        if value < 100:
            return self._under_hundred(value)
        hundreds, remainder = divmod(value, 100)
        if hundreds == 1:
            result = "cem" if not remainder else "cento"
        else:
            result = self._hundreds[hundreds]
        if remainder:
            result += " e " + self._under_hundred(remainder)
        return result

    @staticmethod
    def _scale_connector() -> str:
        return " e "

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
            result += self._scale_connector()
            result += self._under_thousand(remainder)
        return result

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
                result += self._scale_connector()
                result += self._integer_to_words(remainder)
            return result
        raise ValueError(f"pt cannot split number {value}")

    @staticmethod
    def _legacy_scale_postprocess(text: str) -> str:
        for scale_text in (
            "mil", "milhão", "milhões", "mil milhões",
            "bilião", "biliões", "mil biliões",
        ):
            if re.match(rf".*{scale_text} e \w*entos? (?=.*e)", text):
                text = text.replace(f"{scale_text} e", scale_text)
        return text

    def cardinal(self, token: NumberToken) -> str:
        result = self._legacy_scale_postprocess(
            self._integer_to_words(token.integer_value)
        )
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
