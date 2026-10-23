# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Shared three-digit skeleton for default Russian/Ukrainian cardinals."""

from decimal import Decimal

from ..protocol import NumberToken


class EastSlavicNumberGrammar:
    language = ""
    zero_word = ""
    minus_word = ""
    digit_words = ()
    feminine_digit_words = ()
    teens = ()
    tens = ()
    hundreds = ()
    scales = ()
    max_value = 10**12
    oracle_random_upper_bound = 10**12

    @staticmethod
    def _plural_form(value: int) -> int:
        if 11 <= value % 100 <= 19:
            return 2
        if value % 10 == 1:
            return 0
        if value % 10 in (2, 3, 4):
            return 1
        return 2

    def _under_thousand(
        self,
        value: int,
        group_index: int,
        feminine: bool,
    ) -> list[str]:
        ones, tens, hundreds = value % 10, value // 10 % 10, value // 100
        words = []
        if hundreds:
            words.append(self.hundreds[hundreds])
        if tens > 1:
            words.append(self.tens[tens])
        if tens == 1:
            words.append(self.teens[ones])
        elif ones:
            use_feminine = group_index == 1 or (group_index == 0 and feminine)
            table = self.feminine_digit_words if use_feminine else self.digit_words
            words.append(table[ones])
        return words

    def _integer_to_words(self, value: int, feminine: bool = False) -> str:
        if value < 0:
            return f"{self.minus_word} {self._integer_to_words(-value, feminine)}"
        if value == 0:
            return self.zero_word
        if value >= self.max_value:
            raise OverflowError(f"{self.language} number is too large")
        chunks = []
        remaining = value
        while remaining:
            chunks.append(remaining % 1000)
            remaining //= 1000
        words = []
        for group_index in range(len(chunks) - 1, -1, -1):
            chunk = chunks[group_index]
            if not chunk:
                continue
            words.extend(self._under_thousand(chunk, group_index, feminine))
            if group_index:
                words.append(
                    self.scales[group_index - 1][self._plural_form(chunk)]
                )
        return " ".join(words)

    @staticmethod
    def _provider_number_text(token: NumberToken) -> str:
        """Replay bundled num2words Decimal conversion that ran before the language converter."""

        return str(Decimal(token.source))

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return " ".join(self.digit_words[int(digit)] for digit in digits)


class RussianNumberGrammar(EastSlavicNumberGrammar):
    language = "ru"
    zero_word = "ноль"
    minus_word = "минус"
    digit_words = (
        zero_word, "один", "два", "три", "четыре", "пять", "шесть",
        "семь", "восемь", "девять",
    )
    feminine_digit_words = (
        zero_word, "одна", "две", "три", "четыре", "пять", "шесть",
        "семь", "восемь", "девять",
    )
    teens = (
        "десять", "одиннадцать", "двенадцать", "тринадцать",
        "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать",
        "восемнадцать", "девятнадцать",
    )
    tens = (
        "", "", "двадцать", "тридцать", "сорок", "пятьдесят",
        "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
    )
    hundreds = (
        "", "сто", "двести", "триста", "четыреста", "пятьсот",
        "шестьсот", "семьсот", "восемьсот", "девятьсот",
    )
    scales = (
        ("тысяча", "тысячи", "тысяч"),
        ("миллион", "миллиона", "миллионов"),
        ("миллиард", "миллиарда", "миллиардов"),
    )
    decimal_units = (
        ("десятая", "десятых"),
        ("сотая", "сотых"),
        ("тысячная", "тысячных"),
        ("десятитысячная", "десятитысячных"),
        ("стотысячная", "стотысячных"),
        ("миллионная", "миллионных"),
        ("десятимиллионная", "десятимиллионных"),
        ("стомиллионная", "стомиллионных"),
        ("миллиардная", "миллиардных"),
        ("десятимиллиардная", "десятимиллиардных"),
        ("стомиллиардная", "стомиллиардных"),
        ("триллионная", "триллионных"),
    )

    def cardinal(self, token: NumberToken) -> str:
        number_text = self._provider_number_text(token).replace(",", ".")
        if "." not in number_text:
            return self._integer_to_words(int(number_text))
        left, right = number_text.split(".")
        left_value = int(left)
        fraction_value = int(right)
        point_word = "целая" if self._plural_form(left_value) == 0 else "целых"
        singular_denominator = right[-1] == "1" and right[-2:] != "11"
        denominator = self.decimal_units[len(right) - 1][
            0 if singular_denominator else 1
        ]
        result = " ".join((
            self._integer_to_words(left_value, feminine=True),
            point_word,
            self._integer_to_words(fraction_value, feminine=True),
            denominator,
        ))
        if token.negative and left_value == 0 and not token.is_zero:
            return f"{self.minus_word} {result}"
        return result

    def inflect_integer(self, value: int, case: str = "nom", gender: str = "m") -> str:
        """Contextual cardinals. Callers should pass masculine nominative when there is no context."""

        from .ru_cases import inflect_integer

        return inflect_integer(value, case, gender)

    def inflect_ordinal(self, value: int, case: str = "nom", gender: str = "m") -> str:
        """Contextual ordinals. Dates use neuter nominative; years use masculine."""

        from .ru_cases import inflect_ordinal

        return inflect_ordinal(value, case, gender)

    def year_to_words(self, value: int, case: str = "nom") -> str:
        from .ru_cases import year_to_words

        return year_to_words(value, case)


class UkrainianNumberGrammar(EastSlavicNumberGrammar):
    language = "uk"
    zero_word = "нуль"
    minus_word = "мінус"
    digit_words = (
        zero_word, "один", "два", "три", "чотири", "п'ять", "шість",
        "сім", "вісім", "дев'ять",
    )
    feminine_digit_words = (
        zero_word, "одна", "дві", "три", "чотири", "п'ять", "шість",
        "сім", "вісім", "дев'ять",
    )
    teens = (
        "десять", "одинадцять", "дванадцять", "тринадцять",
        "чотирнадцять", "п'ятнадцять", "шістнадцять", "сімнадцять",
        "вісімнадцять", "дев'ятнадцять",
    )
    tens = (
        "", "", "двадцять", "тридцять", "сорок", "п'ятдесят",
        "шістдесят", "сімдесят", "вісімдесят", "дев'яносто",
    )
    hundreds = (
        "", "сто", "двісті", "триста", "чотириста", "п'ятсот",
        "шістсот", "сімсот", "вісімсот", "дев'ятсот",
    )
    scales = (
        ("тисяча", "тисячі", "тисяч"),
        ("мільйон", "мільйони", "мільйонів"),
        ("мільярд", "мільярди", "мільярдів"),
    )

    def cardinal(self, token: NumberToken) -> str:
        number_text = self._provider_number_text(token).replace(",", ".")
        if "." not in number_text:
            return self._integer_to_words(int(number_text))
        left, right = number_text.split(".")
        left_value = int(left)
        leading_zeros = len(right) - len(right.lstrip("0"))
        fraction_words = [self.zero_word] * leading_zeros
        fraction_words.append(self._integer_to_words(int(right)))
        result = " ".join((
            self._integer_to_words(left_value),
            "кома",
            " ".join(fraction_words),
        ))
        if token.negative and left_value == 0 and not token.is_zero:
            return f"{self.minus_word} {result}"
        return result
