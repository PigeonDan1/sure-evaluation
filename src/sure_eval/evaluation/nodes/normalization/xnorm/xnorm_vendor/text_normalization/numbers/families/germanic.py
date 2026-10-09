# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Shared compound-scale grammar for Danish/Norwegian/Swedish/Dutch."""

from ..protocol import NumberToken


class GermanicCompoundGrammar:
    language = ""
    minus_word = ""
    decimal_word = "komma"
    small_words = ()
    tens_words = {}
    reverse_tens = False
    reverse_one_word = ""
    hundred_word = ""
    one_hundred_word = ""
    hundred_coefficient_separator = ""
    hundred_remainder_separator = ""
    thousand_word = ""
    one_thousand_word = ""
    thousand_coefficient_separator = ""
    large_scales = ()
    large_one_word = ""
    oracle_random_upper_bound = 10**12

    @property
    def digit_words(self):
        return self.small_words[:10]

    def _reverse_connector(self, unit_word: str) -> str:
        return ""

    def _under_hundred(self, value: int) -> str:
        if value <= 20:
            return self.small_words[value]
        tens, units = divmod(value, 10)
        tens_word = self.tens_words[tens * 10]
        if not units:
            return tens_word
        unit_word = (
            self.reverse_one_word
            if self.reverse_tens and units == 1
            else self.small_words[units]
        )
        if self.reverse_tens:
            return unit_word + self._reverse_connector(unit_word) + tens_word
        return tens_word + unit_word

    def _under_thousand(self, value: int) -> str:
        if value < 100:
            return self._under_hundred(value)
        hundreds, remainder = divmod(value, 100)
        if hundreds == 1:
            result = self.one_hundred_word
        else:
            result = (
                self.small_words[hundreds]
                + self.hundred_coefficient_separator
                + self.hundred_word
            )
        if remainder:
            result += self.hundred_remainder_separator
            result += self._under_hundred(remainder)
        return result

    def _thousand_remainder_separator(self, thousands: int, remainder: int) -> str:
        return ""

    def _under_million(self, value: int) -> str:
        if value < 1000:
            return self._under_thousand(value)
        thousands, remainder = divmod(value, 1000)
        if thousands == 1:
            result = self.one_thousand_word
        else:
            result = (
                self._integer_to_words(thousands)
                + self.thousand_coefficient_separator
                + self.thousand_word
            )
        if remainder:
            result += self._thousand_remainder_separator(thousands, remainder)
            result += self._under_thousand(remainder)
        return result

    def _large_remainder_separator(self, remainder: int) -> str:
        return " "

    def _integer_to_words(self, value: int) -> str:
        if value < 1_000_000:
            return self._under_million(value)
        for scale, singular, plural in self.large_scales:
            if value < scale:
                continue
            coefficient, remainder = divmod(value, scale)
            unit = singular if coefficient == 1 else plural
            coefficient_word = (
                self.large_one_word
                if coefficient == 1
                else self._integer_to_words(coefficient)
            )
            result = coefficient_word + " " + unit
            if remainder:
                result += self._large_remainder_separator(remainder)
                result += self._integer_to_words(remainder)
            return result
        raise ValueError(f"{self.language} cannot split number {value}")

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


class DanishNumberGrammar(GermanicCompoundGrammar):
    language = "da"
    minus_word = "minus"
    small_words = (
        "nul", "et", "to", "tre", "fire", "fem", "seks", "syv", "otte",
        "ni", "ti", "elleve", "tolv", "tretten", "fjorten", "femten",
        "seksten", "sytten", "atten", "nitten", "tyve",
    )
    tens_words = {
        20: "tyve",
        30: "tredive",
        40: "fyrre",
        50: "halvtreds",
        60: "treds",
        70: "halvfjerds",
        80: "firs",
        90: "halvfems",
    }
    reverse_tens = True
    reverse_one_word = "en"
    hundred_word = "hundrede"
    one_hundred_word = "ethundrede"
    hundred_remainder_separator = " og "
    thousand_word = "tusind"
    one_thousand_word = "ettusind"
    large_one_word = "en"
    large_scales = (
        (10**18, "trillioner", "trillioner"),
        (10**15, "billiarder", "billiarder"),
        (10**12, "billioner", "billioner"),
        (10**9, "milliarder", "milliarder"),
        (10**6, "millioner", "millioner"),
    )

    def _reverse_connector(self, unit_word):
        return "og"

    def _thousand_remainder_separator(self, thousands, remainder):
        return "e og " if thousands * 1000 <= 100_000 else ""


class NorwegianNumberGrammar(GermanicCompoundGrammar):
    language = "no"
    minus_word = "minus"
    small_words = (
        "null", "en", "to", "tre", "fire", "fem", "seks", "syv", "åtte",
        "ni", "ti", "elleve", "tolv", "tretten", "fjorten", "femten",
        "seksten", "sytten", "atten", "nitten", "tjue",
    )
    tens_words = {
        20: "tjue",
        30: "tretti",
        40: "førti",
        50: "femti",
        60: "seksti",
        70: "sytti",
        80: "åtti",
        90: "nitti",
    }
    hundred_word = "hundre"
    one_hundred_word = "en hundre"
    hundred_coefficient_separator = " "
    hundred_remainder_separator = " og "
    thousand_word = "tusen"
    one_thousand_word = "en tusen"
    thousand_coefficient_separator = " "
    large_one_word = "en"
    large_scales = (
        (10**18, "trillion", "trillion"),
        (10**15, "trilliard", "trilliard"),
        (10**12, "billion", "billion"),
        (10**9, "millard", "millard"),
        (10**6, "million", "million"),
    )

    def _thousand_remainder_separator(self, thousands, remainder):
        return " og " if remainder < 100 else " "

    def _large_remainder_separator(self, remainder):
        return " og " if remainder < 100 else " "


class SwedishNumberGrammar(GermanicCompoundGrammar):
    language = "sv"
    minus_word = "minus"
    small_words = (
        "noll", "ett", "två", "tre", "fyra", "fem", "sex",
        "sju", "åtta", "nio", "tio", "elva", "tolv", "tretton",
    )
    small_words += (
        "fjorton", "femton", "sexton", "sjutton", "arton", "nitton", "tjugo",
    )
    tens_words = {
        20: "tjugo",
        30: "trettio",
        40: "förtio",
        50: "femtio",
        60: "sextio",
        70: "sjuttio",
        80: "åttio",
        90: "nittio",
    }
    hundred_word = "hundra"
    one_hundred_word = "etthundra"
    thousand_word = "tusen"
    one_thousand_word = "etttusen"
    large_one_word = "en"
    large_scales = (
        (10**18, "triljon", "triljoner"),
        (10**15, "triljard", "triljarder"),
        (10**12, "biljon", "biljoner"),
        (10**9, "miljard", "miljarder"),
        (10**6, "miljon", "miljoner"),
    )

    def _thousand_remainder_separator(self, thousands, remainder):
        return "" if remainder < 100 else " "

    def _large_remainder_separator(self, remainder):
        return "" if remainder < 100 else " "


class DutchNumberGrammar(GermanicCompoundGrammar):
    language = "nl"
    minus_word = "min"
    small_words = (
        "nul", "één", "twee", "drie", "vier", "vijf", "zes",
        "zeven", "acht", "negen", "tien", "elf", "twaalf", "dertien",
    )
    small_words += (
        "veertien", "vijftien", "zestien", "zeventien", "achttien",
        "negentien", "twintig",
    )
    tens_words = {
        20: "twintig",
        30: "dertig",
        40: "veertig",
        50: "vijftig",
        60: "zestig",
        70: "zeventig",
        80: "tachtig",
        90: "negentig",
    }
    reverse_tens = True
    reverse_one_word = "een"
    hundred_word = "honderd"
    one_hundred_word = "honderd"
    thousand_word = "duizend"
    one_thousand_word = "duizend"
    large_one_word = "een"
    large_scales = (
        (10**18, "triljoen", "triljoen"),
        (10**15, "biljard", "biljard"),
        (10**12, "biljoen", "biljoen"),
        (10**9, "miljard", "miljard"),
        (10**6, "miljoen", "miljoen"),
    )

    def _reverse_connector(self, unit_word):
        return "ën" if unit_word.endswith("e") else "en"
