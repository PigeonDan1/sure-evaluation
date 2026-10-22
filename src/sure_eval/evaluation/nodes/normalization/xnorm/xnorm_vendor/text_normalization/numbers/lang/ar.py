# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Arabic provider-compatible default cardinal grammar。"""

from ..protocol import NumberToken


ONES = (
    "", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة",
    "ثمانية", "تسعة", "عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر",
    "أربعة عشر", "خمسة عشر", "ستة عشر", "سبعة عشر", "ثمانية عشر",
    "تسعة عشر",
)
FEMININE_ONES = (
    "", "إحدى", "اثنتان", "ثلاث", "أربع", "خمس", "ست", "سبع", "ثمان",
    "تسع", "عشر", "إحدى عشرة", "اثنتا عشرة", "ثلاث عشرة", "أربع عشرة",
    "خمس عشرة", "ست عشرة", "سبع عشرة", "ثماني عشرة", "تسع عشرة",
)
TENS = ("عشرون", "ثلاثون", "أربعون", "خمسون", "ستون", "سبعون", "ثمانون", "تسعون")
HUNDREDS = (
    "", "مائة", "مئتان", "ثلاثمائة", "أربعمائة", "خمسمائة", "ستمائة",
    "سبعمائة", "ثمانمائة", "تسعمائة",
)
APPENDED_TWOS = (
    "مئتا", "ألفا", "مليونا", "مليارا", "تريليونا", "كوادريليونا",
    "كوينتليونا", "سكستيليونا", "سبتيليونا", "أوكتيليونا ",
    "نونيليونا", "ديسيليونا", "أندسيليونا", "دوديسيليونا",
    "تريديسيليونا", "كوادريسيليونا", "كوينتينيليونا",
)
TWOS = (
    "مئتان", "ألفان", "مليونان", "ملياران", "تريليونان", "كوادريليونان",
    "كوينتليونان", "سكستيليونان", "سبتيليونان", "أوكتيليونان ",
    "نونيليونان ", "ديسيليونان", "أندسيليونان", "دوديسيليونان",
    "تريديسيليونان", "كوادريسيليونان", "كوينتينيليونان",
)
GROUPS = (
    "مائة", "ألف", "مليون", "مليار", "تريليون", "كوادريليون",
    "كوينتليون", "سكستيليون", "سبتيليون", "أوكتيليون", "نونيليون",
    "ديسيليون", "أندسيليون", "دوديسيليون", "تريديسيليون",
    "كوادريسيليون", "كوينتينيليون",
)
APPENDED_GROUPS = (
    "", "ألفاً", "مليوناً", "ملياراً", "تريليوناً", "كوادريليوناً",
    "كوينتليوناً", "سكستيليوناً", "سبتيليوناً", "أوكتيليوناً",
    "نونيليوناً", "ديسيليوناً", "أندسيليوناً", "دوديسيليوناً",
    "تريديسيليوناً", "كوادريسيليوناً", "كوينتينيليوناً",
)
PLURAL_GROUPS = (
    "", "آلاف", "ملايين", "مليارات", "تريليونات", "كوادريليونات",
    "كوينتليونات", "سكستيليونات", "سبتيليونات", "أوكتيليونات",
    "نونيليونات", "ديسيليونات", "أندسيليونات", "دوديسيليونات",
    "تريديسيليونات", "كوادريسيليونات", "كوينتينيليونات",
)


class ArabicNumberGrammar:
    language = "ar"
    minus_word = "سالب"
    digit_words = ("صفر", *ONES[1:10])
    max_value = 10**51
    oracle_random_upper_bound = 10**12

    def _digit_word(self, digit: int, group_level: int) -> str:
        return FEMININE_ONES[digit] if group_level == -1 else ONES[digit]

    def _group_words(
        self,
        group_number: int,
        group_level: int,
        integer_value: int,
    ) -> str:
        last_two = group_number % 100
        hundreds = group_number // 100
        result = ""
        if hundreds:
            if last_two == 0 and hundreds == 2:
                result = APPENDED_TWOS[0]
            else:
                result = HUNDREDS[hundreds]
                if last_two:
                    result += " و "

        if 0 < last_two < 20:
            if last_two == 2 and not hundreds and group_level > 0:
                magnitude = len(str(integer_value)) - 1
                if integer_value == 2 * 10**magnitude and magnitude % 3 == 0:
                    result = APPENDED_TWOS[group_level]
                else:
                    result = TWOS[group_level]
            elif last_two == 1 and group_level > 0:
                result += GROUPS[group_level]
            else:
                result += self._digit_word(last_two, group_level)
        elif last_two >= 20:
            ones = last_two % 10
            if ones:
                result += self._digit_word(ones, group_level) + " و "
            result += TENS[last_two // 10 - 2]
        return result

    def _integer_to_words(self, value: int) -> str:
        lower_text = ""
        remaining = value
        group_level = 0
        while remaining > 0:
            group_number = remaining % 1000
            remaining //= 1000
            group_text = self._group_words(
                group_number,
                group_level,
                value,
            )
            if group_text:
                if group_level > 0:
                    if lower_text:
                        lower_text = "و " + lower_text
                    if group_number not in (1, 2):
                        if group_number % 100 != 1:
                            if 3 <= group_number <= 10:
                                scale = PLURAL_GROUPS[group_level]
                            elif lower_text:
                                scale = APPENDED_GROUPS[group_level]
                            else:
                                scale = GROUPS[group_level]
                        else:
                            scale = GROUPS[group_level]
                        lower_text = f"{scale} {lower_text}"
                lower_text = f"{group_text} {lower_text}"
            group_level += 1
        return lower_text.strip()

    def cardinal(self, token: NumberToken) -> str:
        if token.integer_value >= self.max_value:
            raise OverflowError("Arabic number is too large")
        fraction = (token.fraction_digits + "00")[:2]
        decimal_value = int(fraction) if token.fraction_digits else 0
        if token.integer_value:
            result = self._integer_to_words(token.integer_value)
        elif token.is_zero:
            result = "صفر"
        else:
            result = ""
        if decimal_value:
            decimal_text = self._group_words(decimal_value, -1, token.integer_value)
            result = (result + "  " if result else "") + ", " + decimal_text
        if token.negative and not token.is_zero:
            return f"{self.minus_word} {result}"
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return " ".join(self.digit_words[int(digit)] for digit in digits)
