# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Grammar for values recursively split into numeric cards and joined by a finite combination policy."""

from typing import Dict, Optional, Tuple

from .protocol import NumberToken


class RecursiveScaleGrammar:
    """Share sign, scale recursion, decimals, and digit sequences; languages only provide card data."""

    language = ""
    minus_word = ""
    decimal_word = ""
    digit_words: Tuple[str, ...] = ()
    number_words: Tuple[Tuple[int, str], ...] = ()
    exact_number_words: Tuple[Tuple[int, str], ...] = ()
    exact_lookup_max: Optional[int] = None
    scale_coefficient_words = {}
    omit_leading_one_scales = frozenset()
    word_separator = " "
    remainder_separator = " "
    scale_word_order = "coefficient_scale"
    sequential_merge = False
    nested_split = False
    reverse_tens_one_scales = frozenset()
    reverse_tens_one_connector = " "
    reverse_right_words = {}
    forward_tens_one_scales = frozenset()
    forward_tens_one_connector = " "
    general_addition_connector = None
    agreement_scales = frozenset()
    agreement_words = {}
    default_merge_operation = "add"
    digit_separator = " "
    fraction_min_digits = 0
    fraction_max_digits = None
    enforce_max_value = True
    oracle_random_upper_bound = None

    def __init__(self):
        ordered = sorted(self.number_words, key=lambda item: item[0], reverse=True)
        if not ordered or ordered[-1][0] != 0:
            raise ValueError(f"{self.language} number_words must include 0")
        if self.scale_word_order not in {"coefficient_scale", "scale_coefficient"}:
            raise ValueError(
                f"{self.language} scale_word_order is invalid: {self.scale_word_order}"
            )
        if self.default_merge_operation not in {"add", "multiply"}:
            raise ValueError(
                f"{self.language} default_merge_operation is invalid: "
                f"{self.default_merge_operation}"
            )
        if self.nested_split and not self.sequential_merge:
            raise ValueError(f"{self.language} nested_split requires sequential_merge")
        self._ordered_words = tuple(ordered)
        self._base_exact_words: Dict[int, str] = dict(ordered)
        self._override_exact_words: Dict[int, str] = dict(
            (int(value), word) for value, word in self.exact_number_words
        )
        self.max_value = ordered[0][0] * 1000

    def _lookup_exact_word(self, value: int) -> Optional[str]:
        override = self._override_exact_words.get(value)
        if override is not None:
            return override
        if self.exact_lookup_max is not None and value > self.exact_lookup_max:
            return None
        return self._base_exact_words.get(value)

    def _split_parts(self, value: int) -> list:
        """Keep observable numeric fragment boundaries for a few legacy merge orders."""

        exact = self._lookup_exact_word(value)
        if exact is not None:
            return [(exact, value)]
        for scale, scale_word in self._ordered_words:
            if scale == 0 or scale > value:
                continue
            coefficient, remainder = divmod(value, scale)
            special_coefficient = self.scale_coefficient_words.get(
                scale, {}
            ).get(coefficient)
            if special_coefficient is not None:
                result = [(special_coefficient, coefficient * scale)]
            elif coefficient == 1:
                if scale in self.omit_leading_one_scales:
                    result = []
                else:
                    result = self._split_parts(1)
                result.append((scale_word, scale))
            else:
                coefficient_parts = self._split_parts(coefficient)
                result = (
                    [coefficient_parts]
                    if self.nested_split
                    else coefficient_parts
                )
                result.append((scale_word, scale))
            if remainder:
                remainder_parts = self._split_parts(remainder)
                if self.nested_split:
                    result.append(remainder_parts)
                else:
                    result.extend(remainder_parts)
            return result
        raise ValueError(f"{self.language} cannot split number {value}")

    def _merge_sequential_pair(
        self,
        left: tuple[str, int],
        right: tuple[str, int],
    ) -> tuple[str, int]:
        left_text, left_value = left
        right_text, right_value = right
        if (
            left_value in self.reverse_tens_one_scales
            and 1 <= right_value <= 9
        ):
            right_text = self.reverse_right_words.get(right_value, right_text)
            text = right_text + self.reverse_tens_one_connector + left_text
            return text, left_value + right_value
        if (
            left_value in self.forward_tens_one_scales
            and 1 <= right_value <= 9
        ):
            return (
                left_text + self.forward_tens_one_connector + right_text,
                left_value + right_value,
            )
        if right_value in self.agreement_scales and left_value >= 2:
            coefficient_word = self.agreement_words.get(left_value)
            if coefficient_word is None:
                coefficient_word = self._compose_sequential_words(left_value)
            return (
                right_text + self.word_separator + coefficient_word,
                left_value * right_value,
            )
        if (
            self.general_addition_connector is not None
            and left_value > right_value
            and right_value > 0
        ):
            return (
                left_text + self.general_addition_connector + right_text,
                left_value + right_value,
            )
        if self.default_merge_operation == "multiply":
            return (
                left_text + self.word_separator + right_text,
                left_value * right_value,
            )
        else:
            text = left_text + self.word_separator + right_text
        # This value only replays the next legacy merge branch.
        return text, left_value + right_value

    def _clean_sequential_parts(
        self,
        parts: list,
    ) -> tuple[str, int]:
        working = parts
        while len(working) != 1:
            output = []
            left, right = working[:2]
            if isinstance(left, tuple) and isinstance(right, tuple):
                output.append(self._merge_sequential_pair(left, right))
                if working[2:]:
                    output.append(working[2:])
            else:
                for element in working:
                    if isinstance(element, list):
                        output.append(
                            element[0]
                            if len(element) == 1
                            else self._clean_sequential_parts(element)
                        )
                    else:
                        output.append(element)
            working = output
        return working[0]

    def _compose_sequential_words(self, value: int) -> str:
        text, _ = self._clean_sequential_parts(self._split_parts(value))
        return text

    def _compose_words(self, value: int) -> str:
        if self.sequential_merge:
            return self._compose_sequential_words(value)
        exact = self._lookup_exact_word(value)
        if exact is not None:
            return exact
        for scale, scale_word in self._ordered_words:
            if scale == 0 or scale > value:
                continue
            coefficient, remainder = divmod(value, scale)
            special_coefficient = self.scale_coefficient_words.get(
                scale, {}
            ).get(coefficient)
            if special_coefficient is not None:
                result = special_coefficient
            elif coefficient == 1:
                if scale in self.omit_leading_one_scales:
                    result = scale_word
                else:
                    coefficient_word = self._compose_words(1)
                    result = self._join_scale(coefficient_word, scale_word)
            else:
                coefficient_word = self._compose_words(coefficient)
                result = self._join_scale(coefficient_word, scale_word)
            if remainder:
                result += self.remainder_separator + self._compose_words(remainder)
            return result
        raise ValueError(f"{self.language} cannot split number {value}")

    def _join_scale(self, coefficient_word: str, scale_word: str) -> str:
        """A few languages use a fixed scale-first word order in multiplicative combinations."""

        if self.scale_word_order == "scale_coefficient":
            return self.word_separator.join((scale_word, coefficient_word))
        return self.word_separator.join((coefficient_word, scale_word))

    def split_words(self, value: int) -> Tuple[str, ...]:
        """Keep old internal method names; composition boundaries are one returned span."""

        return (self._compose_words(value),)

    def integer_to_words(self, value: int) -> str:
        if self.enforce_max_value and value >= self.max_value:
            raise OverflowError(
                f"{self.language} number must be below {self.max_value}"
            )
        return self._compose_words(value)

    def cardinal(self, token: NumberToken) -> str:
        result = self.integer_to_words(token.integer_value)
        fraction = token.significant_fraction_digits
        if fraction:
            if self.fraction_min_digits or self.fraction_max_digits is not None:
                fraction = token.fraction_digits
                if self.fraction_min_digits:
                    fraction = fraction.ljust(self.fraction_min_digits, "0")
                if self.fraction_max_digits is not None:
                    fraction = fraction[: self.fraction_max_digits]
            result += f" {self.decimal_word} " + self.digit_separator.join(
                self.digit_words[int(digit)] for digit in fraction
            )
        if token.negative and not token.is_zero:
            return f"{self.minus_word} {result}"
        return result

    def digit_sequence(self, digits: str) -> str:
        if not digits.isdigit():
            raise ValueError(f"digit sequence must contain only digits 0-9: {digits!r}")
        return self.digit_separator.join(
            self.digit_words[int(digit)] for digit in digits
        )
