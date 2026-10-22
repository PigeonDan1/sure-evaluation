# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Build the shared recursive grammar from versioned locale data."""

import json
from pathlib import Path
from typing import Dict

from .recursive import RecursiveScaleGrammar


class ConfiguredRecursiveGrammar(RecursiveScaleGrammar):
    """Bind pure data config to the single RecursiveScaleGrammar implementation."""

    def __init__(self, language: str, config: dict):
        self.language = language
        self.minus_word = config["minus_word"]
        self.decimal_word = config["decimal_word"]
        self.digit_words = tuple(config["digit_words"])
        self.number_words = tuple(
            (int(value), word) for value, word in config["number_words"]
        )
        self.exact_number_words = tuple(
            (int(value), word) for value, word in config.get("exact_number_words", ())
        )
        exact_lookup_max = config.get("exact_lookup_max")
        self.exact_lookup_max = (
            int(exact_lookup_max) if exact_lookup_max is not None else None
        )
        self.scale_coefficient_words = {
            int(scale): {
                int(coefficient): word
                for coefficient, word in coefficient_words.items()
            }
            for scale, coefficient_words in config.get(
                "scale_coefficient_words", {}
            ).items()
        }
        self.omit_leading_one_scales = frozenset(
            int(value) for value in config["omit_leading_one_scales"]
        )
        self.remainder_separator = config.get("remainder_separator", " ")
        self.scale_word_order = config.get(
            "scale_word_order", "coefficient_scale"
        )
        self.sequential_merge = bool(config.get("sequential_merge", False))
        self.nested_split = bool(config.get("nested_split", False))
        self.reverse_tens_one_scales = frozenset(
            int(value) for value in config.get("reverse_tens_one_scales", ())
        )
        self.reverse_tens_one_connector = config.get(
            "reverse_tens_one_connector", " "
        )
        self.reverse_right_words = {
            int(value): word
            for value, word in config.get("reverse_right_words", {}).items()
        }
        self.forward_tens_one_scales = frozenset(
            int(value) for value in config.get("forward_tens_one_scales", ())
        )
        self.forward_tens_one_connector = config.get(
            "forward_tens_one_connector", " "
        )
        self.general_addition_connector = config.get(
            "general_addition_connector"
        )
        self.agreement_scales = frozenset(
            int(value) for value in config.get("agreement_scales", ())
        )
        self.agreement_words = {
            int(value): word
            for value, word in config.get("agreement_words", {}).items()
        }
        self.default_merge_operation = config.get(
            "default_merge_operation", "add"
        )
        self.fraction_min_digits = int(config.get("fraction_min_digits", 0))
        fraction_max_digits = config.get("fraction_max_digits")
        self.fraction_max_digits = (
            int(fraction_max_digits) if fraction_max_digits is not None else None
        )
        self.enforce_max_value = bool(config.get("enforce_max_value", True))
        oracle_upper = config.get("oracle_random_upper_bound")
        self.oracle_random_upper_bound = (
            int(oracle_upper) if oracle_upper is not None else None
        )
        super().__init__()


def load_recursive_grammar_data(path: Path = None) -> dict:
    data_path = path or Path(__file__).resolve().parent / "data/recursive_light.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("languages"), dict):
        raise ValueError(f"recursive grammar data is invalid: {data_path}")
    return data


def create_configured_recursive_grammars() -> Dict[str, ConfiguredRecursiveGrammar]:
    data = load_recursive_grammar_data()
    return {
        language: ConfiguredRecursiveGrammar(language, config)
        for language, config in data["languages"].items()
    }
