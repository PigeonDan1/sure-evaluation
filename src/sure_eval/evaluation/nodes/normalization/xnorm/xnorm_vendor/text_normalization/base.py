# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Shared base class and per-line pipeline for language normalizers."""

import codecs,os,sys,io
import argparse
import unicodedata
import re
from .distribution import default_nemo_backend
from .logger import logger
from .utils import replace_invisible_chars, simple_parse_pattern, simple_pattern_difference
from .numbers.spans import rewrite_number_spans
from .core.diagnostics import RuntimeCounters
from .numbers.cache import BoundedNumberCache

# Characters are grouped as follows:
# (1) Alphabet: alphabet_pattern for this language
# (2) English: english_word_pattern / english_letter_pattern
#     ASCII: English letters and digits
# (3) Punctuation: marks_pattern
# (4) Diacritics: diacritic_pattern, only some languages, e.g. Hebrew
# (5) Illegal: everything else

alphabet_pattern='a-zA-Z' # Defined per language
diacritic_pattern=''        # Defined per language

# ASCII character pattern
ascii_pattern='\\u0020-\\u007f'

# English-letter pattern
english_word_pattern='\\sa-zA-Z\\x27\\x2d' # Single quote \x27, dash \x2d
english_letter_pattern='a-zA-Z'

# Punctuation pattern: replace marks with space or empty
marks_pattern_category = {
    "latin"      : '\\u00a0\\u00a1\\u00aa\\u00ab\\u00b0\\u00b4\\u00b7\\u00ba\\u00bb\\u00bf', # Common Latin punctuation: ¡ª«°´·º»¿
    "combine"    : '\\u0300-\\u036F', # Combining Diacritical Marks ˋ ˊ ¨ ～ attach to a base (è = e + \u0300), used in Latin/Greek/French.
    "hebrew"     : '\\u05f3-\\u05f4',  # Hebrew Marks. ׳、״
    "armenian"   : '\\u055a-\\u055f\\u0589',  # ՚՛՜՝՞՟։ 
    "arabic"     : '\\u0600-\\u061f\\u0656-\\u065f\\u066a-\\u066d\\u06d4\\u06d6-\\u06ed', # Arabic Marks. ،؍؛؟۔ https://unicodeplus.com/script/Arab
    "devanagari" : '\\u0964-\\u0965', #  । ॥ 
    "telugu"     : '\\u0c64-\\u0c65', # 
    "lao"        : '\\u0e4f\\u0e5a\\u0e5b', # ๏ ๚ ๛
    "sinhala"    : '\\u0df4', # ෴
    "myanmar"    : '\\u104a-\\u104f',  #  ၊ ။ ၌ ၍ ၎ ၏
    "amharic"    : '\\u1361-\\u1368',  #  ፡።  ፣ ፤ ፥ ፦ ፧፨
    "khmer"      : '\\u17d4-\\u17d7',  # ។ ៕ ៖ ៗ
    "common"     : '\\u2000-\\u206F\\uA788-\\uA78C\\uA78F', # Shared punctuation: fullwidth spaces, ellipses, dashes (… = ellipsis)
    "math"       : '\\u2212\\u2213\\u2214\\u2217\\u2219\\u2219\\u22C5\\u22CF', # Math symbols −, ±, ∓
    "drawing"    : '\\u2500-\\u257F', # Box-drawing characters (─ horizontal, │ vertical), used in text tables.
    "japanese"   : '\\u3001-\\u3003\\u3008-\\u301F\\u3030\\u303b\\u309d\\u309e\\u30fb\\u30fd\\u30fe', # Japanese Marks. 、。〃〈〉「」〰〻ゝゞ・ㇽㇾ
    "full_width" : '\\uFF01-\\uFF0F\\uFF1A-\\uFF20\\uFF3B-\\uFF40\\uFF5B-\\uFF65', # Fullwidth punctuation
}

marks_pattern = ''.join(marks_pattern_category.values()) 
# see: https://www.fuhaoku.net/blocks

# All Latin letters
latin_alphabet_pattern = 'a-zA-ZÀ-ÿ\\u00C0-\\u017F\\u0180-\\u024F\\u0250-\\u02AF\\u0370-\\u03FF\\u0400-\\u04FF\\u0500-\\u052F'

# Code-switch language tags, e.g. <de-DE>, </de-DE>, <en>.
CODE_SWITCH_TAG_PATTERN = re.compile(r'</?([a-zA-Z]{2,3})(?:-[A-Za-z0-9]{2,8})?>')

CODE_SWITCH_MODES = {
    "delete",
    "keep_start",
    "keep_start_base",
}


def initialize_nemo_engine(*args, **kwargs):
    """Import dependencies and initialize WFSTs only when NeMo is explicitly configured."""
    from .rule_engines import initialize_nemo_engine as initialize

    return initialize(*args, **kwargs)

# Helpers used while defining language patterns; keep them outside the class
# NFKC normalization
def fun_normalize_nfkc(text: str, debug: int = 0):
    """
    Unicode NFKC (Normalization Form KC): Compatibility Decomposition, followed by Canonical Composition.
    It unifies characters that look or function the same but have different encodings.
    Typical effects:
    - Circled digits (① U+2460) → digit 1 (U+0031)
    - Fullwidth letters (Ａ U+FF21) → ASCII A (U+0041)
    - Fullwidth symbols (＋ U+FF0B) → + (U+002B)
    - Ligatures (ﬁ U+FB01) → f (U+0066) + i (U+0069)
    - e (U+0065) + combining acute (U+0301) → precomposed é (U+00E9)
    - Remove invisible controls such as ZWSP (U+200B)
    - Roman numerals (Ⅳ U+2163) → 4 (U+0034)
    - Vulgar fractions (⅓ U+2153) → 1 (U+0031) + / (U+002F) + 3 (U+0033)
    - Alternate spaces (NBSP U+00A0) → U+0020 in most compatibility cases.
    """
    text2 = unicodedata.normalize("NFKC", text)

    # NFKC may decompose without recomposing; run NFC afterwards.
    # E.g. Thai "กำ" (U+0e01 U+0e33) becomes "กํา" (U+0e01 U+0e4d U+0e32) after NFKC,
    # Looks different (some fonts still render them alike)
    # Ugly fallback map until the normalizer behavior is fully understood
    manual_mapping = { 
        '\u0e4d\u0e32' : '\u0e33' ,
    }
    for decomposed, composed in manual_mapping.items():
        text2 = text2.replace(decomposed, composed)

    if debug > 0 and text != text2:
        logger.debug(f"fun_i: {text}")
        logger.debug(f"fun_o: {text2}")
    if debug == 2 and text != text2:
        logger.debug(f"fun_i: ")
        for ch in text: logger.debug(f"char = {ch} \t unicode = {ord(ch):04x}")
        logger.debug(f"fun_o: ")
        for ch in text2: logger.debug(f"char = {ch} \t unicode = {ord(ch):04x}")

    return text2

# Build fixed code-point maps once; do not recreate 90 entries per input line.
SPECIAL_NUMBER_TRANSLATION = str.maketrans({
    chr(start + offset): str(offset)
    for start in (
        0x0660, 0x06F0, 0x0966, 0x09E6, 0x0E50,
        0x0F20, 0x0A66, 0x0D66, 0x0DE6,
    )
    for offset in range(10)
})


def fun_convert_special_numbers_to_arabic(text: str, debug: int = 0) -> str:
    """
    Convert worldwide digit glyphs to Western Arabic digits:
    Arabic     ٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩ 0 1 2 3 4 5 6 7 8 9
    Persian / Urdu   ۰ ۱ ۲ ۳ ۴ ۵ ۶ ۷ ۸ ۹ 0 1 2 3 4 5 6 7 8 9
    Hindi   ० १ २ ३ ४ ५ ६ ७ ८ ९ 0 1 2 3 4 5 6 7 8 9
    Thai     ๐ ๑ ๒ ๓ ๔ ๕ ๖ ๗ ๘ ๙ 0 1 2 3 4 5 6 7 8 9
    Tibetan     ༠ ༡ ༢ ༣ ༤ ༥ ༦ ༧ ༨ ༩ 0 1 2 3 4 5 6 7 8 9
    Bengali     ০ ১ ২ ৩ ৪ ৫ ৬ ৭ ৮ ৯ 0 1 2 3 4 5 6 7 8 9    
    """
    text2 = text.translate(SPECIAL_NUMBER_TRANSLATION)
    if debug > 0 and text != text2:
        logger.debug(f"fun_i: {text}")
        logger.debug(f"fun_o: {text2}")

    return text2



# Shared regexes, helpers, and the base class
class TextNormalization_Base:
    def __init__(self):
        # Shared regexes and variables can be initialized here
        self.is_tts_pipeline = False
        self.language = "xx"
        self.language_name_en = "XXX"
        self.language_name_zh = "XX语"
        self.hello_world = "Hello, world!" # Sample sentence
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        self.ascii_pattern = ascii_pattern
        self.english_word_pattern = english_word_pattern
        self.english_letter_pattern = english_letter_pattern
        self.marks_pattern = marks_pattern

        # Business mode: ASR training (asr_train), ASR eval (asr_eval), or TTS (tts)
        self.mode = 'asr_eval'

        # Default config values
        self.spaced_writing = True # Most languages are spaced (segmental) writing; unspaced scripts include Chinese, Japanese, Thai, Lao, Burmese, ...
        self.score_method = "WER"  # A few languages should score CER rather than WER
        self.script = ""           # Script families (latin, arabic, cyrillic, devanagari, brahmic, cj, ethiopic), https://worldschoolbooks.com/the-most-widely-used-scripts/
        self.allow_leading_single_quote = False # Dutch (nl) and Hausa (ha) allow a leading quote
        self.allow_trailing_single_quote = False # Greek allows a trailing single quote
        self.debug = False
        self.case = ""
        self.normalize_text = True
        self.keep_empty_lines = True
        self.keep_english_letters = True
        self.remove_lines = False            # Do not drop whole lines by default
        self.remove_brackets = False
        self.remove_diacritic = True
        self.remove_not_ascii_lang_mark = True
        # Strip other-script characters; mode sets the default, language files or users may override
        self.remove_foreign_chars = None
        self.remove_dashes = False
        self.remove_single_quotes = False
        self.normalize_digit_maxlen = 12
        # Official languages load NeMo automatically; others use NumberGrammar only.
        self.itn = False
        self.nemo_backend = "official"
        self.nemo_input_case = "cased"
        self.nemo_itn_input_case = "lower_cased"
        self.nemo_cache_dir = None
        self.nemo_overwrite_cache = False
        self.nemo_whitelist = None
        self._nemo_engine = None
        # Code-switch tag policy defaults come from mode; explicit user values win.
        self.code_switch_tag = None

        # Statistics
        self.num_removed_lines = 0

        self.cached_num_map = BoundedNumberCache(max_size=4096)
        self.runtime_counters = RuntimeCounters()
        # Regexes depend on language and request config; do not share at module level.
        self.re_line_contains_invalid_char = None
        self.re_line_no_letter = None
        self.re_line_no_local_letter = None
        self.re_line_four_consecutive_letters = None
        self.re_char_not_ascii_lang_mark = None
        self.re_char_diacritic = None
        self.re_foreign_chars = None
        self.re_brackets_content = None
        self.re_word_typo_single_quote = None

    def get_language():
        return [self.language, self.language_name_en, self.language_name_zh, self.spaced_writing, self.score_method, self.hello_world]

    def _apply_remove_foreign_chars_default(self):
        # This feature also needs a foreign-char regex, so do not disable it by mode yet.
        if self.mode in ('asr_eval', 'asr_train'):
            self.remove_foreign_chars = True
        elif self.mode == 'tts':
            self.remove_foreign_chars = False

    def _apply_code_switch_tag_default(self):
        # ASR eval strips tags by default; train/TTS keep leading base-language tags for legacy data.
        if self.mode == "asr_eval":
            self.code_switch_tag = "delete"
        elif self.mode in ("asr_train", "tts"):
            self.code_switch_tag = "keep_start_base"

    def config(self, **kwargs):
        if "tn_engine" in kwargs:
            raise ValueError("tn_engine has been removed; TN always runs NeMo first (official languages) then NumberGrammar")
        if "nemo_far" in kwargs:
            raise ValueError("nemo_far has been removed; this project writes direct ITN FARs to nemo_cache_dir")
        if "number_engine" in kwargs:
            raise ValueError("number_engine has been removed; the number path uses NumberGrammar only")
        remove_foreign_chars = kwargs.get("remove_foreign_chars")
        code_switch_tag = kwargs.get("code_switch_tag")
        itn = kwargs.get("itn")
        normalize_text = kwargs.get("normalize_text")
        nemo_backend = kwargs.get("nemo_backend")
        nemo_input_case = kwargs.get("nemo_input_case")
        nemo_itn_input_case = kwargs.get("nemo_itn_input_case")
        nemo_cache_dir = kwargs.get("nemo_cache_dir")
        nemo_overwrite_cache = kwargs.get("nemo_overwrite_cache")
        nemo_whitelist = kwargs.get("nemo_whitelist")

        # Override defaults with kwargs
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        if remove_foreign_chars is not None:
            self.remove_foreign_chars = remove_foreign_chars
        elif self.remove_foreign_chars is None:
            self._apply_remove_foreign_chars_default()

        if code_switch_tag is None:
            self._apply_code_switch_tag_default()

        # Global language instances are reconfigured; restore deterministic defaults per request.
        if itn in (None, False, 0, "0"):
            self.itn = False
        elif itn in (True, 1, "1"):
            self.itn = True
        else:
            raise ValueError("itn must be 0 or 1")
        if normalize_text in (None, True, 1, "1"):
            self.normalize_text = True
        elif normalize_text in (False, 0, "0"):
            self.normalize_text = False
        else:
            raise ValueError("normalize_text must be 0 or 1")
        self.nemo_backend = (
            nemo_backend if nemo_backend is not None else default_nemo_backend()
        )
        self.nemo_input_case = (
            nemo_input_case if nemo_input_case is not None else "cased"
        )
        self.nemo_itn_input_case = (
            nemo_itn_input_case
            if nemo_itn_input_case is not None
            else "lower_cased"
        )
        self.nemo_cache_dir = nemo_cache_dir
        self.nemo_overwrite_cache = (
            nemo_overwrite_cache if nemo_overwrite_cache is not None else False
        )
        self.nemo_whitelist = nemo_whitelist

        if self.code_switch_tag not in CODE_SWITCH_MODES:
            raise ValueError(
                "code_switch_tag must be one of: "
                + ", ".join(sorted(CODE_SWITCH_MODES))
            )

        if self.nemo_backend not in {"official", "direct"}:
            raise ValueError("nemo_backend must be official or direct")
        if self.nemo_input_case not in {"cased", "lower_cased"}:
            raise ValueError("nemo_input_case must be cased or lower_cased")
        if self.nemo_itn_input_case not in {"cased", "lower_cased"}:
            raise ValueError("nemo_itn_input_case must be cased or lower_cased")

        self.init_regex_patterns()

        # Restore FARs for official languages before the line loop; the hot path only infers.
        self._nemo_engine = None
        if self._should_load_nemo():
            self._initialize_nemo_engine()
        elif self.itn:
            logger.warning("language %s has no NeMo ITN; ITN will keep the original text", self.language)

        if self.debug > 0: 
            print("All parameters:", flush=True, file=sys.stderr)
            max_key_len = max(len(str(k)) for k in self.__dict__.keys())
            for k, v in self.__dict__.items():
                print(f"  --{k:<{max_key_len}}    {v}", flush=True, file=sys.stderr)

    def init_regex_patterns(self):
        """Compile every regex once"""
        # alphabet_pattern should exclude marks_pattern
        if True: # Languages that do not support escapes yet can be listed here
            is_debug = False
            if is_debug: 
                logger.info(f"alphabet_pattern = {self.alphabet_pattern}")
                logger.info(f"marks_pattern = {self.marks_pattern}")
            alphabet_pattern_unicode, _ = simple_pattern_difference(self.alphabet_pattern, self.marks_pattern)
            if is_debug: 
                logger.info(f"new alphabet_pattern = {alphabet_pattern_unicode}")
            self.alphabet_pattern = alphabet_pattern_unicode

        # 1. Line contains illegal characters
        if self.keep_english_letters:
            # Keep English
            pattern_a = r'[^' + self.ascii_pattern + self.alphabet_pattern + self.marks_pattern + self.diacritic_pattern + ']'
        else:
            # English is not kept as a special case, but if a-zA-Z is in alphabet_pattern it will not be deleted
            pattern_a = r'[^' + '\\u0000-\\u0040\\u005b-\\u0060\\u007b-\\u007f' + self.alphabet_pattern + self.marks_pattern + self.diacritic_pattern + ']'
        if self.debug:
            logger.info("RE_LINE_CONTAINS_INVALID_CHAR = " + pattern_a)
        self.re_line_contains_invalid_char = re.compile(pattern_a)

        # 2. Line has no letters: no alphabet or English letters
        pattern_b = r'^[^' + self.english_letter_pattern + self.alphabet_pattern + ']*$'
        #logger.info("RE_LINE_NO_LETTER = " + pattern_b)
        self.re_line_no_letter = re.compile(pattern_b)

        # 3. Line has no alphabet letters
        pattern_c = r'^[^' + self.alphabet_pattern + ']*$'
        #logger.info("RE_LINE_NO_LOCAL_LETTER = " + pattern_c)
        self.re_line_no_local_letter = re.compile(pattern_c)

        # 4. Line has 4 identical letters in a row
        pattern_d = r'([' + self.alphabet_pattern + '])\\1\\1\\1'
        #logger.info("RE_LINE_FOUR_CONSECUTIVE_LETTERS = " + pattern_d)
        self.re_line_four_consecutive_letters = re.compile(pattern_d)

        # 5. Not ASCII, language letter, or punctuation
        pattern_chars_a = r'[^' + self.ascii_pattern + self.alphabet_pattern + self.marks_pattern + ']'
        #logger.info("RE_CHAR_NOT_ASCII_LANG_MARK = " + pattern_chars_a)
        self.re_char_not_ascii_lang_mark = re.compile(pattern_chars_a)

        # 6. Not an English or language letter
        # After verbalization, leftover digits are stripped. Keep ASCII digits when TN engines are off.
        foreign_keep = self.english_word_pattern + self.alphabet_pattern
        if not self.normalize_text:
            foreign_keep += '0-9'
        pattern_chars_b = r'[^' + foreign_keep + ']'
        #logger.info("RE_FOREIGN_CHARS = " + pattern_chars_b)
        self.re_foreign_chars = re.compile(pattern_chars_b)

        # 7. Diacritics
        if self.diacritic_pattern:
            d_pattern = r'[' + self.diacritic_pattern + ']'
            if self.debug > 0: 
                logger.info("RE_CHAR_DIACRITIC = " + d_pattern)
            self.re_char_diacritic = re.compile(d_pattern)
        else:
            self.re_char_diacritic = None

        # 8. Brackets and their contents
        pattern_brackets = r'\([^()]*\)'
        #logger.info("RE_BRACKETS_CONTENT = " + pattern_brackets)
        self.re_brackets_content = re.compile(pattern_brackets)

        # 9. Replace irregular quotes only inside words
        # Char   Unicode  meaning                      where it appears
        # ’      U+2019  right single quote (most common)    I’m, don’t
        # ‘      U+2018  left single quotation mark     rare typos
        # ＇     U+FF07  fullwidth apostrophe          IME fullwidth slips
        # ′      U+2032  prime                     OCR/typesetting noise
        # `      U+0060  backtick                      keyboard typos
        alphabet_code_points = simple_parse_pattern(self.alphabet_pattern)
        # Characters a language puts in alphabet are orthography, not misspelled quotes.
        word_typo_single_quote_chars = ''.join(
            ch for ch in "‘’＇′`"
            if ord(ch) not in alphabet_code_points
        )
        escaped_single_quote_chars = ''.join(
            re.escape(ch) for ch in word_typo_single_quote_chars
        )
        if escaped_single_quote_chars:
            pattern_word_type_single_quote = r'(?<=[' + latin_alphabet_pattern + r'])[' + escaped_single_quote_chars + r'](?=[' + latin_alphabet_pattern + r'])'
        else:
            pattern_word_type_single_quote = r'(?!)'
        #logger.info("RE_WORD_TYPO_SINGLE_QUOTE = " + pattern_word_type_single_quote)
        self.re_word_typo_single_quote = re.compile(pattern_word_type_single_quote)


    # Drop the line if it matches the pattern
    def fun_remove_lines_pattern(self, text: str, pattern: re.Pattern):
        # Type check
        if not isinstance(pattern, re.Pattern):
            logger.error(f"pattern must be re.Pattern, not {type(pattern).__name__}")
            raise TypeError("aborting")

        if self.remove_lines:
            match = pattern.search(text)
            if match:  # Drop the line if it has illegal characters
                if self.debug > 0:
                    ch = match.group()
                    logger.debug(
                        f"matched: '{ch}', unicode: {ord(ch):04x}, line: {text}"
                    )
                self.num_removed_lines += 1
                return ""  # Drop the whole line

        return text


    # Delete characters matching the pattern
    def fun_remove_chars_pattern(self, text: str, pattern: re.Pattern, replacement:str):
        # Type check
        if not isinstance(pattern, re.Pattern):
            logger.error(f"pattern must be re.Pattern, not {type(pattern).__name__}")
            raise TypeError("aborting")

        # Replace every other illegal character
        text2 = pattern.sub(replacement, text) # Usually replace with space or empty

        if self.debug > 0 and text != text2:
            logger.debug(f"fun_i: {text}")
            logger.debug(f"fun_o: {text2}")
        if self.debug == 2 and text != text2:
            logger.debug(f"pattern: {pattern}")
            logger.debug(f"text: {text}")
            logger.debug(f"text: ")
            for ch in text: logger.debug(f"char = {ch} \t unicode = {ord(ch):04x}")
            logger.debug(f"alphabet_pattern: {self.alphabet_pattern}")
            for ch in self.alphabet_pattern: logger.debug(f"char = {ch} \t unicode = {ord(ch):04x}")

        return text2

    def fun_remove_chars_dashes(self, text: str):
        """
        Strip dashes:
        remove_dashes = 0: replace dashes that are not word joiners with space
        remove_dashes = 1: replace every dash with space (some languages may override to empty)
        """
        text2 = text
        if self.remove_dashes:
            # remove all dashes          
            text2 = re.sub(r'-+', " ", text2)
        else:
            # remove dashes that not a word connector
            text2 = re.sub(r'-+', r"-", text2)
            text2 = re.sub(r'([\s\'])[-]+', r"\g<1> ", text2)
            text2 = re.sub(r'[-]+([\s\'])', r" \g<1>", text2)
            text2 = re.sub(r'^-', r"", text2)
            text2 = re.sub(r'-$', r"", text2)

        text2 = re.sub(r'[ ]+', r" ", text2)
        
        if self.debug > 0 and text != text2:
            logger.debug(f"fun_i: {text}")
            logger.debug(f"fun_o: {text2}")
        return text2

    def fun_remove_chars_single_quotes(self, text: str):
        """
        Strip single quotes: some languages allow a leading quote
        remove_single_quotes = 0: replace quotes that are not word joiners with space
        remove_single_quotes = 1: replace every single quote with space
        """
        text2 = text
        if self.remove_single_quotes:
            # remove all single quotes          
            text2 = re.sub(r'\'+', " ", text2)
        else:
            # remove single quotes that not a word connector
            text2 = re.sub(r'\'+', r"'", text2) # merge continous single quotes

            if not self.allow_trailing_single_quote:
                text2 = re.sub(r'\'+([\s-])', r" \g<1>", text2) # remove single quotes before dash or space
                text2 = re.sub(r'\'$', r"", text2)

            if self.allow_leading_single_quote:
                text2 = re.sub(r'([-])\'+', r"\g<1> ", text2) # remove single quotes after dash
            else:
                text2 = re.sub(r'([\s-])\'+', r"\g<1> ", text2) # remove single quotes after dash or space
                text2 = re.sub(r'^\'', r"", text2) 

        text2 = re.sub(r'[ ]+', r" ", text2)
        
        if self.debug > 0 and text != text2:
            logger.debug(f"fun_i: {text}")
            logger.debug(f"fun_o: {text2}")
        return text2

    def fun_convert_case(self, text: str):
        """German, Turkish, and Azerbaijani must override this!"""
        
        if self.case == "upper":
            text2 = text.upper()
        elif self.case == "lower":
            text2 = text.lower()
        else:
            text2 = text

        # Enable debug logs only for special languages
        #if self.debug > 0 and text != text2:
        #    logger.debug(f"fun_i: {text}")
        #    logger.debug(f"fun_o: {text2}")
        return text2

    def fun_remove_english_letters(self, text: str):
        """Vietnamese must override this!"""
        if self.keep_english_letters:
            return text
        else:
            if self.keep_empty_lines:
                text2 = re.sub(r'[a-zA-Z]', r" ", text) # Replace with a space
            else:
                text2 = re.sub(r'^.*[a-zA-Z].*$', r"", text) # Drop the whole line

            if self.debug > 0 and text != text2:
                logger.debug(f"fun_i: {text}")
                logger.debug(f"fun_o: {text2}")
            return text2

    def _get_code_switch_joiner(self) -> str:
        """Decide whether a tag needs an explicit space before the body, based on writing system."""
        return " " if self.spaced_writing else ""

    def _normalize_code_switch_tag(self, tag: str) -> str:
        """Rewrite legal code-switch language tags according to product mode."""
        if self.mode not in ("asr_eval", "asr_train", "tts"):
            return ""

        if self.code_switch_tag == "delete" or tag.startswith("</"):
            return ""

        if self.code_switch_tag == "keep_start":
            return tag

        match = CODE_SWITCH_TAG_PATTERN.fullmatch(tag)
        if self.code_switch_tag == "keep_start_base" and match:
            # Lowercase base language tags so <DE-DE> never enters training labels.
            return f"<{match.group(1).lower()}>"

        return ""

    def _join_code_switch_parts(self, parts: list[str]) -> str:
        """Join tag/body fragments without gluing spaced languages or inserting extra spaces in unspaced ones."""
        parts = [part for part in parts if part]
        if not parts:
            return ""
        joiner = self._get_code_switch_joiner()
        return joiner.join(parts)

    def _pipeline_with_code_switch_tags(self, text: str) -> str:
        """Split code-switch tags so legal tags are not damaged by generic character cleanup."""
        parts = []
        last_end = 0

        for match in CODE_SWITCH_TAG_PATTERN.finditer(text):
            raw_text = text[last_end:match.start()]
            normalized_text = self._pipeline_without_code_switch_tags(raw_text)
            if normalized_text:
                parts.append(normalized_text)

            normalized_tag = self._normalize_code_switch_tag(match.group())
            if normalized_tag:
                parts.append(normalized_tag)

            last_end = match.end()

        raw_text = text[last_end:]
        normalized_text = self._pipeline_without_code_switch_tags(raw_text)
        if normalized_text:
            parts.append(normalized_text)

        return self._join_code_switch_parts(parts)

    # Cleanup pipeline
    # Step order is mostly free, but verbalization must run before ASCII digits are stripped
    def _pipeline_without_code_switch_tags(self, text):
        text = text.strip()
        
        # Replace nonstandard controls with space
        # Later NFKC only turns NBSP (U+00A0) and ideographic space (U+3000) into U+0020,
        # Unusual spaces (halfwidth, zero-width, ...) still need this pass.
        if not text: return text
        text = replace_invisible_chars(text)
        
        # NFKC normalization
        if not text: return text
        text = fun_normalize_nfkc(text, debug=self.debug)

        # Normalize bad quotes inside words
        if not text: return text
        text = self.fun_remove_chars_pattern(
            text, self.re_word_typo_single_quote, "'"
        )
        
        # Strip diacritics
        if not text: return text
        if self.remove_diacritic and self.diacritic_pattern:
            text = self.fun_remove_chars_pattern(text, self.re_char_diacritic, "")
        
        # Map special digit glyphs to ASCII 0-9
        if not text: return text
        text = fun_convert_special_numbers_to_arabic(text, debug=self.debug)
        
        # Optional: drop lines with illegal characters
        if not text: return text
        text = self.fun_remove_lines_pattern(
            text, self.re_line_contains_invalid_char
        )
        
        # Replace non-ASCII / non-alphabet / non-punctuation with space
        if not text: return text
        if self.remove_not_ascii_lang_mark:
            text = self.fun_remove_chars_pattern(
                text,
                self.re_char_not_ascii_lang_mark,
                " " * self.spaced_writing,
            )
        
        # Verbalize written numbers/long-tail here; leftover ASCII digits are kept if this stage is skipped.
        if self.debug > 0:
            logger.debug(f"tn_begin: {text}")
        if not text: return text
        if self.normalize_text:
            if self._nemo_engine is not None:
                text = self._nemo_engine.normalize(text)
            # Digits already verbalized by NeMo will not match again; only leftover digits are filled. ITN returns early in pipeline().
            text = self.rewrite_numbers(text)
        if self.debug > 0:
            logger.debug(f"tn_end  : {text}")

        # Optional: drop lines that are only punctuation
        if not text: return text
        text = self.fun_remove_lines_pattern(text, self.re_line_no_letter)

        # Optional: strip brackets and contents
        if not text: return text
        if self.remove_brackets:
            text = self.fun_remove_chars_pattern(
                text, self.re_brackets_content, " " * self.spaced_writing
            )
        
        # Strip other-script characters
        if not text: return text
        if self.remove_foreign_chars:
            text = self.fun_remove_chars_pattern(
                text, self.re_foreign_chars, " " * self.spaced_writing
            )
        
        # Optional: strip English letters
        if not text: return text
        if not self.keep_english_letters:
            text = self.fun_remove_english_letters(text)
        
        # Optional: strip dashes
        if not text: return text
        text = self.fun_remove_chars_dashes(text)
        
        # Optional: strip illegal single quotes
        if not text: return text
        text = self.fun_remove_chars_single_quotes(text)

        # Drop lines that contain a run of more than 4 identical characters
        if not text: return text
        if self.mode != "asr_eval":
            # ASR-eval input is recognizer output; repeated letters may be model artifacts, so do not drop the line as in train/TTS.
            text = self.fun_remove_lines_pattern(
                text, self.re_line_four_consecutive_letters
            )

        # Optional: case conversion
        if not text: return text
        text = self.fun_convert_case(text)
        
        # Collapse leftover spaces before writing output
        if not text: return text
        text = ' '.join(text.split())

        return text


    def _should_load_nemo(self) -> bool:
        """Load NeMo only from language capability; default direct must not compose languages that have no pack."""
        from .rule_engines.nemo import (
            uses_official_nemo_itn,
            uses_official_nemo_tn,
            uses_private_ru_tn,
        )
        if self.itn:
            return uses_official_nemo_itn(self.language)
        if not self.normalize_text:
            return False
        return uses_official_nemo_tn(self.language) or uses_private_ru_tn(self.language)

    def _initialize_nemo_engine(self) -> None:
        """direct compose failure falls back to official; official init failure skips NeMo so the whole chain never crashes."""
        from .rule_engines.nemo import NemoDependencyError, resolve_nemo_language

        kwargs = {
            "language": resolve_nemo_language(self.language),
            "input_case": (
                self.nemo_itn_input_case if self.itn else self.nemo_input_case
            ),
            "cache_dir": self.nemo_cache_dir,
            "overwrite_cache": self.nemo_overwrite_cache,
            "whitelist": self.nemo_whitelist,
            "itn": self.itn,
            "backend": self.nemo_backend,
            "debug": self.debug,
        }
        try:
            self._nemo_engine = initialize_nemo_engine(**kwargs)
            return
        except NemoDependencyError:
            # Missing Pynini/NeMo is an environment error; do not silently degrade.
            raise
        except Exception as exc:
            if self.nemo_backend == "direct":
                logger.warning(
                    "direct graph build failed, falling back to official: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                self.nemo_backend = "official"
                kwargs["backend"] = "official"
                try:
                    self._nemo_engine = initialize_nemo_engine(**kwargs)
                    return
                except NemoDependencyError:
                    raise
                except Exception as official_exc:
                    logger.warning(
                        "official NeMo init failed, skipping NeMo: %s: %s",
                        type(official_exc).__name__,
                        official_exc,
                    )
                    self._nemo_engine = None
                    return
            logger.warning(
                "NeMo init failed, skipping NeMo: %s: %s",
                type(exc).__name__,
                exc,
            )
            self._nemo_engine = None

    @property
    def tn_engine(self):
        """Diagnostics field: nemo if NeMo loaded, otherwise number grammar only."""
        return "nemo" if self._nemo_engine is not None else "numbers"

    def classify_number_match(self, text, match):
        """Language-contextual numbers. Default is none so span dispatch has no language branches."""
        return None

    def rewrite_numbers(self, text: str) -> str:
        """Rewrite number spans with NumberGrammar."""
        return rewrite_number_spans(
            text,
            language=self.language,
            debug=self.debug,
            cached_num_map=self.cached_num_map,
            max_len=self.normalize_digit_maxlen,
            counters=self.runtime_counters,
            classify_match=self.classify_number_match,
        )

    def pipeline(self, text):
        try:
            return self._pipeline_unchecked(text)
        except Exception as exc:
            logger.warning(
                "pipeline failed, keeping original %r: %s: %s",
                text,
                type(exc).__name__,
                exc,
            )
            return text

    def _pipeline_unchecked(self, text):
        if self.itn:
            # Deployed FAR/ITN output must not go through TN cleanup, or dates and symbols break.
            if self._nemo_engine is None:
                return text.strip() if isinstance(text, str) else text
            return self._nemo_engine.normalize(text.strip())
        if not CODE_SWITCH_TAG_PATTERN.search(text):
            return self._pipeline_without_code_switch_tags(text)

        return self._pipeline_with_code_switch_tags(text)
