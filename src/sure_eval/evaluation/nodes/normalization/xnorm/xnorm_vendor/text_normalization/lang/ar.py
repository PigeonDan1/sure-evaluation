# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

from ..base import TextNormalization_Base
from ..utils import to_unicode_codepoints
import re

# Alphabet
alphabet_pattern = '\\u0600-\\u06FF'
diacritic_pattern = '\\u064B-\\u0652'  # Arabic diacritical marks (Fatha, Damma, Hamza, etc.)'
# ASR strips vocalization by default. Standard Arabic still has a few marks; the rules are too heavy to model here.
# TTS may need to keep phonetic marks.

# https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard/blob/main/eval.py

tts_arabic_punctuation_marks = [
    # Arabic punctuation
    "،",  # Arabic comma
    "؛",  # Arabic semicolon
    "؟",  # Arabic question mark
    "٪",  # Arabic percent sign
    "٫",  # Arabic decimal point
    "٬",  # Arabic thousands separator
    "«",
    "»",  # Arabic quotation marks
    "ـ",  # Tatweel (kashida)

    # Western punctuation (often appears in mixed Arabic text)
    ".",
    ",",
    ";",
    ":",
    "?",
    "!",  # sentence marks
    '"',
    "'",
    "‘",
    "’",
    "“",
    "”",  # quotes (straight + curly)
    # "(",
    # ")",
    # "[",
    # "]",
    # "{",
    # "}",  # brackets / parentheses
    "-",
    "–",
    "—",  # hyphen/en dash/em dash
    "...",
    "…",  # ellipsis forms
    # "/",
    # "\\",
    # "|",
    # "_",  # slashes and connectors
    # "@",
    # "#",
    # "$",
    # "%",
    # "^",
    # "&",
    # "*",
    # "+",
    # "=",  # symbols
    # "<",
    # ">",
    # "~",
    # "`",  # misc
]

class TextNormalization_AR(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "ar"
        self.language_name_en = "Arabic"
        self.language_name_zh = "阿拉伯语"
        self.hello_world = "مرحبا، العالم!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.spaced_writing = True # Spaced writing
        self.score_method = "CER"  # Score by character, not word
        self.script = "Arabic"
        #self.keep_english_letters = False # this language has no English letters
        
    def config(self, **kwargs):
        # modify patterns before super's init_regex_pattern()
        if kwargs['mode'] == 'tts':
            self.remove_lines = True
            self.remove_diacritic = False
            self.remove_not_ascii_lang_mark = False
            self.remove_foreign_chars = False

            marks_pattern = ''.join(tts_arabic_punctuation_marks)
            marks_pattern = to_unicode_codepoints(marks_pattern)
            #print(f"marks_pattern = {marks_pattern}")
            if len(self.marks_pattern) > 0:
                self.ascii_pattern='a-zA-Z0-9 	'
                self.marks_pattern = marks_pattern

        super().config(**kwargs)
        
    def pipeline(self, text: str):

        text = super().pipeline(text)

        # Replace nonstandard Persian/Urdu letter variants
        if not text: return text
        text = re.sub('پ', 'ب', text)
        text = re.sub('ڤ', 'ف', text)

        if self.mode == 'asr_eval':
            if self.debug:
                print(f"mode = {self.mode}")
            # Strip hamza/madda for evaluation
            text = re.sub(r'[آ]', 'ا', text)
            text = re.sub(r'[أإ]', 'ا', text)
            text = re.sub(r'[ؤ]', 'و', text)
            text = re.sub(r'[ئ]', 'ي', text)
            text = re.sub(r'[ء]', '', text) # Arabic hamza

        return text
