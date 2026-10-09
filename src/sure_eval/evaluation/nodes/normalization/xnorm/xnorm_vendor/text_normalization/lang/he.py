# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

from ..base import TextNormalization_Base

# Alphabet
# Hebrew: 22 base letters + 5 finals + maqaf (־ U+05BE); everything else is punctuation
alphabet_pattern = '\\u05be\\u05d0-\\u05ea'

# Diacritic pattern: replace with empty, not space
# Hebrew vowel points are part of the word; replacing them with spaces splits tokens
diacritic_pattern = '\\u0591-\\u05af\\u05b0-\\u05bd\\u05bf\\u05c0-\\u05c7'

class TextNormalization_HE(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "he"
        self.language_name_en = "Hebrew"
        self.language_name_zh = "希伯来语"
        self.hello_world = "שלום עולם!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.script = "Hebrew"
        #self.keep_english_letters = False # this language has no English letters
        
