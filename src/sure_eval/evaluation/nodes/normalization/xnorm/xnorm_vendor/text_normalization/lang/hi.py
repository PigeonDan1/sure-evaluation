# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

import re
from ..base import TextNormalization_Base

devanagari_pattern = '\\u0900-\\u097F'
#devanagari_pattern = 'अआइईउऊऋॠऌॡएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळक्षज्ञािीुूृॄॢॣेैोौंःँ'  # incomplete
alphabet_pattern = devanagari_pattern
diacritic_pattern = ''

class TextNormalization_HI(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "hi"
        self.language_name_en = "Hindi"
        self.language_name_zh = "印地语"
        self.hello_world = "नमस्ते, दुनिया!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.spaced_writing = True
        self.script = "Devanagari"
        #self.keep_english_letters = False # this language has no English letters
