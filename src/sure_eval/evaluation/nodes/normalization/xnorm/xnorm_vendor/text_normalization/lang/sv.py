# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

import re
from ..base import TextNormalization_Base
from ..logger import logger

alphabet_pattern = 'a-zA-ZÅåÄäÖö'  
diacritic_pattern = '' 

class TextNormalization_SV(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "sv"
        self.language_name_en = "Swedish"
        self.language_name_zh = "瑞典语"
        self.hello_world = "Hej, världen!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Other regexes need no override unless you know why

        # Default config values
        self.script = "Latin"
