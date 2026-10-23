# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

import re
from ..base import TextNormalization_Base
from ..logger import logger

alphabet_pattern = '\\u0530-\\u058F' 
diacritic_pattern = '' 

class TextNormalization_HY(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "hy"
        self.language_name_en = "Armenian"
        self.language_name_zh = "亚美尼亚语"
        self.hello_world = "Բարև, աշխարհ!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Other regexes need no override unless you know why

        # Default config values
        self.spaced_writing = True 
        self.score_method = "WER"  
        self.script = "Armenian"         
        

