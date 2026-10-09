# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

from opencc import OpenCC

from ..base import TextNormalization_Base
from ..logger import logger

# Alphabet
alphabet_pattern = '\\u4e00-\\u9fff'
diacritic_pattern = ''

class TextNormalization_ZH(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "zh"
        self.language_name_en = "Chinese"
        self.language_name_zh = "中文"
        self.hello_world = "你好，世界！"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.spaced_writing = False
        self.score_method = "CER"  # Score by character, not word
        self.script = "Chinese" # Chinese and Japanese
        #self.keep_english_letters = False # this language has no English letters

        self.cc = None
        # CLI defaults chinese_conversion to t2s; API/compile_pipeline still need the attribute when omitted.
        self.chinese_conversion = "t2s"

    def config(self, **kwargs):
        super().config(**kwargs)

        if self.chinese_conversion and self.chinese_conversion not in ['none', '']:
            assert self.chinese_conversion in \
                ['hk2s', 's2hk', 's2t', 's2tw', 's2twp', \
                  't2hk', 't2s', 't2tw', 'tw2s', 'tw2sp']
            self.cc = OpenCC(self.chinese_conversion)

    def convert_chinese(self, text:str):
        """ Traditional/simplified conversion
            hk2s: Traditional Chinese (Hong Kong standard) to Simplified Chinese
            s2hk: Simplified Chinese to Traditional Chinese (Hong Kong standard)
            s2t: Simplified Chinese to Traditional Chinese
            s2tw: Simplified Chinese to Traditional Chinese (Taiwan standard)
            s2twp: Simplified Chinese to Traditional Chinese (Taiwan standard, with phrases)
            t2hk: Traditional Chinese to Traditional Chinese (Hong Kong standard)
            t2s: Traditional Chinese to Simplified Chinese
            t2tw: Traditional Chinese to Traditional Chinese (Taiwan standard)
            tw2s: Traditional Chinese (Taiwan standard) to Simplified Chinese
            tw2sp: Traditional Chinese (Taiwan standard) to Simplified Chinese (with phrases)
        """
        if self.cc:
            text2 = self.cc.convert(text)

            if self.debug > 0 and text != text2:
                logger.debug(f"Chinese Converion mode = {self.chinese_conversion}")
                logger.debug(f"fun_i: {text}")
                logger.debug(f"fun_o: {text2}")

            return text2
        else:
            return text
        
        
    def pipeline(self, text: str):

        if not text: return text
        text = self.convert_chinese(text)

        if not text: return text
        text = super().pipeline(text)

        return text
