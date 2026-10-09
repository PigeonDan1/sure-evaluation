# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

import re
from ..logger import logger
from ..base import TextNormalization_Base

# Alphabet
alphabet_pattern = 'AaBbCcDdEeFfGgHhIiKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZzÀàÁáÂâÃãÈèÉéÊêÌìÍíÒòÓóÔôÕõÙùÚúÝýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊ    ịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
diacritic_pattern = ''

class TextNormalization_VI(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "vi"
        self.language_name_en = "Vietnamese"
        self.language_name_zh = "越南语"
        self.hello_world = "Xin chào, thế giới!"
        self.alphabet_pattern = alphabet_pattern
        self.diacritic_pattern = diacritic_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.spaced_writing = True
        self.script = "Latin"

    def fun_remove_english_letters(self, text: str):
        """Vietnamese must override this! FJWZ occur only in Vietnamese loanwords."""
        if self.keep_english_letters:
            return text
        else:
            if self.keep_empty_lines:
                text2 = re.sub(r'[fjwzFJWZ]', r" ", text) # Replace with a space
            else:
                text2 = re.sub(r'^.*[fjwzFJWZ].*$', r"", text) # Drop the whole line

            if self.debug > 0 and text != text2:
                logger.debug(f"before: {text}")
                logger.debug(f" after: {text2}")
            return text2
        
