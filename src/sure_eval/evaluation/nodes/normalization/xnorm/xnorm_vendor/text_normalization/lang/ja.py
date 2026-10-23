# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

from pathlib import Path
import re

from ..base import TextNormalization_Base

# \u4e00-\u9fff is the shared CJK pool, not Japanese-only kanji, so there is no simple way to drop Simplified-only characters.
# See: https://www.doubao.com/thread/w8e0e9496c666fcbe
# Unicode Character Database (https://unicode.org/ucd/) exposes per-character properties
#    kTraditionalVariant: traditional form (if set, the current character is simplified);
#    kSimplifiedVariant: simplified glyph (points to simplified if traditional; empty/self if already simplified);
#    kHanYu: listed in Hanyu Da Zidian (likely Simplified Chinese).
# TODO: collect Japanese-only kanji from those fields later.
alphabet_pattern = '\\u3005\\u3007\\u3040-\\u309c\\u30a0-\\u30fc\\u30ff\\u4e00-\\u9fff' # Japanese: 々 \u3005; 〇 \u3007; hiragana \u3040-\u309c without the trailing iterator; katakana \u30a0-\u30fc without the iterator, keep ・; kanji \u4e00-\u9fff
diacritic_pattern = ''

# Simplified-to-Japanese kanji is a glyph rewrite, not a semiotic class, so it stays out of overlay whitelist.
_SIMPLIFIED_HANZI_TSV = (
    Path(__file__).resolve().parent / "data" / "ja" / "simplified_hanzi.tsv"
)


def load_simplified_hanzi_pairs(path=_SIMPLIFIED_HANZI_TSV):
    """Load the two-column TSV at init. Skip if the public tree has no table; duplicate keys error."""
    path = Path(path)
    if not path.is_file():
        # The public export omits this TSV; keep the original if it is missing.
        return []
    pairs = []
    seen = {}
    text = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), 1):
        raw = line.split("#", 1)[0].strip()
        if not raw:
            continue
        parts = raw.split("\t")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"{path}:{line_no} must be simplified<TAB>japanese")
        src, dst = parts[0], parts[1]
        if src in seen:
            raise ValueError(f"{path}:{line_no} duplicate key {src!r}, already defined on line {seen[src]}")
        seen[src] = line_no
        pairs.append((src, dst))
    return pairs


def compile_simplified_hanzi_replacer(pairs):
    # Replace longer keys first so two-character entries are not split.
    if not pairs:
        return lambda text: text
    ordered = tuple(sorted(pairs, key=lambda item: len(item[0]), reverse=True))
    mapping = dict(ordered)
    pattern = re.compile("|".join(re.escape(src) for src, _dst in ordered))

    def replace(text):
        if not text:
            return text
        return pattern.sub(lambda match: mapping[match.group(0)], text)

    return replace


_replace_simplified_hanzi = compile_simplified_hanzi_replacer(
    load_simplified_hanzi_pairs()
)

class TextNormalization_JA(TextNormalization_Base):
    def __init__(self):
        super().__init__()
        # Language-specific fields
        self.language = "ja"
        self.language_name_en = "Japanese"
        self.language_name_zh = "日语"
        self.hello_world = "こんにちは、世界！"
        self.alphabet_pattern = alphabet_pattern
        # Language-specific regexes can also be initialized here

        # Default config values
        self.spaced_writing = False
        self.score_method = "CER"  # Score by character, not word
        self.script = "Japanese" # Chinese and Japanese
        #self.keep_english_letters = False # this language has no English letters

    def convert_simplified_hanzi(self, text):
        """Replace Simplified Hanzi with Japanese jōyō forms before NeMo / NumberGrammar."""
        return _replace_simplified_hanzi(text)

    def pipeline(self, text):
        if not text:
            return text
        text = self.convert_simplified_hanzi(text)
        if not text:
            return text
        return super().pipeline(text)
