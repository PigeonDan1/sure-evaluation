# Copyright (c) 2026, AISpeech Co., Ltd. All rights reserved.

"""Character cleanup and simple regex helpers."""

import unicodedata
import re
import logging

# Unicode assigns a category string; its first letter is the major class. C=control, Z=separator. Details:
# Control category (C)
# Controls have no glyph and drive devices/layout. Category C has these subcategories:
# Cc (control)
#    ASCII controls: \x00 NUL, \x07 BEL, \x08 BS, \x09 TAB, \x0A LF, \x0D CR, ...
#    Other controls such as \x1B (ESC) for terminal formatting.
# Cf (format)
#    Zero-width space \u200B, invisible but can affect wrapping.
#    Right-to-left mark \u200F, used for Arabic, Hebrew, etc.
# Cs (surrogate)
#    Surrogates encode non-BMP characters. High: \uD800-\uDBFF, low: \uDC00-\uDFFF.
# Co (private use)
#    Private-use characters, e.g. \uE000-\uF8FF on the BMP.
# Cn (unassigned)
#    Unassigned code points in Unicode.
#
# Separator category (Z)
# Separator category Z splits paragraphs, sentences, words, etc., with several subcategories:
# Zs (space separator)
#    Space \x20, the common word separator.
#    NBSP \u00A0 looks like space but does not wrap.
#    Ideographic space \u3000, a fullwidth space in CJK layout.
# Zl (line separator)
#    \u2028 line separator, similar to newline in some processors.
# Zp (paragraph separator)
#    \u2029 paragraph separator.

# Common zero-width code points
ZERO_WIDTH_CHARS = {
    '\u200B',  # Zero-width space
    '\u200C',  # Zero-width non-joiner
    '\u200D',  # Zero-width joiner
    '\uFEFF',  # BOM / word joiner
    '\u180B',  # Mongolian Free Variation Selector 1
    '\u180C',  # Mongolian Free Variation Selector 2
    '\u180E',  # Mongolian vowel separator
    '\u2060',  # Word joiner
}

def is_zero_width(char):
    """Whether a character is zero-width"""
    return len(char) == 1 and char in ZERO_WIDTH_CHARS

def replace_invisible_chars(text):
    # Plain ASCII lines skip per-character Unicode lookups; this is the million-line fast path.
    if text.isascii() and text.isprintable():
        return text
    keep_chars = {'\n', '\t'}
    output = []
    # Replace every Unicode invisible control
    for char in text:
        if is_zero_width(char): # Some zero-width characters are outside C/Z, e.g. \u180B \u180C
            #print(f"char = {char}, unicode = {ord(char):04x}, width = 0")
            continue
        elif unicodedata.category(char)[0] in {'C', 'Z'} and char not in keep_chars:
            #print(f"char = {char}, unicode = {ord(char):04x}, width > 0")
            output.append(" ")
        else:
            output.append(char)

    return "".join(output)

def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')    


def simple_parse_pattern(pattern):
    """Parse a pattern string into the set of Unicode code points. Only simple patterns: characters or ranges, no escapes or wildcards."""
    code_points = set()

    # Match a character, a Unicode value, or a range
    pattern_re = re.compile(
        r'(?:'
        r'\\u[0-9a-fA-F]{4}|'  # Unicode value, e.g. \u0020
        r'[^-]'                 # Single character other than '-'
        r')'
        r'(?:-(?:'
        r'\\u[0-9a-fA-F]{4}|'  # Unicode value at range end
        r'[^-]'                 # Single character at range end
        r'))?'                  # Optional range part
    )
    # Walk every match
    for part in pattern_re.findall(pattern):
        if '-' in part:
            # Handle ranges
            start_str, end_str = part.split('-', 1)
            # Parse range start
            if start_str.startswith('\\u'):
                start = int(start_str[2:], 16)
            else:
                start = ord(start_str)
            # Parse range end
            if end_str.startswith('\\u'):
                end = int(end_str[2:], 16)
            else:
                end = ord(end_str)
            # Start code point must not exceed end
            if start > end:
                start, end = end, start
            # Add every code point in the range
            for code in range(start, end + 1):
                code_points.add(code)
        else:
            # Handle a single character or code point
            if part.startswith('\\u'):
                code = int(part[2:], 16)
            else:
                code = ord(part)
            code_points.add(code)
    return code_points

def simple_merge_intervals(code_points):
    """Merge code points into contiguous ranges. Only simple patterns: characters or ranges, no escapes or wildcards."""
    if not code_points:
        return []
    # Sort code points
    sorted_codes = sorted(code_points)
    intervals = []
    # Initialize the first range
    current_start = current_end = sorted_codes[0]
    # Merge remaining consecutive code points
    for code in sorted_codes[1:]:
        if code == current_end + 1:
            current_end = code
        else:
            intervals.append((current_start, current_end))
            current_start = current_end = code
    # Append the last range
    intervals.append((current_start, current_end))
    return intervals

def simple_format_interval(interval, use_chars):
    """Format one range as a string. Only simple patterns: characters or ranges, no escapes or wildcards."""
    start, end = interval
    # Format range start
    if use_chars and 32 <= start <= 126:
        start_str = chr(start)
    else:
        start_str = f'\\u{start:04x}'
    # Format range end (omit it when it equals the start)
    if start == end:
        return start_str
    if use_chars and 32 <= end <= 126:
        end_str = chr(end)
    else:
        end_str = f'\\u{end:04x}'
    return f'{start_str}-{end_str}'

def simple_pattern_difference(pattern1, pattern2):
    """Set-difference two patterns and return both forms. Only simple patterns: characters or ranges, no escapes or wildcards."""
    # Parse code-point sets of both patterns
    set1 = simple_parse_pattern(pattern1)
    set2 = simple_parse_pattern(pattern2)
    # Compute set difference
    diff_set = set1 - set2
    # Merge ranges
    intervals = simple_merge_intervals(diff_set)
    # Produce both forms
    mixed = ''.join(simple_format_interval(iv, use_chars=True) for iv in intervals)
    unicode_only = ''.join(simple_format_interval(iv, use_chars=False) for iv in intervals)
    return mixed, unicode_only

def to_unicode_codepoints(items):
    """
    Convert a character or code point to a unified Unicode escape (e.g. '\\u2026').

    - str: per character, e.g. '…' -> '\\u2026', '...' -> '\\u002e\\u002e\\u002e'
    - int: treated as a code point, e.g. 0x2026 -> '\\u2026'
    Other types raise TypeError.
    """

    result = []
    for item in items:
        if isinstance(item, str):
            result.append(''.join(f"\\u{ord(ch):04x}" for ch in item))
        elif isinstance(item, int):
            result.append(f"\\u{item:04x}")
        else:
            raise TypeError(f"Unsupported type in to_unicode_codepoints: {type(item)}")
    return ''.join(result)
