"""Canonical written-form normalization chain for token-level CER.

Design goal: be insensitive to different WRITTEN FORMS of the same speech
while never forgiving a real recognition error. Numbers are canonicalized
into the written space with inverse text normalization (ITN, spoken form ->
written form), which is a many-to-one mapping -- e.g. 2024, 二零二四 and
两千零二十四 all collapse to the same canonical string -- unlike TN
(written -> spoken), which must pick one reading and mis-scores the others.

Chain (``normalize_text``):

  NFKC + lowercase
  -> mask numeral-bearing idioms / 百分之 / unit words so ITN cannot mangle
     them or absorb them into adjacent numbers
  -> cn2an ITN (spoken -> written canonical)
  -> span-wise second pass for CJK numeral runs cn2an left behind
     (万/亿 cut points preferred; a dangling 点 is never swallowed)
  -> 百分之X -> X%
  -> mixed written-number expansion (4500亿 -> 450000000000, exact Decimal)
  -> punctuation replaced by spaces, never deleted (no ASCII gluing);
     % $ ¥ ° and digit-context . / - survive; letter-internal apostrophes
     are deleted (i'm -> im)

Tokenization (``tokenize``): CJK one token per character, latin letter runs
one token per word, digits one token per character, surviving symbols one
token each. ``norm_tokens_full`` is the chain composed with tokenization.

Determinism note: results depend on the pinned ``cn2an`` version. The engine
is REQUIRED -- there is deliberately no silent fallback; a missing dependency
raises instead of quietly changing the metric. Per-string cn2an failures fall
back to the unconverted string (deterministic for a given cn2an version) and
are counted in ``itn_fallback_count``.

Known limitations (kept as-is, to be revised only with a rules-version bump):
clock times (两点半 vs 2:30) and unit lexemes (千克 vs kg) are not unified;
approximate numerals such as 七八个 may be merged with 78个 by ITN; filler
words splitting a numeral run cause per-segment conversion; cn2an context
quirks can convert only one side near a real error.
"""

from __future__ import annotations

import re
import string
import unicodedata
from decimal import Decimal
from functools import lru_cache

RULES_VERSION = "v1"

_cn2an = None
_ITN_FALLBACKS = 0


def _require_cn2an():
    global _cn2an
    if _cn2an is None:
        try:
            import warnings

            import cn2an

            warnings.filterwarnings("ignore", module=r"cn2an.*")
            _cn2an = cn2an
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "normalization/canonical_itn requires the 'cn2an' package. "
                "Install it with: pip install \"sure-evaluation[canonical]\" "
                "(or check the node with: sure-eval env check --node normalization/canonical_itn). "
                "There is no silent fallback: a missing engine would change the metric."
            ) from exc
    return _cn2an


def engine_info() -> dict:
    cn2an = _require_cn2an()
    return {
        "engine": "cn2an",
        "engine_version": getattr(cn2an, "__version__", "unknown"),
        "rules_version": RULES_VERSION,
    }


def itn_fallback_count() -> int:
    """Number of cn2an transform calls that fell back to the raw string."""

    return _ITN_FALLBACKS


def fold(s: str) -> str:
    """NFKC (full-width -> half-width, etc.) plus lowercase."""

    return unicodedata.normalize("NFKC", s).lower()


# --------------------------------------------------------------------------- #
# Pre-ITN masking: numeral-bearing idioms, 百分之, unit words
# --------------------------------------------------------------------------- #
_IDIOMS_PRE = ["一五一十", "一心一意", "三三两两", "二话不说", "说一不二",
               "五花八门", "乱七八糟", "一模一样", "一起", "一样", "一下",
               "一些", "一直", "一定", "一般", "一致", "万一", "统一", "唯一"]
_NUMCTX = "零〇一二三四五六七八九十百千万亿点两幺0-9"
# Context guards: an idiom match may not sit inside a numeral run, otherwise
# 万一 would fire inside 一千万一千万 and 统一 inside 系统一千万.
_IDIOM_RE = re.compile(f"(?<![{_NUMCTX}])(" + "|".join(_IDIOMS_PRE) + f"|十分(?!钟))(?![{_NUMCTX}])")
_MASK = "\ue000"        # private-use placeholders pass through cn2an untouched
_PCT_MASK = "\ue001"
_UNITW_MASK = "\ue002"
_UNITW_RE = re.compile(r"(千米|千克|千瓦|千卡|千帕|千斤)(?![隆拉])")


def _mask_idioms(s: str):
    hits: list[str] = []

    def rep(m):
        hits.append(m.group(0))
        return f"{_MASK}{len(hits) - 1}{_MASK}"

    return _IDIOM_RE.sub(rep, s), hits


def _unmask_idioms(s: str, hits) -> str:
    for i, w in enumerate(hits):
        s = s.replace(f"{_MASK}{i}{_MASK}", w)
    return s


def _mask_pct_units(s: str):
    unit_hits: list[str] = []

    def urep(m):
        unit_hits.append(m.group(0))
        return f"{_UNITW_MASK}{len(unit_hits) - 1}{_UNITW_MASK}"

    s = _UNITW_RE.sub(urep, s)
    return s.replace("百分之", _PCT_MASK), unit_hits


def _unmask_pct_units(s: str, unit_hits) -> str:
    s = s.replace(_PCT_MASK, "百分之")
    for i, w in enumerate(unit_hits):
        s = s.replace(f"{_UNITW_MASK}{i}{_UNITW_MASK}", w)
    return s


# --------------------------------------------------------------------------- #
# ITN: cn2an spoken -> written transform
# --------------------------------------------------------------------------- #
def _itn_transform(s: str) -> str:
    global _ITN_FALLBACKS
    cn2an = _require_cn2an()
    try:
        return cn2an.transform(s, "cn2an")
    except Exception:
        _ITN_FALLBACKS += 1
        return s


# --------------------------------------------------------------------------- #
# Pass 2: span-wise conversion of CJK numeral runs cn2an left behind
# --------------------------------------------------------------------------- #
_CJK_NUM = "零〇一二三四五六七八九十百千万亿点两幺"
_SPAN = re.compile(f"[{_CJK_NUM}]{{2,}}")
_MAX_NUM = 16   # longest sensible CJK numeral; bounds the worst case to O(L*16)


def _parse_num(seg: str):
    cn2an = _require_cn2an()
    try:
        v = cn2an.cn2an(seg, "smart")
    except Exception:
        return None
    return str(int(v)) if float(v).is_integer() else str(v)


@lru_cache(maxsize=100000)
def _conv_span(span: str) -> str:
    out, i, n = [], 0, len(span)   # out: (is_num, text)
    while i < n:
        hit = None
        lim = min(n, i + _MAX_NUM)
        # Natural-reading preference: a number ends after 万/亿 when more
        # digits follow; try those cut points longest-first, then greedy.
        cuts = [j for j in range(i + 1, lim) if span[j - 1] in "万亿"]
        for j in sorted(cuts, reverse=True) + list(range(lim, i, -1)):
            seg = span[i:j]
            if seg.endswith("点"):      # 两点半: never let cn2an eat a dangling 点
                continue
            v = _parse_num(seg)
            if v is not None:
                hit = (j, v)
                break
        if hit is None:
            out.append((False, span[i]))
            i += 1
        else:
            j, v = hit
            out.append((True, v))
            i = j
    # Space ONLY between two adjacent converted numbers; unconverted chars
    # (residual units 万/亿/千, dangling 点, ...) attach directly so
    # _expand_mixed still sees 1.8万 / 10万亿 as one unit.
    parts = []
    for k, (isnum, t) in enumerate(out):
        if k and isnum and out[k - 1][0]:
            parts.append(" ")
        parts.append(t)
    return "".join(parts)


def _span_pass(s: str) -> str:
    return _SPAN.sub(lambda m: _conv_span(m.group(0)), s)


# --------------------------------------------------------------------------- #
# 百分之X -> X%
# --------------------------------------------------------------------------- #
_PCT = re.compile(r"百分之([零〇一二三四五六七八九十百千点两]+|\d+(?:\.\d+)?)")


def _pct_pass(s: str) -> str:
    def pct(m):
        body = m.group(1)
        if body == "百":
            return "100%"
        if body[0].isdigit():
            return f"{body}%"
        rest = m.string[m.end():]
        for k in range(len(body), 0, -1):
            if (body[k:] + rest).startswith("分之"):
                continue    # would break a following 百分之X
            v = _parse_num(body[:k])
            if v is not None:
                return f"{v}%{body[k:]}"
        return m.group(0)

    prev = None
    while prev != s:
        prev, s = s, _PCT.sub(pct, s)
    return s


# --------------------------------------------------------------------------- #
# Mixed written-number expander: 4500亿 / 1.8万 / 90个亿 / 3千 -> pure digits
# --------------------------------------------------------------------------- #
_MIX = re.compile(r"(\d+(?:\.\d+)?)个?(万亿|亿|万|千(?![米克瓦卡帕斤]))")
_UNIT = {"万亿": 10**12, "亿": 10**8, "万": 10**4, "千": 10**3}


def _expand_mixed(s: str) -> str:
    def rep(m):
        v = Decimal(m.group(1)) * _UNIT[m.group(2)]
        return str(int(v)) if v == v.to_integral_value() else str(v)

    prev = None
    while prev != s:            # 3万亿 resolves over two passes
        prev, s = s, _MIX.sub(rep, s)
    return s


# --------------------------------------------------------------------------- #
# Punctuation -> space, with semantic-symbol protection
# --------------------------------------------------------------------------- #
_PUNCT_SET = set(string.punctuation) | {
    '，', '。', '！', '？', '：', '；', '、', '（', '）',
    '“', '”', '‘', '’', '【', '】', '《', '》', '…', '\\',
}
_PROTECT = set('%$¥°')          # NFKC folds ％ -> %, ￥ -> ¥, ℃ -> °C


def _strip_punct_space(s: str) -> str:
    out = []
    for k, ch in enumerate(s):
        try:
            unicodedata.name(ch)
        except ValueError:
            continue
        if not ch.isprintable():
            continue
        if ch == '.' and 0 < k < len(s) - 1 and s[k - 1].isdigit() and s[k + 1].isdigit():
            out.append(ch)      # digit-context decimal point survives (41.3 != 413)
        elif ch in "'’" and 0 < k < len(s) - 1 and s[k - 1].isalpha() and s[k + 1].isalpha():
            continue            # i'm -> im (delete, do not split)
        elif ch == '-' and k < len(s) - 1 and s[k + 1].isdigit():
            out.append(ch)      # minus sign: -3度 must stay an error vs 3度
        elif ch in _PROTECT:
            out.append(ch)
        elif ch in _PUNCT_SET:
            out.append(' ')     # replace, never delete: no ASCII gluing
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------- #
# Tokenizer: CJK char / latin word / digit chars / symbol chars
# --------------------------------------------------------------------------- #
def tokenize(s: str) -> list[str]:
    toks: list[str] = []
    run: list[str] = []

    def flush():
        if run:
            toks.append("".join(run))
            run.clear()

    for ch in s:
        cat = unicodedata.category(ch)
        if ch.isspace() or cat in ("Zs", "Cn"):
            flush()
        elif cat == "Lo":                    # CJK etc.: one token per char
            flush()
            toks.append(ch)
        elif ch.isdigit():                   # digits split per char
            flush()
            toks.append(ch)
        elif ch.isalpha():                   # latin letter run = one word token
            run.append(ch)
        else:                                # %, ., $, °, ¥, - ...: own token
            flush()
            toks.append(ch)
    flush()
    return toks


# --------------------------------------------------------------------------- #
# Public entrypoints
# --------------------------------------------------------------------------- #
def normalize_text(s: str) -> str:
    """The full canonicalization chain, text -> normalized text."""

    if not s:
        return ""
    s = fold(s)
    s, hits = _mask_idioms(s)
    s, uhits = _mask_pct_units(s)      # ITN never sees 百分之 / unit words
    s = _itn_transform(s)              # cn2an spoken -> written (many-to-one)
    s = fold(s)                        # cn2an may emit unfolded characters
    s = _span_pass(s)                  # leftover CJK numerals -> digits
    s = _unmask_pct_units(s, uhits)
    s = _pct_pass(s)                   # 百分之<digits> -> <digits>%
    s = _expand_mixed(s)
    s = _unmask_idioms(s, hits)
    return _strip_punct_space(s)


def normalize_text_no_numnorm(s: str) -> str:
    """Same chain minus all number canonicalization (inspection toggle)."""

    if not s:
        return ""
    return _strip_punct_space(fold(s))


def norm_tokens_full(s: str) -> tuple[str, ...]:
    """Full chain composed with tokenization -> scoring tokens."""

    if not s:
        return ()
    return tuple(tokenize(normalize_text(s)))


def norm_tokens_no_numnorm(s: str) -> tuple[str, ...]:
    if not s:
        return ()
    return tuple(tokenize(normalize_text_no_numnorm(s)))
