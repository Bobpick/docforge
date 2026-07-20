"""Ligature repair, OCR word fixes, and code-vs-drawing heuristics."""

from __future__ import annotations

import re
from typing import Dict, List, Match, Tuple


# Explicit Unicode ligature codepoints → ASCII
LIGATURE_CHARS: Dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}

# Dropped-ligature / partial-word repairs (safe, low false-positive)
# Applied with IGNORECASE; replacement casing is normalized separately.
SAFE_LIGATURE_FIXES: List[Tuple[str, str]] = [
    (r"\bnding\b", "finding"),
    (r"\bnancial\b", "financial"),
    (r"\bnance\b", "finance"),
    (r"\bnished\b", "finished"),
    (r"\bnish\b", "finish"),
    (r"\bnal\b", "final"),
    (r"\brst\b", "first"),
    (r"\beld\b", "field"),
    (r"\bnger\b", "finger"),
    (r"\btted\b", "fitted"),
    (r"\btting\b", "fitting"),
    (r"\bexible\b", "flexible"),
    (r"\boating\b", "floating"),
    (r"\bush\b", "flush"),
    (r"\buid\b", "fluid"),
    (r"\biciency\b", "efficiency"),
    (r"\bicient\b", "efficient"),
    (r"\bective\b", "effective"),
    (r"\bectively\b", "effectively"),
    (r"electried\b", "electrified"),
    (r"electrication\b", "electrification"),
    (r"\bve-year\b", "five-year"),
    # Silent / similar (OCR often drops i/l inside short words)
    (r"\bslent\b", "silent"),
    (r"\bslence\b", "silence"),
    (r"\bslently\b", "silently"),
]

# Whole-word OCR / dropped-glyph fixes (case-insensitive keys).
# Intentionally conservative — only high-confidence technical/patent terms.
HIGH_CONFIDENCE_OCR = {
    "slent": "silent",
    "slence": "silence",
    "slently": "silently",
    "electried": "electrified",
    "electrication": "electrification",
    "nancial": "financial",
    "nance": "finance",
    "nding": "finding",
    "rst": "first",
    "eld": "field",
    "exible": "flexible",
    "iciency": "efficiency",
    "icient": "efficient",
    "ective": "effective",
    "flgure": "figure",
    "clalm": "claim",
    "clalms": "claims",
    "patcnt": "patent",
    "patcnts": "patents",
    "inventlon": "invention",
    "embodlment": "embodiment",
    "embodlments": "embodiments",
    "applicatlon": "application",
    "freqnency": "frequency",
    "frequcncy": "frequency",
    "amplifer": "amplifier",
    "ampliier": "amplifier",
    "transmltter": "transmitter",
    "receivcr": "receiver",
    "receivor": "receiver",
    "capacitcr": "capacitor",
    "capacltor": "capacitor",
    "circnit": "circuit",
    "voltagc": "voltage",
    "systcm": "system",
    "systern": "system",
    "devlce": "device",
    "devlces": "devices",
    "methcd": "method",
    "diagiam": "diagram",
    "dlagram": "diagram",
    "comprlsing": "comprising",
    "accordlng": "according",
    "inc1uding": "including",
    "signa1": "signal",
    "outpu1": "output",
    "outpnt": "output",
    "inpnt": "input",
    "fiilter": "filter",
    "fiilters": "filters",
    "cnrrent": "current",
    "crrrent": "current",
    "procese": "process",
    "procees": "process",
}

# Alias for callers/docs
OCR_WORD_FIXES = HIGH_CONFIDENCE_OCR


def _match_case(src: str, replacement: str) -> str:
    """Preserve ALL-CAPS / Title / lower casing of the original token."""
    if not src:
        return replacement
    if src.isupper():
        return replacement.upper()
    if src[0].isupper() and src[1:].islower():
        return replacement.capitalize()
    if src.islower():
        return replacement.lower()
    # Mixed / weird — prefer lowercase replacement with first char of src
    if src[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def fix_ligatures_and_ocr(text: str) -> str:
    """Apply Unicode ligatures, safe dropped-ligature regexes, and OCR word map."""
    if not text:
        return text

    for lig, repl in LIGATURE_CHARS.items():
        text = text.replace(lig, repl)

    for pattern, replacement in SAFE_LIGATURE_FIXES:
        def _sub(m: Match[str], repl: str = replacement) -> str:
            return _match_case(m.group(0), repl)

        text = re.sub(pattern, _sub, text, flags=re.IGNORECASE)

    def word_fix(m: Match[str]) -> str:
        word = m.group(0)
        key = word.lower()
        if key in HIGH_CONFIDENCE_OCR:
            return _match_case(word, HIGH_CONFIDENCE_OCR[key])
        return word

    text = re.sub(r"\b[A-Za-z][A-Za-z0-9']{2,}\b", word_fix, text)
    return text


def looks_like_drawing_label(text: str) -> bool:
    """True for patent drawing callouts / schematic labels (not source code)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    if len(lines) > 12:
        return False

    # Strong code signals → not a drawing label
    if re.search(
        r"[{};]|::|->|=>|</?\w+>|\b(def|class|function|return|import|include|const|var|let|void)\b",
        text,
    ):
        return False

    short = all(len(ln) <= 48 for ln in lines)
    if not short:
        return False

    # Mostly ALL CAPS words (OUTPUT BANDPASS FILTER)
    caps_lines = 0
    for ln in lines:
        letters = re.sub(r"[^A-Za-z]", "", ln)
        if not letters:
            continue
        if letters.isupper() and len(letters) >= 2:
            caps_lines += 1
        elif re.fullmatch(r"[A-Z0-9][A-Z0-9 \-_/]{1,}", ln):
            caps_lines += 1

    if caps_lines >= max(1, len(lines) * 0.6) and len(text) < 200:
        return True

    # Scattered single-char / pipe drawing fragments
    words = text.split()
    if words and len(text) < 60:
        single = sum(1 for w in words if len(w) == 1)
        if single / len(words) >= 0.5:
            return True

    return False


def looks_like_code(text: str, is_mono_font: bool = False) -> bool:
    """Decide whether a text block should be fenced as a code block."""
    stripped = text.strip()
    if not stripped:
        return False

    if looks_like_drawing_label(stripped):
        return False

    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        return False

    score = 0

    # Language / syntax signals
    if re.search(r"[{};]", stripped):
        score += 2
    if re.search(r"::|->|=>", stripped):
        score += 2
    if re.search(
        r"\b(def|class|function|return|import|from|const|var|let|void|int|float|public|private|package)\b",
        stripped,
    ):
        score += 2
    if re.search(r"[=<>]=|&&|\|\||<<|>>", stripped):
        score += 1
    if re.search(r"^\s*#include\b|^\s*using\s+namespace\b", stripped, re.M):
        score += 2
    if re.search(r"</?[A-Za-z][^>]*>", stripped):
        score += 1

    # Indentation variety (real code, not a label)
    indented = sum(1 for ln in lines if ln.startswith((" ", "\t")))
    if len(lines) >= 3 and indented >= 2:
        score += 1

    # Many non-alnum characters typical of code
    non_alnum = sum(1 for c in stripped if not c.isalnum() and not c.isspace())
    if len(stripped) > 20 and non_alnum / len(stripped) > 0.18:
        score += 1

    # Mono font alone is weak — need length + structure
    if is_mono_font:
        if len(lines) >= 4 and len(stripped) >= 40:
            score += 1
        elif len(lines) >= 2 and score >= 1:
            score += 1

    # Short mono blocks that are plain English → paragraph
    if is_mono_font and len(lines) <= 2 and score < 2:
        alpha = sum(1 for c in stripped if c.isalpha())
        if alpha / max(len(stripped), 1) > 0.7:
            return False

    need = 2 if is_mono_font else 3
    return score >= need


def collapse_spaced_text(text: str) -> str:
    """Collapse spaced-out form text like 'P r o p o s a l   S u m m a r y'."""

    def _collapse_spaced_word(spaced: str) -> str:
        chars = spaced.replace(" ", "")
        if len(chars) <= 2:
            return spaced
        if chars.isupper():
            return chars.capitalize()
        return chars

    lines = text.split("\n")
    result_lines = []
    spaced_char_group = r"(?:\w[^\S\n]{1,2})"
    spaced_word_re = spaced_char_group + r"{3,}\w"
    full_pattern = spaced_word_re + r"(?:[^\S\n]{3,}" + spaced_word_re + r")*"

    for line in lines:

        def replace_spaced_run(m: Match[str]) -> str:
            run = m.group(0)
            spaced_words = re.split(r"[^\S\n]{3,}", run)
            collapsed = [_collapse_spaced_word(w) for w in spaced_words if w.strip()]
            return " ".join(collapsed)

        line = re.sub(full_pattern, replace_spaced_run, line)
        result_lines.append(line)

    return "\n".join(result_lines)
