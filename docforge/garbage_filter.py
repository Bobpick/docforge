"""Reject shredded tables and clean PDF extraction garbage.

Priority: drop noise aggressively. Prefer missing a real table over
emitting letter-soup grids that destroy body text for RAG/LLM use.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Table garbage detection ──────────────────────────────────────────

def _cells(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out: List[str] = [str(h or "") for h in headers]
    for r in rows:
        out.extend(str(c or "") for c in r)
    return out


_KNOWN_SHORT = {
    "min", "max", "typ", "avg", "std", "var", "mhz", "ghz", "khz", "hz",
    "dbm", "db", "v", "mv", "ma", "ua", "na", "pa", "rf", "if", "dc",
    "ac", "bw", "tx", "rx", "id", "no", "yes", "n/a", "na", "nil", "usd",
    "eur", "qty", "pct", "pdf", "doi", "ieee", "s21", "s11", "s22", "s12",
    "oip3", "iip3", "p1db", "nf", "vs", "to", "of", "or", "and", "the",
    "a", "b", "c", "d", "e", "x", "y", "z",  # common column labels
}


def is_fragment_cell(c: str) -> bool:
    """True for mid-word shards from multi-column PDF shredding."""
    c = (c or "").strip()
    if not c:
        return False
    if re.fullmatch(r"[-+]?\d+(\.\d+)?%?", c):
        return False
    if c.lower() in _KNOWN_SHORT:
        return False
    # Mid-word hyphenation leftovers: "tightly-" / "y-"
    if c.endswith("-") or (c.startswith("-") and len(c) <= 4):
        return True
    # Lowercase short shards: "led", "ola", "up", "en"
    if len(c) <= 3 and c.isalpha() and c.islower():
        return True
    # Single uppercase letter alone is often title shred ("A | RMY") — still
    # counted by caller when many appear; not always a fragment alone.
    if len(c) == 1 and c.isalpha() and c.isupper():
        return True
    # Mixed junk shards without vowels (except short known)
    if 2 <= len(c) <= 4 and c.isalpha() and not any(v in c.lower() for v in "aeiou"):
        return True
    return False


def fragmentation_ratio(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> float:
    """Share of non-empty cells that look like split word pieces."""
    cells = [c.strip() for c in _cells(headers, rows) if c and str(c).strip()]
    if not cells:
        return 1.0
    frag = sum(1 for c in cells if is_fragment_cell(c))
    return frag / len(cells)


def empty_ratio(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> float:
    cells = _cells(headers, rows)
    if not cells:
        return 1.0
    return sum(1 for c in cells if not str(c).strip()) / len(cells)


def looks_like_prose_shred(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> bool:
    """Body prose forced into many columns with mid-word breaks."""
    width = max(len(headers), max((len(r) for r in rows), default=0))
    if width < 6:
        return False
    cells = [c.strip() for c in _cells(headers, rows) if c and str(c).strip()]
    if len(cells) < 8:
        return False
    # Many cells end mid-word (lowercase letter, no punctuation) and next continues
    mid_word = sum(
        1
        for c in cells
        if len(c) >= 3
        and c[-1].islower()
        and not c[-1] in ".,;:!?)"
        and " " not in c
    )
    if mid_word / len(cells) >= 0.35:
        return True
    # Join all cells: if it reads like continuous English with broken spaces
    joined = " ".join(cells)
    # High ratio of single-letter tokens
    tokens = joined.split()
    if tokens:
        singles = sum(1 for t in tokens if len(t) == 1 and t.isalpha())
        if singles / len(tokens) >= 0.25:
            return True
    return False


def is_garbage_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    min_score: float = 8.0,
) -> bool:
    """Return True if this table should be discarded entirely."""
    from .table_utils import score_table

    headers = list(headers or [])
    rows = [list(r) for r in (rows or [])]

    width = max(len(headers), max((len(r) for r in rows), default=0))
    n_rows = len(rows)

    if width == 0 or (n_rows == 0 and sum(1 for h in headers if str(h).strip()) < 2):
        return True

    # Too wide with few rows → almost always layout shred
    if width >= 12 and n_rows <= 4:
        return True
    if width >= 18:
        return True

    frag = fragmentation_ratio(headers, rows)
    empty = empty_ratio(headers, rows)

    # Sparse min/typ/max datasheets are often ~30–50% empty — only kill emptier
    if empty >= 0.65 and width >= 4:
        return True
    if empty >= 0.80:
        return True
    # Fragmentation: strict on wide grids, looser on normal 2–5 col tables
    if width >= 8 and frag >= 0.25:
        return True
    if width >= 5 and frag >= 0.38:
        return True
    if frag >= 0.55:
        return True
    if looks_like_prose_shred(headers, rows):
        return True

    # Letter-spaced title rows: "A | RMY | OP | E | N"
    titleish = " ".join(str(h) for h in headers)
    if re.search(r"(?:\b[A-Z]\b\s*){4,}", titleish):
        return True
    row0 = " ".join(str(c) for c in (rows[0] if rows else []))
    if re.search(r"(?:\b[A-Z]\b\s*){5,}", row0):
        return True

    # Score gate (uses tightened scorer)
    if score_table(list(headers), rows) < min_score:
        return True

    return False


def filter_tables(
    tables: List[Dict[str, Any]],
    *,
    min_score: float = 8.0,
) -> List[Dict[str, Any]]:
    """Keep only non-garbage tables."""
    kept: List[Dict[str, Any]] = []
    for t in tables:
        headers = t.get("headers") or []
        rows = t.get("rows") or []
        if is_garbage_table(headers, rows, min_score=min_score):
            continue
        kept.append(t)
    return kept


def strip_garbage_markdown_tables(markdown: str) -> str:
    """Remove markdown table blocks that fail quality checks."""
    if not markdown or "|" not in markdown:
        return markdown

    lines = markdown.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip().startswith("|"):
            out.append(line)
            i += 1
            continue

        # Collect contiguous table lines
        block: List[str] = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            block.append(lines[i])
            i += 1
        if _md_table_is_garbage(block):
            # drop block (and following blank if any — handled by clean_whitespace later)
            continue
        out.extend(block)

    return "\n".join(out)


def _md_table_is_garbage(block: List[str]) -> bool:
    rows_raw = [ln for ln in block if ln.strip().startswith("|")]
    if len(rows_raw) < 2:
        return True
    parsed: List[List[str]] = []
    for ln in rows_raw:
        if re.match(r"^\|[\s\-:|]+\|$", ln.strip()):
            continue  # separator
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        parsed.append(cells)
    if not parsed:
        return True
    headers, body = parsed[0], parsed[1:]
    return is_garbage_table(headers, body)


# ── Text cleanup (non-table garbage) ─────────────────────────────────

def defragment_bold(text: str) -> str:
    """Join **word** **word** runs into **word word** (IEEE PDF span bold)."""
    if "**" not in text:
        return text

    # Repeated: **a** **b** → **a b** (same line)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(
            r"\*\*([^*]+?)\*\*(\s+)\*\*([^*]+?)\*\*",
            r"**\1\2\3**",
            text,
        )
    # Clean bold that wraps only punctuation
    text = re.sub(r"\*\*([,.;:]+)\*\*", r"\1", text)
    return text


def reflow_hard_wraps(text: str) -> str:
    """Join consecutive short PDF line-wraps into paragraphs.

    Conservative: only merge when both lines look like prose (not lists/tables/headers).
    """
    lines = text.split("\n")
    if not lines:
        return text

    def is_prose_line(ln: str) -> bool:
        s = ln.strip()
        if not s or len(s) < 20:
            return False
        if s.startswith(("#", "|", "```", ">", "-", "*", "$$")):
            return False
        if re.match(r"^\d+[\.\)]\s", s):
            return False
        if re.match(r"^Table\s+\d+", s, re.I):
            return False
        if re.match(r"^Figure\s+\d+", s, re.I):
            return False
        if re.match(r"^Rev\.\s*[A-Z]\s*\|", s, re.I):
            return False
        # Prefer lines that don't end a sentence hard
        return True

    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if not is_prose_line(cur):
            out.append(cur)
            i += 1
            continue
        # Merge following prose wraps
        buf = cur.rstrip()
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                break
            if not is_prose_line(nxt):
                break
            # Don't merge if current ends with sentence terminator and next starts capital
            # after blank — but we're mid-wrap: if cur is short-ish and nxt continues
            if len(buf) > 120 and buf[-1] in ".!?" and nxt.lstrip()[:1].isupper():
                break
            # Mid-word hyphenation: "tightly-\ncoupled" → "tightly-coupled"
            if buf.endswith("-"):
                buf = buf + nxt.lstrip()
            else:
                buf = buf + " " + nxt.lstrip()
            i += 1
            if len(buf) > 2000:
                break
        out.append(buf)
    return "\n".join(out)


def strip_page_chrome_lines(text: str) -> str:
    """Extra line-level junk not always caught by ArtifactRemover."""
    patterns = [
        r"^Rev\.\s*[A-Z]\s*\|\s*Page\s+\d+\s+of\s+\d+\s*$",
        r"^Page\s+\d+\s+of\s+\d+\s*$",
        r"^\d{5}-\d{3}\s*$",  # figure codes like 06879-001
        r"^Authorized\s+licensed\s+use\b.*$",
        r"^Downloaded\s+on\b.*$",
        r"^Restrictions\s+apply\.\s*$",
    ]
    compiled = [re.compile(p, re.I) for p in patterns]
    out = []
    for ln in text.split("\n"):
        if any(c.match(ln.strip()) for c in compiled):
            continue
        out.append(ln)
    return "\n".join(out)


def clean_extracted_markdown(markdown: str) -> str:
    """Full post-pass: drop garbage tables, fix bold, reflow, chrome."""
    md = strip_garbage_markdown_tables(markdown)
    md = defragment_bold(md)
    md = reflow_hard_wraps(md)
    md = strip_page_chrome_lines(md)
    # Collapse 3+ blanks
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def clean_section_content(section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Filter/clean a structured section; return None to drop."""
    sec_type = section.get("type")
    if sec_type == "table":
        headers = section.get("headers") or []
        rows = section.get("rows") or []
        if is_garbage_table(headers, rows):
            return None
        return section
    if sec_type in ("paragraph", "heading") and "content" in section:
        c = section["content"]
        if isinstance(c, str):
            c = defragment_bold(c)
            c = strip_page_chrome_lines(c)
            # Drop near-empty or pure fragment paragraphs
            if len(c.strip()) < 2:
                return None
            section = {**section, "content": c}
    return section
