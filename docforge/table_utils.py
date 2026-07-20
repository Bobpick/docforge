"""Table extraction quality helpers for PDF form/layout tables."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


# pdfplumber strategies to try (order does not matter; we score results)
TABLE_SETTINGS: List[Optional[Dict[str, Any]]] = [
    None,  # library default
    {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
        "snap_tolerance": 3,
    },
    {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
        "text_tolerance": 3,
        "text_x_tolerance": 3,
        "text_y_tolerance": 3,
    },
    {
        "vertical_strategy": "lines",
        "horizontal_strategy": "text",
        "min_words_horizontal": 1,
        "intersection_tolerance": 5,
    },
    {
        "vertical_strategy": "text",
        "horizontal_strategy": "lines",
        "min_words_vertical": 1,
        "intersection_tolerance": 5,
    },
    {
        # Form-style SBIR budgets: loose text clustering
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
        "text_x_tolerance": 5,
        "text_y_tolerance": 2,
        "snap_x_tolerance": 5,
        "snap_y_tolerance": 3,
    },
]


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\n", " ")).strip()


def normalize_table(raw: Sequence[Sequence[Any]]) -> Optional[Tuple[List[str], List[List[str]]]]:
    """Normalize a raw table into headers + rows; drop empty junk."""
    if not raw or len(raw) < 1:
        return None

    matrix = [[clean_cell(c) for c in row] for row in raw if row is not None]
    # Drop fully empty rows
    matrix = [row for row in matrix if any(cell for cell in row)]
    if not matrix:
        return None

    # Pad to consistent width
    width = max(len(r) for r in matrix)
    matrix = [r + [""] * (width - len(r)) for r in matrix]

    # If first row is empty-ish, treat whole as body with blank headers
    headers = matrix[0]
    rows = matrix[1:] if len(matrix) > 1 else []

    # Single-row "table" is usually noise unless multi-column with content
    if not rows and sum(1 for h in headers if h) < 2:
        return None

    return headers, rows


def score_table(headers: List[str], rows: List[List[str]]) -> float:
    """Higher is better. Penalize 1-column form dumps and empty grids."""
    if not headers and not rows:
        return -100.0

    width = max(len(headers), max((len(r) for r in rows), default=0))
    if width == 0:
        return -100.0

    all_rows = [headers] + rows
    lengths = [len(r) for r in all_rows]
    consistency = 1.0 - (max(lengths) - min(lengths)) / max(width, 1)

    cells = [c for r in all_rows for c in r]
    nonempty = sum(1 for c in cells if c)
    fill = nonempty / max(len(cells), 1)

    # Prefer multi-column tables
    col_score = min(width, 10) * 3.0
    row_score = min(len(rows), 40) * 0.15

    # Heavy penalty for single-column tables with many rows (form layout fail)
    one_col_penalty = 0.0
    if width == 1 and len(rows) >= 2:
        one_col_penalty = 12.0 + min(len(rows), 20) * 0.5

    # Bonus if header cells look like real labels
    header_bonus = 0.0
    if any(headers):
        short_labels = sum(1 for h in headers if 1 <= len(h) <= 40)
        header_bonus = short_labels * 0.4

    # Penalty if almost everything is empty
    if fill < 0.15:
        return -50.0

    return col_score + row_score + consistency * 4.0 + fill * 6.0 + header_bonus - one_col_penalty


def split_aligned_cell(cell: str) -> List[str]:
    """Split a single form cell that actually holds multiple columns."""
    if not cell:
        return [""]

    # Tabs
    if "\t" in cell:
        parts = [p.strip() for p in cell.split("\t")]
        if len(parts) >= 2:
            return parts

    # 2+ spaces (common in pdfplumber text strategy failures)
    parts = re.split(r" {2,}", cell.strip())
    if len(parts) >= 2:
        return parts

    # Pipe-ish remnants
    if " | " in cell:
        parts = [p.strip() for p in cell.split(" | ")]
        if len(parts) >= 2:
            return parts

    return [cell.strip()]


def reconstruct_single_column_table(
    headers: List[str], rows: List[List[str]]
) -> Tuple[List[str], List[List[str]]]:
    """If a table collapsed to 1 column, try to rebuild columns from alignment.

    SBIR / government form tables often extract as N rows × 1 cell, with
    multi-field content jammed into each cell separated by spaces.
    """
    width = max(len(headers), max((len(r) for r in rows), default=0))
    if width != 1:
        return headers, rows

    # Collect candidate splits from header + body
    candidates: List[List[str]] = []
    header_cell = headers[0] if headers else ""
    if header_cell:
        candidates.append(split_aligned_cell(header_cell))
    for r in rows:
        cell = r[0] if r else ""
        candidates.append(split_aligned_cell(cell))

    # Need at least half the rows to split into the same column count >= 2
    multi = [c for c in candidates if len(c) >= 2]
    if len(multi) < max(2, len(candidates) // 2):
        return headers, rows

    # Modal column count among multi-splits
    counts: Dict[int, int] = {}
    for c in multi:
        counts[len(c)] = counts.get(len(c), 0) + 1
    target_cols = max(counts, key=counts.get)
    if target_cols < 2:
        return headers, rows

    def pad(parts: List[str]) -> List[str]:
        if len(parts) == target_cols:
            return parts
        if len(parts) == 1:
            # Put whole string in first col
            return parts + [""] * (target_cols - 1)
        if len(parts) > target_cols:
            # Merge extras into last column
            return parts[: target_cols - 1] + [" ".join(parts[target_cols - 1 :])]
        return parts + [""] * (target_cols - len(parts))

    rebuilt = [pad(c) for c in candidates]
    if header_cell:
        new_headers, new_rows = rebuilt[0], rebuilt[1:]
    else:
        new_headers = [f"Col{i+1}" for i in range(target_cols)]
        new_rows = rebuilt

    # Only accept if score improves
    if score_table(new_headers, new_rows) > score_table(headers, rows):
        return new_headers, new_rows
    return headers, rows


def pick_best_table(
    candidates: List[Tuple[List[str], List[List[str]]]]
) -> Optional[Tuple[List[str], List[List[str]]]]:
    """Pick highest-scoring table among candidates; apply 1-col reconstruction."""
    best = None
    best_score = -1e9
    for headers, rows in candidates:
        headers, rows = reconstruct_single_column_table(headers, rows)
        s = score_table(headers, rows)
        if s > best_score:
            best_score = s
            best = (headers, rows)
    if best is None or best_score < 0:
        return None
    return best


def tables_are_duplicates(
    a: Tuple[List[str], List[List[str]]],
    b: Tuple[List[str], List[List[str]]],
    threshold: float = 0.85,
) -> bool:
    """Detect near-duplicate tables (Google Patents citation tables xN)."""
    ha, ra = a
    hb, rb = b
    if [h.lower() for h in ha] != [h.lower() for h in hb]:
        # Allow empty-header match on body
        if any(ha) or any(hb):
            if [h.lower() for h in ha if h] != [h.lower() for h in hb if h]:
                return False

    def sig(rows: List[List[str]]) -> List[str]:
        return ["|".join(c.lower() for c in r) for r in rows]

    sa, sb = sig(ra), sig(rb)
    if not sa or not sb:
        return sa == sb
    # Jaccard on row signatures
    set_a, set_b = set(sa), set(sb)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return True
    return (inter / union) >= threshold


def dedupe_table_list(
    tables: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop near-duplicate tables within a document (cross-page chrome)."""
    unique: List[Dict[str, Any]] = []
    for t in tables:
        headers = t.get("headers") or []
        rows = t.get("rows") or []
        cur = (list(headers), [list(r) for r in rows])
        is_dup = False
        for u in unique:
            other = (list(u.get("headers") or []), [list(r) for r in (u.get("rows") or [])])
            if tables_are_duplicates(cur, other):
                is_dup = True
                break
        if not is_dup:
            unique.append(t)
    return unique
