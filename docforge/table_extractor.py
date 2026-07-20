"""
DocForge — Clean Table Extraction Module
=========================================

Multi-strategy PDF table extraction with quality filtering.

Replaces the old _extract_tables_pdfplumber() method in pdf_handler.py.
Drop this code into docforge/table_extractor.py and import:

    from docforge.table_extractor import extract_tables_from_pdf

Strategy:
  1. PyMuPDF find_tables() — primary (fast, best line detection)
  2. Camelot lattice — secondary (best for ruled tables, requires Ghostscript)
  3. pdfplumber — tertiary fallback only (most false positives)

Key innovations:
  - Quality scoring rejects chart/graph noise (S-parameter plots, etc.)
  - Source-specific thresholds (pdfplumber requires higher scores)
  - Garbled-header detection (pdfplumber column-splitting artifacts)
  - Multi-line cell expansion (spec sheet "Gain\\nOIP3\\nNF" → 3 rows)
  - Cell repair (broken subscripts: P_OUT, T_A)
  - Content-based deduplication (same table from different extractors)
  - Drawing-fragment rejection (dimension labels, pin diagrams)

Tested on: ADL5542.pdf (Analog Devices RF amplifier datasheet)
Result: 6 clean tables, 0 chart noise, 0 duplicates
"""

import re
from typing import List, Dict, Any, Tuple
from collections import Counter


# ═══════════════════════════════════════════════════════════════════
#  Common English/technical words for garbled-text detection
# ═══════════════════════════════════════════════════════════════════

_COMMON_WORDS = {
    # General table headers
    'parameter', 'test', 'conditions', 'comments', 'min', 'typ', 'max',
    'unit', 'frequency', 'gain', 'output', 'input', 'power', 'supply',
    'voltage', 'current', 'temperature', 'range', 'description', 'function',
    'component', 'value', 'default', 'model', 'package', 'option',
    'branding', 'pin', 'mnemonic', 'magnitude', 'angle', 'rating',
    # Technical terms
    'absolute', 'maximum', 'operating', 'storage', 'dissipation', 'noise',
    'figure', 'compression', 'intercept', 'isolation', 'return', 'loss',
    'interface', 'exposed', 'paddle', 'name', 'number', 'type', 'speed',
    'bandwidth', 'impedance', 's11', 's12', 's21', 's22', 'oip3', 'p1db',
    'vpos', 'rfout', 'rfin', 'capacitance', 'inductance', 'resistance',
    'material', 'dimension', 'weight', 'height', 'width', 'length',
    'depth', 'thickness', 'table', 'note', 'notes', 'symbol', 'typical',
    'characteristic', 'performance', 'specification', 'electrical',
    'characteristics', 'overview', 'revision', 'history', 'ordering',
    'guide', 'information', 'general', 'application', 'circuit',
}


# ═══════════════════════════════════════════════════════════════════
#  Quality scoring
# ═══════════════════════════════════════════════════════════════════

def table_quality_score(data: List[List[str]]) -> float:
    """Score 0–1 on likelihood this is a real data table (not noise).
    
    Factors (weighted):
    - Fill ratio (30%): % of cells with content
    - Meaningful ratio (30%): % of cells with >2 chars
    - Header quality (20%): % of headers with substantive text
    - Column consistency (20%): % of rows with same column count
    """
    if not data or len(data) < 2:
        return 0.0
    total = sum(len(r) for r in data)
    if total == 0:
        return 0.0
    
    fill = sum(1 for r in data for c in r if c and c.strip()) / total
    meaningful = sum(1 for r in data for c in r if c and len(c.strip()) > 2) / total
    headers = data[0]
    real_h = sum(1 for h in headers if h and len(h.strip()) > 2 and not h.strip().isdigit())
    h_ratio = real_h / len(headers) if headers else 0
    cols = [len(r) for r in data]
    mx = max(cols) if cols else 0
    consist = sum(1 for c in cols if c == mx) / len(cols) if cols else 0
    
    return fill * 0.30 + meaningful * 0.30 + h_ratio * 0.20 + consist * 0.20


# ═══════════════════════════════════════════════════════════════════
#  Garbled-text detection
# ═══════════════════════════════════════════════════════════════════

def _has_garbled_text(text: str) -> bool:
    """Detect garbled text from pdfplumber column splitting.
    
    Garbled text contains word fragments like "ipation" (from "dissipation"),
    "AXIMUM" (from "MAXIMUM"), "Supp" (from "Supply"), "ings" (from "RATINGS").
    
    Heuristic: extract alpha words, count how many are NOT in a common-words
    dictionary AND look like fragments (short uppercase, lowercase-starting,
    or common suffix-only).
    """
    words = re.findall(r'[A-Za-z]{2,}', text)
    if len(words) < 4:
        return False
    
    uncommon = 0
    for w in words:
        wl = w.lower()
        if wl in _COMMON_WORDS:
            continue
        if re.match(r'^[kMGT]Hz|dBm?|mW|VDC|ppm$', wl):
            continue
        # Short all-caps that isn't a common term (AXIMU, OS, etc.)
        if w.isupper() and len(w) <= 6 and wl not in _COMMON_WORDS:
            uncommon += 1
        # Lowercase-starting short word (ipation, upply, etc.)
        elif wl[0].islower() and len(wl) <= 7 and wl not in _COMMON_WORDS:
            uncommon += 1
        # Common suffix-only fragments
        elif re.match(r'^(ation|ment|ings|ity|ness|ence|ance|ible|able|ical)$', wl):
            uncommon += 1
    
    return uncommon / len(words) > 0.25 and uncommon >= 2


# ═══════════════════════════════════════════════════════════════════
#  Table validation
# ═══════════════════════════════════════════════════════════════════

def is_valid_table(data: List[List[str]], min_score: float = 0.45,
                   source: str = 'pymupdf') -> bool:
    """Validate extracted table data with source-specific thresholds.
    
    Filters applied:
    - Minimum quality score (higher for pdfplumber: 0.65, camelot: 0.50)
    - Must have substantive header cells (>2 chars, not just digits)
    - No garbled header text (pdfplumber column-splitting artifacts)
    - Not dominated by single-char words (pin diagrams, drawing labels)
    - Not a drawing fragment (dimension labels, package outlines)
    - First data row must have content
    """
    if not data or len(data) < 2:
        return False
    
    # Source-specific minimum score
    eff_min = {'pdfplumber': 0.65, 'camelot': 0.50}.get(
        source.split('-')[0], min_score)
    if table_quality_score(data) < eff_min:
        return False
    
    headers = data[0]
    real_h = sum(1 for h in headers if h and len(h.strip()) > 2 and not h.strip().isdigit())
    if real_h == 0:
        return False
    if source.startswith('pdfplumber') and real_h < 2:
        return False
    
    # Garbled header check
    all_header_text = ' '.join(h.strip() for h in headers if h)
    if _has_garbled_text(all_header_text):
        return False
    
    # Single-char-dominated content (pin diagrams, drawing labels)
    all_text = ' '.join(c.strip() for r in data for c in r if c)
    words = all_text.split()
    if len(words) > 5:
        short = sum(1 for w in words
                    if len(w) <= 2 and not re.match(r'^[±−+\d.]+$', w))
        if short / len(words) > 0.5:
            return False
    
    # Drawing fragments (dimension labels, package outlines)
    if len(data) <= 2 and len(headers) <= 3:
        combined = (' '.join(c.strip() for r in data[1:] for c in r if c)
                    + ' ' + ' '.join(c.strip() for c in headers if c))
        if len(re.findall(r'\d+\.\d+mm|PIN\s*\d+|EXPOSE|PAD\b', combined, re.I)) >= 2:
            return False
    
    # First data row must have content
    if len(data) > 1:
        row1_content = sum(1 for c in data[1] if c and len(c.strip()) > 1)
        if row1_content == 0 and len(headers) > 1:
            return False
    
    return True


# ═══════════════════════════════════════════════════════════════════
#  Cell repair
# ═══════════════════════════════════════════════════════════════════

def repair_table_cells(data: List[List[str]]) -> List[List[str]]:
    """Fix common PDF table extraction artifacts.
    
    Repairs:
    - None → "" (null cells)
    - "P )" → "P_OUT)" (broken subscript in spec sheets)
    - "T A" → "T_A" (ambient temperature subscript)
    """
    data = [[c or '' for c in r] for r in data]
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            cell = re.sub(r'\bP\s*\)', 'P_OUT)', cell)
            cell = re.sub(r'\bT\s*A\b(?!≥|≤)', 'T_A', cell)
            data[i][j] = cell
    return data


# ═══════════════════════════════════════════════════════════════════
#  Multi-row header detection
# ═══════════════════════════════════════════════════════════════════

def merge_split_headers(data: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """Merge multi-row headers if row 2 is complementary text, not data.
    
    If row 1 is mostly numeric → keep as data row.
    If row 1 is mostly text → merge with row 0 as sub-headers.
    """
    if len(data) < 2:
        return data[0] if data else [], data[1:]
    
    header = list(data[0])
    row1 = data[1]
    
    num = sum(1 for c in row1
              if c.strip() and re.match(r'^[±−+\d.,\s°µ%VΩW]+$', c.strip()))
    txt = sum(1 for c in row1
              if c.strip() and not re.match(r'^[±−+\d.,\s°µ%VΩW]+$', c.strip()))
    
    if txt > num and txt > 0:
        for j in range(min(len(header), len(row1))):
            if row1[j].strip():
                header[j] = (f"{header[j]} {row1[j]}".strip()
                             if header[j].strip() else row1[j])
        return header, data[2:]
    
    return header, data[1:]


# ═══════════════════════════════════════════════════════════════════
#  Multi-line cell expansion
# ═══════════════════════════════════════════════════════════════════

def expand_multiline_cells(data: List[List[str]],
                          min_common: int = 3) -> List[List[str]]:
    """Expand cells with \\n into multiple sub-rows.
    
    Spec sheets often pack multiple values in one cell:
      "Gain\\nOIP3\\nNF" | "20.9 dB\\n38 dBm\\n2.9 dB"
    → 3 separate rows, one per parameter.
    
    Only expands when most cells agree on the same line count
    (indicating aligned sub-rows).
    """
    if not data or len(data) < 2:
        return data
    
    if not any('\n' in c for r in data[1:] for c in r):
        return data
    
    header = data[0]
    out = []
    
    for row in data[1:]:
        line_counts = [c.count('\n') + 1 for c in row if c.strip()]
        if not line_counts:
            out.append(row)
            continue
        
        common = Counter(line_counts).most_common(1)[0][0]
        if common < min_common:
            out.append(row)
            continue
        
        agreeing = sum(1 for lc in line_counts if abs(lc - common) <= 1)
        if agreeing / len(line_counts) < 0.6:
            out.append(row)
            continue
        
        # Expand this row
        splits = [c.split('\n') for c in row]
        for s in splits:
            while len(s) < common:
                s.append('')
        for line_idx in range(common):
            out.append([s[line_idx] if line_idx < len(s) else ''
                       for s in splits])
    
    return [header] + out


# ═══════════════════════════════════════════════════════════════════
#  Deduplication
# ═══════════════════════════════════════════════════════════════════

def bboxes_overlap(b1, b2, threshold=0.5):
    """Check if two bounding boxes overlap significantly."""
    if not b1 or not b2:
        return False
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if x1 >= x2 or y1 >= y2:
        return False
    inter = (x2 - x1) * (y2 - y1)
    area = min((b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1]))
    return area > 0 and inter / area > threshold


def _tables_are_duplicates(t1, t2):
    """Are these two tables likely the same real table from different extractors?
    
    Checks: same page, similar column count, overlapping header/content words.
    """
    if t1['page'] != t2['page']:
        return False
    # Allow 1 column difference (Camelot sometimes adds a freq column)
    if abs(len(t1['headers']) - len(t2['headers'])) > 1:
        return False
    
    # Compare first data row content overlap
    if t1['rows'] and t2['rows']:
        r1_words = set(' '.join(c.strip().lower() for c in t1['rows'][0] if c).split())
        r2_words = set(' '.join(c.strip().lower() for c in t2['rows'][0] if c).split())
        if r1_words and r2_words:
            overlap = len(r1_words & r2_words) / max(len(r1_words), len(r2_words))
            if overlap > 0.3:
                return True
    
    # Compare header content
    stop_words = {'', 'of', 'the', 'a', 'and', 'or', 'for', 'in', 'to'}
    h1_words = set(' '.join(c.strip().lower() for c in t1['headers'] if c).split()) - stop_words
    h2_words = set(' '.join(c.strip().lower() for c in t2['headers'] if c).split()) - stop_words
    if h1_words and h2_words:
        overlap = len(h1_words & h2_words) / max(len(h1_words), len(h2_words))
        if overlap > 0.3:
            return True
    
    return False


# ═══════════════════════════════════════════════════════════════════
#  Camelot DataFrame fix
# ═══════════════════════════════════════════════════════════════════

def _fix_camelot_columns(data):
    """Fix Camelot DataFrames where column names are integers (0, 1, 2...).
    
    Camelot's df.columns = [0, 1, 2, ...] and the real headers are in the
    first data row. This detects that pattern and shifts the data.
    """
    if not data or len(data) < 2:
        return data
    headers = data[0]
    if all(re.match(r'^\d+$', str(h).strip()) for h in headers if str(h).strip()):
        return data[1:]  # Drop integer column headers
    return data


# ═══════════════════════════════════════════════════════════════════
#  Processing pipeline
# ═══════════════════════════════════════════════════════════════════

def _process(data, page, source, bbox=None):
    """Apply full processing pipeline to raw table data."""
    data = [[str(c) if c is not None else '' for c in r] for r in data]
    
    # Fix Camelot integer column headers
    if source.startswith('camelot'):
        data = _fix_camelot_columns(data)
    
    # Quality gate
    if not is_valid_table(data, source=source):
        return None
    
    # Cell repair
    data = repair_table_cells(data)
    
    # Header merging
    h, rows = merge_split_headers(data)
    
    # Multi-line expansion
    full = [h] + rows
    expanded = expand_multiline_cells(full)
    if len(expanded) > len(full):
        h, rows = expanded[0], expanded[1:]
    
    return {
        'page': page,
        'headers': h,
        'rows': rows,
        'bbox': bbox,
        'score': table_quality_score(data),
        'source': source,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main extraction
# ═══════════════════════════════════════════════════════════════════

def extract_tables_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Multi-strategy table extraction with quality filtering.
    
    Strategy priority:
      1. PyMuPDF find_tables() — fast, best line detection for most PDFs
      2. Camelot lattice — best for ruled tables with visible lines
      3. pdfplumber — fallback for pages where both above find nothing
    
    Returns list of dicts with keys:
      page, headers, rows, bbox, score, source
    
    Usage:
      tables = extract_tables_from_pdf('report.pdf')
      for t in tables:
          headers = t['headers']  # List[str]
          rows = t['rows']        # List[List[str]]
          page = t['page']        # 0-indexed page number
    """
    results = []
    page_entries = {}  # page → list of tables
    
    def _add(t):
        """Add table with deduplication."""
        p = t['page']
        if p not in page_entries:
            page_entries[p] = []
        
        # Check for duplicates
        for existing in list(page_entries[p]):
            # Content-based dedup
            if _tables_are_duplicates(t, existing):
                # Keep the one with better expansion × score
                t_val = len(t['rows']) * t['score']
                e_val = len(existing['rows']) * existing['score']
                if t_val <= e_val:
                    return
                if existing in results:
                    results.remove(existing)
                page_entries[p].remove(existing)
                break
            
            # Bbox-based dedup
            if t.get('bbox') and existing.get('bbox'):
                if bboxes_overlap(t['bbox'], existing['bbox']):
                    t_val = len(t['headers']) * t['score']
                    e_val = len(existing['headers']) * existing['score']
                    if t_val <= e_val:
                        return
                    if existing in results:
                        results.remove(existing)
                    page_entries[p].remove(existing)
                    break
        
        results.append(t)
        page_entries[p].append(t)
    
    # Get total page count
    total_pages = 0
    try:
        import fitz
        doc = fitz.open(file_path)
        total_pages = len(doc)
        doc.close()
    except ImportError:
        pass
    
    # ── Pass 1: PyMuPDF find_tables() ──
    try:
        import fitz
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            try:
                tabs = doc[page_num].find_tables()
            except Exception:
                continue
            for tab in tabs.tables:
                data = tab.extract()
                if data:
                    t = _process(data, page_num, 'pymupdf', tuple(tab.bbox))
                    if t:
                        _add(t)
        doc.close()
    except ImportError:
        pass
    
    # ── Pass 2: Camelot lattice on ALL pages ──
    try:
        import camelot
        page_str = ','.join(str(p + 1) for p in range(total_pages))
        for ct in camelot.read_pdf(file_path, pages=page_str, flavor='lattice'):
            df = ct.df
            data = [df.columns.tolist()] + df.values.tolist()
            t = _process(data, int(ct.page) - 1, 'camelot-lattice')
            if t:
                _add(t)
    except (ImportError, Exception):
        pass
    
    # ── Pass 3: pdfplumber for uncovered pages ──
    covered = {t['page'] for t in results}
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if page_num in covered:
                    continue
                
                best = None
                for v_strat, h_strat in [
                    ('lines', 'lines'), ('lines', 'text'),
                    ('text', 'lines'), ('text', 'text'),
                ]:
                    try:
                        tables = page.extract_tables(table_settings={
                            'vertical_strategy': v_strat,
                            'horizontal_strategy': h_strat,
                            'snap_tolerance': 5,
                            'join_tolerance': 5,
                        })
                    except Exception:
                        continue
                    for table in tables:
                        if not table or len(table) < 2:
                            continue
                        t = _process(table, page_num, f'pdfplumber-{v_strat}-{h_strat}')
                        if t and (not best or t['score'] > best['score']):
                            best = t
                
                if best:
                    _add(best)
    except ImportError:
        pass
    
    # Sort by page number, then vertical position
    results.sort(key=lambda t: (t['page'], t['bbox'][1] if t['bbox'] else 0))
    return results
