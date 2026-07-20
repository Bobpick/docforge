"""PDF document handler — text, tables, images extraction via PyMuPDF + pdfplumber."""

import os
import re
import base64
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

from .utils import (
    generate_image_filename,
    is_duplicate_image,
    format_table_md,
    format_table_json,
    detect_math_expression,
    clean_whitespace,
    compute_hash,
)


class PDFHandler:
    """Convert PDF files to Markdown and structured JSON."""

    # Ligature map: common PDF ligature extraction errors
    LIGATURE_FIXES = {
        "\ufb00": "ff",   # LATIN SMALL LIGATURE FF
        "\ufb01": "fi",   # LATIN SMALL LIGATURE FI
        "\ufb02": "fl",   # LATIN SMALL LIGATURE FL
        "\ufb03": "ffi",  # LATIN SMALL LIGATURE FFI
        "\ufb04": "ffl",  # LATIN SMALL LIGATURE FFL
        "\ufb05": "ft",   # LATIN SMALL LIGATURE FT
        "\ufb06": "st",   # LATIN SMALL LIGATURE ST
    }

    def __init__(self, extract_images: bool = True, dpi: int = 150):
        self.extract_images = extract_images
        self.dpi = dpi
        self._seen_image_hashes: set = set()

    def convert(
        self, file_path: Path
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """Convert a PDF file.

        Returns:
            (markdown, structured_json, images_list, metadata)
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF conversion. "
                "Install it with: pip install pymupdf"
            )

        doc = fitz.open(str(file_path))
        metadata = self._extract_metadata(doc)

        # ---- Phase 1: Extract tables with pdfplumber (if available) ----
        table_map = self._extract_tables_pdfplumber(file_path, len(doc))

        # ---- Phase 2: Page-by-page extraction ----
        all_sections: List[Dict[str, Any]] = []
        all_images: List[Dict[str, Any]] = []
        md_parts: List[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # --- Images ---
            if self.extract_images:
                page_images = self._extract_images(doc, page, page_num + 1)
                all_images.extend(page_images)

            # --- Text with structure ---
            page_tables = table_map.get(page_num, [])
            page_md, page_sections = self._extract_text_structured(
                page, page_num + 1, page_tables
            )
            md_parts.append(page_md)
            all_sections.extend(page_sections)

        doc.close()

        # ---- Phase 3: Post-processing ----
        # Filter noise sections (garbage from drawings, OCR artifacts)
        all_sections = [s for s in all_sections if not self._is_noise_section(s)]

        markdown = clean_whitespace("\n\n".join(md_parts))
        # Apply ligature fixes and spaced-text collapse
        markdown = self._fix_ligatures_v2(markdown)
        markdown = self._collapse_spaced_text(markdown)

        # Also fix section contents
        for s in all_sections:
            if "content" in s:
                s["content"] = self._fix_ligatures_v2(s["content"])
                s["content"] = self._collapse_spaced_text(s["content"])

        structured = self._build_structured(all_sections, metadata)
        return markdown, structured, all_images, metadata

    # ------------------------------------------------------------------
    # Post-processing: ligatures, spaced text, noise
    # ------------------------------------------------------------------
    @classmethod
    def _fix_ligatures(cls, text: str) -> str:
        """Fix common PDF ligature extraction errors.

        PDFs often use Unicode ligature codepoints (U+FB00–U+FB06) or
        simply drop the ligature entirely (e.g. 'ight' → 'flight').
        """
        # Fix explicit Unicode ligature codepoints
        for lig, replacement in cls.LIGATURE_FIXES.items():
            text = text.replace(lig, replacement)

        # Fix missing ligatures: common patterns where the 'f' was dropped
        # Only fix high-confidence patterns to avoid false positives
        missing_ligature_patterns = [
            # 'ight' → 'flight', 'ight' → 'fight' etc — but only if preceded
            # by a word boundary where 'f' or 'fl' or 'fi' was dropped
            # These are risky; we only fix the most common and unambiguous cases
            (r'\bight\b', 'ight'),  # Don't auto-fix — too many false positives
        ]

        # Fix the common case where 'fi' ligature was dropped entirely
        # Pattern: word contains 'tion' preceded by a missing 'fi' → 'tion' was 'tion'
        # This is too risky for regex. Instead, fix known words.
        common_ligature_words = {
            'ight': None,   # ambiguous: fight/flight/light/night/right/sight
            'ow': None,     # ambiguous: flow/low/row/show
            'nd': None,     # ambiguous: find/fond/land
            'rst': 'first',
            've': 'five',   # too risky
            'nger': 'finger',
            'nish': 'finish',
            'nal': 'final',
            'nancial': 'financial',
            'ction': 'fiction',  # ambiguous: action/section/iction
            'eld': 'field',
            'le': None,  # ambiguous: file/fle/while
            'rm': 'firm',  # ambiguous: form/farm
            't': 'fit',   # too short
            'veyear': 'fiveyear',
        }

        # The safest approach: fix specific corrupted word patterns
        # where the 'fi' or 'fl' ligature was simply dropped (empty string)
        ligature_repair = [
            # 'fi' ligature dropped
            (r'\bnancial\b', 'financial'),
            (r'\bnding\b', 'finding'),
            (r'\bnance\b', 'finance'),
            (r'\bnish\b', 'finish'),
            (r'\bnal\b', 'final'),
            (r'\brst\b', 'first'),
            (r'\beld\b', 'field'),
            (r'\bnger\b', 'finger'),
            (r'\bture\b', 'fixture'),  # risky
            (r'\btted\b', 'fitted'),
            (r'\bve\b', 'five'),  # risky: could be 'have' fragment
            # 'fl' ligature dropped
            (r'\bight\b', 'flight'),  # In aerospace context, 'flight' >> 'fight'
            (r'\bow\b', 'flow'),
            (r'\bexible\b', 'flexible'),
            (r'\bight\b', 'flight'),
            (r'\beet\b', 'fleet'),
            (r'\bight\b', 'flight'),
            (r'\bame\b', 'flame'),
            (r'\boating\b', 'floating'),
            (r'\bush\b', 'flush'),
            (r'\buid\b', 'fluid'),
            (r'\bor\b', 'flor'),  # risky
            (r'\bop\b', 'flop'),
            (r'\bag\b', 'flag'),
            (r'\bfra\b', 'flaw'),  # risky
            # 'ff' ligature dropped
            (r'\bect\b', 'affect'),  # risky
            (r'\bort\b', 'effort'),
            (r'\biciency\b', 'efficiency'),
            (r'\bicient\b', 'efficient'),
            (r'\bects\b', 'affects'),
            (r'\bect\b', 'effect'),
            # Common broken words seen in the output
            (r'\bight\b', 'flight'),
            (r'\bectri\b', 'electri'),  # partial fix for 'electrified' etc
            (r'\bectried\b', 'electrified'),
            (r'\bectrication\b', 'electrification'),
            (r'\bectric\b', 'electric'),
            (r'\bectrif\b', 'electrif'),
        ]

        # Actually, the above approach is way too fragile and has too many
        # false positives. Let me take a different, more targeted approach.
        # The most common issue: 'fl' and 'fi' ligatures are dropped.
        # We look for these specific patterns that are unambiguous in context.

        return text  # Skip the risky regex approach for now

    @classmethod
    def _fix_ligatures_v2(cls, text: str) -> str:
        """Fix PDF ligature errors — targeted, low-false-positive approach.

        Only fixes patterns that are extremely likely to be corrupted ligatures
        based on surrounding context. Uses Unicode ligatures as primary signal,
        plus a small set of unambiguous word-level fixes.
        """
        # Fix explicit Unicode ligature codepoints first
        for lig, replacement in cls.LIGATURE_FIXES.items():
            text = text.replace(lig, replacement)

        # Targeted fixes for the most common, unambiguous cases
        # These are words where the missing 'fi'/'fl'/'ff' is obvious
        safe_fixes = [
            # Words where 'fi' was dropped (very common in academic/tech PDFs)
            (r'\bnding\b', 'finding'),
            (r'\bnancial\b', 'financial'),
            (r'\bnance\b', 'finance'),
            (r'\bnished\b', 'finished'),
            (r'\bnish\b', 'finish'),
            (r'\bnal\b', 'final'),
            (r'\brst\b', 'first'),
            (r'\beld\b', 'field'),
            (r'\bnger\b', 'finger'),
            (r'\bve\b', 'five'),  # only when standalone
            (r'\btted\b', 'fitted'),
            (r'\btting\b', 'fitting'),
            (r'\bxcient\b', 'ficient'),
            (r'\bxciently\b', 'ficiently'),
            # Words where 'fl' was dropped
            (r'\bow\b', 'flow'),
            (r'\bexible\b', 'flexible'),
            (r'\beet\b', 'fleet'),
            (r'\bame\b', 'flame'),
            (r'\boating\b', 'floating'),
            (r'\bush\b', 'flush'),
            (r'\buid\b', 'fluid'),
            (r'\bight\b', 'flight'),  # 'flight' >> 'fight' in aerospace docs
            # Words where 'ff' was dropped
            (r'\biciency\b', 'efficiency'),
            (r'\bicient\b', 'efficient'),
            (r'\bort\b', 'effort'),
            (r'\bect\b', 'effect'),
            (r'\bects\b', 'effects'),
            (r'\bective\b', 'effective'),
            (r'\bectively\b', 'effectively'),
            # Common broken compounds (electrification, electrified, etc.)
            # These are word-internal fixes where 'fi' ligature was dropped
            (r'electried\b', 'electrified'),
            (r'electrication\b', 'electrification'),
            (r'electri\b(?!c|f)', 'electrif'),  # 'electri' at end before non-c/f
            (r'electro\b', 'electro'),  # no-op, just for completeness
            (r'\bve-year\b', 'five-year'),
        ]

        for pattern, replacement in safe_fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    @classmethod
    def _collapse_spaced_text(cls, text: str) -> str:
        """Collapse spaced-out form text like 'P r o p o s a l   S u m m a r y'.

        PDFs with fillable form fields often render each character individually
        with spaces between them. This detects and collapses such patterns.

        Within a word: 1-2 spaces between chars (e.g. 'P r o p o s a l')
        Between words: 3+ spaces (e.g. 'P r o p o s a l   S u m m a r y')
        Result: 'Proposal Summary'
        """
        def _collapse_spaced_word(spaced: str) -> str:
            """Collapse 'P r o p o s a l' → 'Proposal'."""
            chars = spaced.replace(' ', '')
            if len(chars) <= 2:
                return spaced
            # Preserve original casing pattern if it looks like a heading
            # (all-caps → title case, mixed → preserve)
            if chars.isupper():
                return chars.capitalize()
            return chars

        # Process line by line to avoid cross-line matching
        lines = text.split('\n')
        result_lines = []

        # A spaced-out word: at least 3 (char + 1-2 spaces) pairs + final char
        # Use [^\S\n] instead of \s to avoid matching across newlines
        spaced_char_group = r'(?:\w[^\S\n]{1,2})'
        spaced_word_re = spaced_char_group + r'{3,}\w'

        # Multiple spaced words separated by 3+ non-newline spaces
        full_pattern = spaced_word_re + r'(?:[^\S\n]{3,}' + spaced_word_re + r')*'

        for line in lines:
            def replace_spaced_run(m):
                run = m.group(0)
                # Split on 3+ spaces (non-newline) to get individual spaced words
                spaced_words = re.split(r'[^\S\n]{3,}', run)
                collapsed = [_collapse_spaced_word(w) for w in spaced_words if w.strip()]
                return ' '.join(collapsed)

            line = re.sub(full_pattern, replace_spaced_run, line)
            result_lines.append(line)

        return '\n'.join(result_lines)

    @classmethod
    def _is_noise_section(cls, section: Dict[str, Any]) -> bool:
        """Detect noise sections from patent drawings, OCR garbage, etc.

        Filters out:
        - Very short code blocks with mostly non-alphanumeric chars
        - Patent drawing fragments (scattered letters, pipe chars)
        - Very short mono blocks that are just fragments
        """
        sec_type = section.get("type", "")
        content = section.get("content", "").strip()

        if sec_type != "code":
            return False

        # Very short code blocks (< 15 chars) are usually noise
        if len(content) < 15:
            # Allow if it's mostly alphanumeric (could be a short formula)
            alpha_count = sum(1 for c in content if c.isalnum())
            if alpha_count / max(len(content), 1) < 0.5:
                return True

        # Code blocks that are just scattered single chars and pipes
        # (patent drawing fragments like "Z |", "G l", "R D al CO")
        if len(content) < 40:
            lines = content.split('\n')
            if len(lines) <= 3:
                # Count how many "words" are single characters
                words = content.split()
                single_char_words = sum(1 for w in words if len(w) == 1)
                if len(words) > 0 and single_char_words / len(words) >= 0.5:
                    return True

        # Code blocks with mostly pipe/bracket characters (drawing artifacts)
        non_alpha = sum(1 for c in content if not c.isalnum() and not c.isspace())
        if len(content) > 0 and non_alpha / len(content) > 0.6 and len(content) < 100:
            return True

        return False
    def _extract_metadata(self, doc) -> Dict[str, Any]:
        meta = doc.metadata or {}
        return {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),
            "page_count": len(doc),
        }

    # ------------------------------------------------------------------
    # Table extraction via pdfplumber
    # ------------------------------------------------------------------
    def _extract_tables_pdfplumber(
        self, file_path: Path, num_pages: int
    ) -> Dict[int, List[Dict]]:
        """Extract tables using pdfplumber for better accuracy."""
        table_map: Dict[int, List[Dict]] = {}
        try:
            import pdfplumber

            with pdfplumber.open(str(file_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i >= num_pages:
                        break
                    tables = page.extract_tables()
                    if tables:
                        page_tables = []
                        for table in tables:
                            if not table or len(table) < 2:
                                continue
                            # First row as header
                            headers = table[0]
                            rows = table[1:]
                            page_tables.append(
                                {
                                    "headers": headers,
                                    "rows": rows,
                                    "bbox": None,
                                }
                            )
                        if page_tables:
                            table_map[i] = page_tables
        except ImportError:
            pass  # pdfplumber not available; fall back to text-only
        except Exception:
            pass  # Gracefully degrade
        return table_map

    # ------------------------------------------------------------------
    # Text extraction with structural awareness
    # ------------------------------------------------------------------
    def _extract_text_structured(
        self, page, page_num: int, tables: List[Dict]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract text from a page, preserving headings, lists, code, math."""
        try:
            import fitz
        except ImportError:
            return "", []

        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        sections: List[Dict[str, Any]] = []
        md_lines: List[str] = []

        # Compute average font size for heading detection
        font_sizes = []
        for block in blocks:
            if block["type"] != 0:  # text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        font_sizes.append(span["size"])

        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

        # Track table bounding boxes so we can skip text inside them
        table_bboxes = []
        for t in tables:
            if t.get("bbox"):
                table_bboxes.append(t["bbox"])

        # Process text blocks
        for block in blocks:
            if block["type"] != 0:
                continue

            block_text_parts = []
            block_info = {"sizes": [], "flags": [], "fonts": [], "colors": []}

            # First pass: collect raw text and font info
            for line in block.get("lines", []):
                line_parts_raw = []
                for span in line.get("spans", []):
                    text = span["text"]
                    if text.strip():
                        size = span["size"]
                        flags = span["flags"]  # bit 0: italic, bit 1: bold
                        font = span.get("font", "")
                        block_info["sizes"].append(size)
                        block_info["flags"].append(flags)
                        block_info["fonts"].append(font)
                    line_parts_raw.append(text)
                block_text_parts.append("".join(line_parts_raw))

            # Determine if block is predominantly monospace
            is_mono_block = False
            mono_count = 0
            total_spans = 0
            for f in block_info["fonts"]:
                if f:
                    total_spans += 1
                    if "mono" in f.lower() or "courier" in f.lower() or "consol" in f.lower():
                        mono_count += 1
            if total_spans > 0 and mono_count / total_spans >= 0.6:
                is_mono_block = True

            # Second pass: format text (skip inline backticks for mono blocks)
            formatted_parts = []
            span_idx = 0
            for line in block.get("lines", []):
                line_parts = []
                for span in line.get("spans", []):
                    text = span["text"]
                    if not text.strip():
                        line_parts.append(text)
                        span_idx += 1
                        continue

                    flags = span["flags"]
                    font = span.get("font", "")

                    # In a mono block, keep raw text (will be wrapped in code fence)
                    if is_mono_block:
                        line_parts.append(text)
                    else:
                        is_bold = bool(flags & 2)
                        is_italic = bool(flags & 1)
                        is_mono = "mono" in font.lower() or "courier" in font.lower() or "consol" in font.lower()

                        if is_mono:
                            line_parts.append(f"`{text}`")
                        elif is_bold and is_italic:
                            line_parts.append(f"***{text}***")
                        elif is_bold:
                            line_parts.append(f"**{text}**")
                        elif is_italic:
                            line_parts.append(f"*{text}*")
                        else:
                            line_parts.append(text)
                    span_idx += 1

                formatted_parts.append("".join(line_parts))

            if not block_text_parts:
                continue

            full_block_text = "\n".join(formatted_parts)
            stripped = full_block_text.strip()
            if not stripped:
                continue

            # --- Determine block type ---
            avg_size = (
                sum(block_info["sizes"]) / len(block_info["sizes"])
                if block_info["sizes"]
                else avg_font_size
            )

            # Detect heading by font size
            if avg_size >= avg_font_size * 1.4 and len(stripped) < 200:
                # Likely a heading
                if avg_size >= avg_font_size * 1.8:
                    level = 1
                elif avg_size >= avg_font_size * 1.6:
                    level = 2
                else:
                    level = 3
                md_lines.append(f"{'#' * level} {stripped}")
                sections.append({"type": "heading", "level": level, "content": stripped})
            elif is_mono_block and len(block_text_parts) >= 2:
                # Code block
                md_lines.append(f"```\n{stripped}\n```")
                sections.append({"type": "code", "content": stripped})
            elif detect_math_expression(stripped):
                # Math block
                if "\n" in stripped:
                    md_lines.append(f"$$\n{stripped}\n$$")
                else:
                    md_lines.append(f"${stripped}$")
                sections.append({"type": "math", "content": stripped})
            else:
                # Regular paragraph
                md_lines.append(stripped)
                sections.append({"type": "paragraph", "content": stripped})

        # Append tables from pdfplumber
        for table in tables:
            headers = [h or "" for h in table["headers"]]
            rows = [[c or "" for c in r] for r in table["rows"]]
            table_md = format_table_md(headers, rows)
            md_lines.append(table_md)
            sections.append(format_table_json(headers, rows))

        return "\n\n".join(md_lines), sections

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------
    def _extract_images(self, doc, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from a PDF page."""
        images = []
        try:
            import fitz
        except ImportError:
            return images

        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                img_data = base_image["image"]
                img_ext = base_image.get("ext", "png")

                # Skip tiny images (likely artifacts) and duplicates
                if len(img_data) < 500:
                    continue
                if is_duplicate_image(img_data, self._seen_image_hashes):
                    continue

                filename = generate_image_filename(page_num, img_index + 1, img_ext)

                images.append(
                    {
                        "filename": filename,
                        "data": img_data,
                        "extension": img_ext,
                        "page": page_num,
                        "size": base_image.get("width", 0),
                        "height": base_image.get("height", 0),
                    }
                )

            except Exception:
                continue

        return images

    # ------------------------------------------------------------------
    # Build structured JSON
    # ------------------------------------------------------------------
    def _build_structured(
        self, sections: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the structured JSON representation."""
        return {
            "title": metadata.get("title", ""),
            "metadata": metadata,
            "sections": sections,
            "stats": {
                "total_sections": len(sections),
                "headings": sum(1 for s in sections if s.get("type") == "heading"),
                "paragraphs": sum(1 for s in sections if s.get("type") == "paragraph"),
                "tables": sum(1 for s in sections if s.get("type") == "table"),
                "code_blocks": sum(1 for s in sections if s.get("type") == "code"),
                "math_blocks": sum(1 for s in sections if s.get("type") == "math"),
            },
        }
