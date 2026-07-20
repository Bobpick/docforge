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

        markdown = clean_whitespace("\n\n".join(md_parts))
        structured = self._build_structured(all_sections, metadata)
        return markdown, structured, all_images, metadata

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
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
