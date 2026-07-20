"""Image handler — OCR-based conversion of image files to Markdown and JSON."""

import re
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

from .utils import (
    format_table_md,
    format_table_json,
    detect_math_expression,
    clean_whitespace,
)


class ImageHandler:
    """Convert image files (PNG, JPG, TIFF, etc.) to Markdown via OCR."""

    def __init__(self, extract_images: bool = True, languages: List[str] = None):
        """
        Args:
            extract_images: Whether to include the source image in the output.
            languages: OCR languages (e.g. ['eng', 'fra']). Defaults to English.
        """
        self.extract_images = extract_images
        self.languages = languages or ["eng"]
        self._seen_image_hashes: set = set()

    def convert(
        self, file_path: Path
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """Convert an image file using OCR."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required for image conversion. "
                "Install it with: pip install Pillow"
            )

        # Open the image to get metadata
        img = Image.open(str(file_path))
        metadata = {
            "format": img.format or file_path.suffix.lstrip("."),
            "size": list(img.size),
            "mode": img.mode,
            "source_file": file_path.name,
        }

        # --- OCR extraction ---
        ocr_text = self._perform_ocr(img)
        images = []

        if self.extract_images:
            # Save the source image as an extracted image
            img_byte_arr = __import__("io").BytesIO()
            img.save(img_byte_arr, format=img.format or "PNG")
            img_data = img_byte_arr.getvalue()
            ext = file_path.suffix.lstrip(".") or "png"
            images.append(
                {
                    "filename": f"source_image.{ext}",
                    "data": img_data,
                    "extension": ext,
                    "page": 1,
                }
            )

        # --- Structure the OCR output ---
        sections: List[Dict[str, Any]] = []
        md_parts: List[str] = []

        if ocr_text.strip():
            # Split into paragraphs and classify
            paragraphs = re.split(r"\n\s*\n", ocr_text)

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # Try to classify the paragraph
                classified = self._classify_text(para)
                md_parts.append(classified["markdown"])
                sections.append(classified["section"])

        # Add image reference
        if self.extract_images:
            img_ref = f"![Source Image](images/source_image.{file_path.suffix.lstrip('.') or 'png'})"
            md_parts.append(img_ref)
            sections.append({"type": "image", "path": "images/source_image"})

        markdown = clean_whitespace("\n\n".join(md_parts))
        structured = self._build_structured(sections, metadata)
        return markdown, structured, images, metadata

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    def _perform_ocr(self, img) -> str:
        """Perform OCR on an image using pytesseract.
        
        Tries to preserve layout for table-like structures.
        """
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract is required for image OCR. "
                "Install it with: pip install pytesseract\n"
                "Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
            )

        lang = "+".join(self.languages)

        # Try layout-preserving OCR first
        try:
            text = pytesseract.image_to_string(
                img, lang=lang, config="--psm 6"
            )
        except Exception:
            try:
                text = pytesseract.image_to_string(img, lang=lang)
            except Exception:
                return ""

        # Also try to extract structured data for table detection
        try:
            data = pytesseract.image_to_data(
                img, lang=lang, output_type=pytesseract.Output.DICT
            )
            tables = self._detect_tables_in_ocr(data)
            if tables:
                # Append detected tables to the text
                table_texts = []
                for table in tables:
                    table_md = format_table_md(table["headers"], table["rows"])
                    table_texts.append(table_md)
                text = text + "\n\n" + "\n\n".join(table_texts)
        except Exception:
            pass

        return text

    def _detect_tables_in_ocr(self, data: dict) -> List[Dict]:
        """Attempt to detect tabular structures from Tesseract output.
        
        Uses position and alignment heuristics to group text into rows and columns.
        """
        words = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue
            words.append(
                {
                    "text": text,
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                    "conf": int(data["conf"][i]),
                }
            )

        if len(words) < 6:
            return []

        # Group words into rows by Y position (within a tolerance)
        rows = self._group_into_rows(words)
        if len(rows) < 2:
            return []

        # Check if rows are roughly aligned into columns
        tables = self._extract_aligned_tables(rows)
        return tables

    def _group_into_rows(self, words: list, y_tolerance: int = 10) -> List[List[dict]]:
        """Group words into rows based on Y position."""
        if not words:
            return []

        # Sort by Y then X
        sorted_words = sorted(words, key=lambda w: (w["y"], w["x"]))

        rows = []
        current_row = [sorted_words[0]]

        for word in sorted_words[1:]:
            if abs(word["y"] - current_row[0]["y"]) <= y_tolerance:
                current_row.append(word)
            else:
                rows.append(sorted(current_row, key=lambda w: w["x"]))
                current_row = [word]

        if current_row:
            rows.append(sorted(current_row, key=lambda w: w["x"]))

        return rows

    def _extract_aligned_tables(self, rows: List[List[dict]]) -> List[Dict]:
        """Detect column alignment across rows to identify tables."""
        # This is a heuristic approach:
        # If most rows have a similar number of items and their X positions
        # cluster into consistent columns, treat it as a table.

        if len(rows) < 2:
            return []

        # Count items per row
        row_lengths = [len(r) for r in rows]
        max_len = max(row_lengths)

        # If most rows have a similar number of columns (>=2), it's likely a table
        consistent_cols = sum(1 for l in row_lengths if l >= 2 and l >= max_len - 1)

        if consistent_cols < len(rows) * 0.5 or max_len < 2:
            return []

        # Build the table
        table_rows = []
        for row in rows:
            cells = [w["text"] for w in row]
            table_rows.append(cells)

        # First row as header
        headers = table_rows[0] if table_rows else []
        data_rows = table_rows[1:] if len(table_rows) > 1 else []

        return [{"headers": headers, "rows": data_rows}]

    # ------------------------------------------------------------------
    # Text classification
    # ------------------------------------------------------------------
    def _classify_text(self, text: str) -> Dict[str, Any]:
        """Classify a text block as heading, paragraph, code, math, etc."""
        lines = text.split("\n")

        # Short, single-line text could be a heading
        if len(lines) == 1 and len(text) < 100:
            # Check if it looks like a heading (starts with capital, short)
            if text[0].isupper() or text[0].isdigit():
                return {
                    "markdown": f"### {text}",
                    "section": {"type": "heading", "level": 3, "content": text},
                }

        # Check for code-like content
        if self._looks_like_code(text):
            return {
                "markdown": f"```\n{text}\n```",
                "section": {"type": "code", "content": text},
            }

        # Check for math
        if detect_math_expression(text):
            if "\n" in text:
                return {
                    "markdown": f"$$\n{text}\n$$",
                    "section": {"type": "math", "content": text},
                }
            else:
                return {
                    "markdown": f"${text}$",
                    "section": {"type": "math", "content": text},
                }

        # Default: paragraph
        return {
            "markdown": text,
            "section": {"type": "paragraph", "content": text},
        }

    def _looks_like_code(self, text: str) -> bool:
        """Heuristic to detect if text looks like code."""
        code_indicators = [
            (r"^\s*(def |class |import |from |if __name__)", 3),
            (r"^\s*(function |var |const |let |=>)", 2),
            (r"^\s*(#include |int main|void |public |private )", 3),
            (r"[{};]\s*$", 1),
            (r"^\s*\d+\.\s+\w+", 0),  # Numbered list (not code)
        ]

        score = 0
        lines = text.split("\n")
        for line in lines:
            for pattern, weight in code_indicators:
                if re.search(pattern, line):
                    score += weight

        # Threshold: if enough lines look like code
        return score >= len(lines) * 0.4 + 2

    # ------------------------------------------------------------------
    # Structured JSON
    # ------------------------------------------------------------------
    def _build_structured(
        self, sections: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the structured JSON representation."""
        return {
            "title": metadata.get("source_file", ""),
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
