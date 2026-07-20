"""PDF document handler — text, tables, images extraction via PyMuPDF + pdfplumber."""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

from .utils import (
    generate_image_filename,
    is_duplicate_image,
    format_table_md,
    format_table_json,
    detect_math_expression,
    clean_whitespace,
)
from .table_utils import (
    TABLE_SETTINGS,
    normalize_table,
    pick_best_table,
    score_table,
    dedupe_table_list,
)
from .text_cleanup import (
    LIGATURE_CHARS,
    fix_ligatures_and_ocr,
    collapse_spaced_text,
    looks_like_code,
    looks_like_drawing_label,
)


class PDFHandler:
    """Convert PDF files to Markdown and structured JSON."""

    # Kept for backward compatibility with older call sites / tests
    LIGATURE_FIXES = LIGATURE_CHARS

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

        # ---- Phase 1: Multi-strategy table extraction ----
        table_map = self._extract_tables(file_path, doc)

        # ---- Phase 2: Page-by-page extraction ----
        all_sections: List[Dict[str, Any]] = []
        all_images: List[Dict[str, Any]] = []
        md_parts: List[str] = []
        collected_tables: List[Dict[str, Any]] = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            if self.extract_images:
                page_images = self._extract_images(doc, page, page_num + 1)
                all_images.extend(page_images)

            page_tables = table_map.get(page_num, [])
            collected_tables.extend(page_tables)
            page_md, page_sections = self._extract_text_structured(
                page, page_num + 1, page_tables
            )
            md_parts.append(page_md)
            all_sections.extend(page_sections)

        doc.close()

        # ---- Phase 3: Post-processing ----
        all_sections = [s for s in all_sections if not self._is_noise_section(s)]

        # Drop near-duplicate tables (Google Patents citation chrome x5)
        all_sections = self._dedupe_table_sections(all_sections)

        markdown = clean_whitespace("\n\n".join(md_parts))
        markdown = self._dedupe_markdown_tables(markdown)
        markdown = fix_ligatures_and_ocr(markdown)
        markdown = collapse_spaced_text(markdown)

        for s in all_sections:
            if "content" in s and isinstance(s["content"], str):
                s["content"] = fix_ligatures_and_ocr(s["content"])
                s["content"] = collapse_spaced_text(s["content"])

        structured = self._build_structured(all_sections, metadata)
        return markdown, structured, all_images, metadata

    # ------------------------------------------------------------------
    # Post-processing: ligatures, spaced text, noise (compat wrappers)
    # ------------------------------------------------------------------
    @classmethod
    def _fix_ligatures(cls, text: str) -> str:
        return fix_ligatures_and_ocr(text)

    @classmethod
    def _fix_ligatures_v2(cls, text: str) -> str:
        return fix_ligatures_and_ocr(text)

    @classmethod
    def _collapse_spaced_text(cls, text: str) -> str:
        return collapse_spaced_text(text)

    @classmethod
    def _is_noise_section(cls, section: Dict[str, Any]) -> bool:
        """Detect noise sections from patent drawings, OCR garbage, etc."""
        sec_type = section.get("type", "")
        content = section.get("content", "").strip()

        if sec_type == "code":
            if looks_like_drawing_label(content):
                return True
            if len(content) < 15:
                alpha_count = sum(1 for c in content if c.isalnum())
                if alpha_count / max(len(content), 1) < 0.5:
                    return True
            if len(content) < 40:
                words = content.split()
                single_char_words = sum(1 for w in words if len(w) == 1)
                if words and single_char_words / len(words) >= 0.5:
                    return True
            non_alpha = sum(
                1 for c in content if not c.isalnum() and not c.isspace()
            )
            if content and non_alpha / len(content) > 0.6 and len(content) < 100:
                return True
            return False

        # Drawing labels wrongly kept as paragraphs with only fragments
        if sec_type == "paragraph" and looks_like_drawing_label(content):
            # Keep readable labels like "OUTPUT BANDPASS FILTER"; drop junk
            words = content.split()
            if words and sum(1 for w in words if len(w) == 1) / len(words) >= 0.5:
                return True

        return False

    @staticmethod
    def _dedupe_table_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preserve order; drop near-duplicate tables (citation chrome xN)."""
        from .table_utils import tables_are_duplicates

        result: List[Dict[str, Any]] = []
        seen_tables: List[Dict[str, Any]] = []
        for s in sections:
            if s.get("type") != "table":
                result.append(s)
                continue
            cur = (list(s.get("headers") or []), [list(r) for r in (s.get("rows") or [])])
            dup = False
            for u in seen_tables:
                other = (
                    list(u.get("headers") or []),
                    [list(r) for r in (u.get("rows") or [])],
                )
                if tables_are_duplicates(cur, other):
                    dup = True
                    break
            if not dup:
                seen_tables.append(s)
                result.append(s)
        return result

    @staticmethod
    def _dedupe_markdown_tables(markdown: str) -> str:
        """Remove near-identical markdown tables that repeat across pages."""
        import re
        from hashlib import md5

        parts = re.split(r"(\n\n)", markdown)
        seen: set = set()
        out: List[str] = []
        for part in parts:
            if part.startswith("|") and "\n|" in part:
                # Normalize whitespace for signature
                sig = md5(
                    re.sub(r"\s+", " ", part.strip()).lower().encode("utf-8")
                ).hexdigest()
                if sig in seen:
                    continue
                seen.add(sig)
            out.append(part)
        return "".join(out)

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
    # Table extraction — multi-strategy pdfplumber + PyMuPDF fallback
    # ------------------------------------------------------------------
    def _extract_tables(
        self, file_path: Path, doc
    ) -> Dict[int, List[Dict]]:
        """Extract tables with multiple strategies; keep best-scoring grids."""
        table_map: Dict[int, List[Dict]] = {}
        num_pages = len(doc)

        # --- pdfplumber multi-strategy ---
        try:
            import pdfplumber

            with pdfplumber.open(str(file_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i >= num_pages:
                        break
                    page_tables = self._extract_page_tables_pdfplumber(page)
                    if page_tables:
                        table_map[i] = page_tables
        except ImportError:
            pass
        except Exception:
            pass

        # --- PyMuPDF find_tables fallback / fill gaps ---
        try:
            for i in range(num_pages):
                if i in table_map and any(
                    score_table(t["headers"], t["rows"]) >= 8 for t in table_map[i]
                ):
                    continue  # already have a decent table
                page = doc[i]
                if not hasattr(page, "find_tables"):
                    break
                found = page.find_tables()
                if not found:
                    continue
                tables = getattr(found, "tables", found) or []
                candidates = []
                for tab in tables:
                    try:
                        raw = tab.extract()
                    except Exception:
                        continue
                    norm = normalize_table(raw)
                    if norm:
                        candidates.append(norm)
                if not candidates:
                    continue
                best = pick_best_table(candidates)
                if not best:
                    continue
                headers, rows = best
                entry = {"headers": headers, "rows": rows, "bbox": None}
                if i in table_map:
                    # Keep whichever scores higher overall
                    existing_best = max(
                        (score_table(t["headers"], t["rows"]) for t in table_map[i]),
                        default=-1,
                    )
                    if score_table(headers, rows) > existing_best:
                        table_map[i] = [entry]
                else:
                    table_map[i] = [entry]
        except Exception:
            pass

        return table_map

    def _extract_page_tables_pdfplumber(self, page) -> List[Dict]:
        """Try several pdfplumber settings; pick non-overlapping best tables."""
        all_candidates: List[Tuple[List[str], List[List[str]]]] = []

        for settings in TABLE_SETTINGS:
            try:
                if settings is None:
                    tables = page.extract_tables() or []
                else:
                    tables = page.extract_tables(table_settings=settings) or []
            except Exception:
                continue
            for table in tables:
                norm = normalize_table(table)
                if norm:
                    all_candidates.append(norm)

        if not all_candidates:
            return []

        # Greedy: take best, then next non-duplicate, etc.
        remaining = list(all_candidates)
        selected: List[Dict] = []
        from .table_utils import tables_are_duplicates

        while remaining:
            best = pick_best_table(remaining)
            if not best:
                break
            headers, rows = best
            selected.append({"headers": headers, "rows": rows, "bbox": None})
            # Remove duplicates of this table from remaining
            remaining = [
                c
                for c in remaining
                if not tables_are_duplicates(c, best, threshold=0.75)
            ]
            if len(selected) >= 8:
                break

        return selected

    # Back-compat alias
    def _extract_tables_pdfplumber(
        self, file_path: Path, num_pages: int
    ) -> Dict[int, List[Dict]]:
        try:
            import fitz
            doc = fitz.open(str(file_path))
            try:
                return self._extract_tables(file_path, doc)
            finally:
                doc.close()
        except Exception:
            return {}

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
                if avg_size >= avg_font_size * 1.8:
                    level = 1
                elif avg_size >= avg_font_size * 1.6:
                    level = 2
                else:
                    level = 3
                md_lines.append(f"{'#' * level} {stripped}")
                sections.append({"type": "heading", "level": level, "content": stripped})
            elif looks_like_code(stripped, is_mono_font=is_mono_block) and (
                is_mono_block or len(block_text_parts) >= 2
            ):
                # Real code — not patent drawing labels in monospace
                md_lines.append(f"```\n{stripped}\n```")
                sections.append({"type": "code", "content": stripped})
            elif looks_like_drawing_label(stripped) and is_mono_block:
                # Drawing callouts: keep as plain paragraph, never code fence
                md_lines.append(stripped)
                sections.append({"type": "paragraph", "content": stripped})
            elif detect_math_expression(stripped):
                if "\n" in stripped:
                    md_lines.append(f"$$\n{stripped}\n$$")
                else:
                    md_lines.append(f"${stripped}$")
                sections.append({"type": "math", "content": stripped})
            else:
                md_lines.append(stripped)
                sections.append({"type": "paragraph", "content": stripped})

        # Append tables (multi-strategy extraction)
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
