"""Artifact removal: headers, footers, page numbers, watermarks."""

import re
from typing import Dict, Any, Optional
from collections import Counter


class ArtifactRemover:
    """Detects and removes document artifacts like headers, footers, and page numbers."""

    def __init__(self, min_repetition: int = 2):
        """
        Args:
            min_repetition: Minimum number of times a line must appear
                           across pages to be considered an artifact.
        """
        self.min_repetition = min_repetition

    def remove(self, markdown: str, structured: Optional[Dict[str, Any]] = None) -> str:
        """Remove detected artifacts from markdown text.
        
        Args:
            markdown: The full markdown text.
            structured: Optional structured data for additional context.
            
        Returns:
            Cleaned markdown text.
        """
        lines = markdown.split("\n")

        patent_print = self._looks_like_patent_print(markdown)
        
        # Strategy 1: Remove page number patterns
        lines = self._remove_page_numbers(lines)
        
        # Strategy 2: Remove repeated header/footer lines
        # Patent print-to-PDF repeats chrome on every page — more aggressive
        lines = self._remove_repeated_lines(
            lines, min_repetition=2 if patent_print else self.min_repetition
        )
        
        # Strategy 3: Remove common artifact patterns
        lines = self._remove_artifact_patterns(lines)
        
        # Strategy 4: Clean up section-page artifacts (e.g., "Section Title .... 5")
        lines = self._remove_leader_dots(lines)

        # Strategy 5: Collapse duplicate consecutive paragraphs (print chrome)
        lines = self._collapse_duplicate_runs(lines)

        # Strategy 6: Deduplicate identical markdown tables
        result = "\n".join(lines)
        result = self._dedupe_markdown_tables(result)
        
        # Clean up excessive blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    @staticmethod
    def _looks_like_patent_print(text: str) -> bool:
        signals = 0
        for needle in (
            "patents.google.com",
            "Find Prior Art",
            "Family To Family Citations",
            "Cited by examiner",
            "Patent Citations",
            "Non-Patent Citations",
            "Download PDF",
        ):
            if needle.lower() in text.lower():
                signals += 1
        return signals >= 2

    def _remove_page_numbers(self, lines: list) -> list:
        """Remove standalone page numbers."""
        page_num_patterns = [
            r'^\s*\d+\s*$',                           # Just a number
            r'^\s*[-–—]\s*\d+\s*[-–—]\s*$',           # - 5 -
            r'^\s*page\s+\d+\s*$',                      # Page 5
            r'^\s*p\.\s*\d+\s*$',                       # p. 5
            r'^\s*\d+\s*/\d+\s*$',                      # 5/10
            r'^\s*\[\s*\d+\s*\]\s*$',                   # [5]
            r'^\s*\d+\s+of\s+\d+\s*$',                  # 5 of 10
        ]
        
        cleaned = []
        for line in lines:
            is_page_num = False
            for pattern in page_num_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_page_num = True
                    break
            if not is_page_num:
                cleaned.append(line)
        return cleaned

    def _remove_repeated_lines(
        self, lines: list, min_repetition: Optional[int] = None
    ) -> list:
        """Remove lines that repeat across many 'pages' (heuristic).
        
        In markdown converted from multi-page documents, page breaks may be
        represented by horizontal rules or blank lines. We look for lines that
        appear many times and are short (typical of headers/footers).
        """
        threshold = min_repetition if min_repetition is not None else self.min_repetition
        line_counts = Counter(lines)
        
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned.append(line)
                continue
            
            count = line_counts.get(line, 1)
            is_short = len(stripped) < 100
            is_repeated = count >= threshold
            
            if is_repeated and is_short and not self._is_structural(stripped):
                continue  # Skip this artifact
            
            cleaned.append(line)
        return cleaned

    def _collapse_duplicate_runs(self, lines: list) -> list:
        """Drop consecutive identical non-empty lines (print header spam)."""
        cleaned = []
        prev = None
        for line in lines:
            stripped = line.strip()
            if stripped and stripped == prev and not self._is_structural(stripped):
                continue
            cleaned.append(line)
            prev = stripped if stripped else prev
            if not stripped:
                prev = None  # blank line resets run tracking for paragraphs
        return cleaned

    @staticmethod
    def _dedupe_markdown_tables(markdown: str) -> str:
        """Keep first copy of each identical markdown table block."""
        from hashlib import md5

        blocks = re.split(r"(\n{2,})", markdown)
        seen = set()
        out = []
        for block in blocks:
            if block.startswith("|") and "\n|" in block:
                sig = md5(re.sub(r"\s+", " ", block.strip()).lower().encode()).hexdigest()
                if sig in seen:
                    continue
                seen.add(sig)
            out.append(block)
        return "".join(out)

    def _is_structural(self, line: str) -> bool:
        """Check if a line is structural markdown that shouldn't be removed."""
        structural_patterns = [
            r'^#+\s',       # Headings
            r'^\|',         # Table rows
            r'^```',        # Code fences
            r'^\s*[-*+]\s', # List items
            r'^\s*\d+\.\s', # Numbered list items
            r'^>',          # Blockquotes
            r'^---',        # Horizontal rules
            r'^\$\$',       # Math blocks
        ]
        for pattern in structural_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _remove_artifact_patterns(self, lines: list) -> list:
        """Remove common artifact patterns like 'Confidential', 'Draft', etc.

        Also removes web-printed page chrome (Google Patents, etc.) and
        common PDF reader artifacts.
        """
        artifact_patterns = [
            # Standard document artifacts
            r'(?i)^\s*draft\b.*$',
            r'(?i)^\s*confidential\b.*$',
            r'(?i)^\s*do\s+not\s+distribute\b.*$',
            r'(?i)^\s*internal\s+use\s+only\b.*$',
            r'(?i)^\s*proprietary\s+and\s+confidential\b.*$',
            r'(?i)^\s*sample\b.*$',
            r'(?i)^\s*watermark\b.*$',
            r'(?i)^\s*not\s+for\s+(distribution|publication|release)\b.*$',
            r'(?i)^\s*for\s+internal\s+use\b.*$',
            r'(?i)^\s*trade\s+secret\b.*$',
            # Web-printed page chrome (Google Patents, Google Scholar, etc.)
            r'(?i)^\s*Download\s+PDF\b.*$',
            r'(?i)^\s*Find\s+Prior\s+Art\b.*$',
            r'(?i)^\s*Similar\s*$',
            r'(?i)^\s*Show\s+more\s*$',
            r'(?i)^\s*Send\s+Feedback\s*$',
            r'(?i)^\s*About\s*$',
            r'(?i)^\s*Public\s+Datasets\b.*$',
            r'(?i)^\s*Terms\s*$',
            r'(?i)^\s*Privacy\s+Policy\s*$',
            r'(?i)^\s*Help\s*$',
            r'(?i)^\s*View\s+\d+\s+more\s+classifications?\s*$',
            r'(?i)^\s*\*?\s*Cited\s+by\s+examiner.*$',
            r'(?i)^\s*Family\s+To\s+Family\s+Citations\s*$',
            r'(?i)^\s*Data\s+provided\s+by\b.*$',
            r'(?i)^\s*Patent\s+Citations\s*$',
            r'(?i)^\s*Non[- ]Patent\s+Citations\s*$',
            r'(?i)^\s*Cited\s+By\s*$',
            r'(?i)^\s*Classifications\s*$',
            r'(?i)^\s*Legal\s+Events\s*$',
            r'(?i)^\s*Concepts\s*$',
            r'(?i)^\s*Worldwide\s+applications\s*$',
            r'(?i)^\s*Discuss\s*$',
            r'(?i)^\s*Add\s+to\s+list\s*$',
            r'(?i)^\s*Google\s+Patents\s*$',
            r'(?i)^\s*Advanced\s+search\s*$',
            r'(?i)^\s*My\s+account\s*$',
            r'(?i)^\s*Sign\s+in\s*$',
            # Google Patents page counters
            r'^\s*\d+\s+of\s+\d+\s*$',
            # Patent document patterns
            r'(?i)^\s*Sheet\s+\d+\s+of\s+\d+\s*$',
            r'^\s*US\d+[A-Z]\d?\s*$',  # Patent number standalone
            r'(?i)^\s*United\s+States\s+Patent\s*$',
            r'(?i)^\s*\(\d{2}\)\s*Patent\s+No\.\s*$',
            # Web page URL footers
            r'^\s*https?://\S+\s*$',
            # Page indicators from print
            r'^\s*\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?\s*$',
            r'(?i)^\s*\d{1,2}/\d{1,2}/\d{2,4},?\s*.*patents\.google\.com.*$',
        ]
        
        cleaned = []
        for line in lines:
            is_artifact = False
            for pattern in artifact_patterns:
                if re.match(pattern, line.strip()):
                    is_artifact = True
                    break
            if not is_artifact:
                cleaned.append(line)
        return cleaned

    def _remove_leader_dots(self, lines: list) -> list:
        """Remove TOC-style leader dots: 'Chapter 1 ............. 5'"""
        leader_pattern = r'^\s*.+\s*[\.•·]{3,}\s*\d+\s*$'
        return [line for line in lines if not re.match(leader_pattern, line)]

    def remove_from_structured(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """Remove artifacts from structured JSON data as well."""
        if "sections" not in structured:
            return structured
        
        cleaned_sections = []
        for section in structured["sections"]:
            sec_type = section.get("type", "")
            content = section.get("content", "")
            
            # Skip sections that look like artifacts
            if sec_type == "paragraph":
                stripped = content.strip()
                if not stripped:
                    continue
                if re.match(r'^\s*\d+\s*$', stripped):
                    continue
                if re.match(r'(?i)^\s*(draft|confidential|watermark)\s*$', stripped):
                    continue
            
            cleaned_sections.append(section)
        
        structured["sections"] = cleaned_sections
        return structured
