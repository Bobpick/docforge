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
        
        # Strategy 1: Remove page number patterns
        lines = self._remove_page_numbers(lines)
        
        # Strategy 2: Remove repeated header/footer lines
        lines = self._remove_repeated_lines(lines)
        
        # Strategy 3: Remove common artifact patterns
        lines = self._remove_artifact_patterns(lines)
        
        # Strategy 4: Clean up section-page artifacts (e.g., "Section Title .... 5")
        lines = self._remove_leader_dots(lines)
        
        # Clean up excessive blank lines
        result = "\n".join(lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

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

    def _remove_repeated_lines(self, lines: list) -> list:
        """Remove lines that repeat across many 'pages' (heuristic).
        
        In markdown converted from multi-page documents, page breaks may be
        represented by horizontal rules or blank lines. We look for lines that
        appear many times and are short (typical of headers/footers).
        """
        line_counts = Counter(lines)
        
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Keep lines that are:
            # - Empty (preserve spacing)
            # - Don't repeat too many times
            # - Are long enough to be content
            # - Are structural markdown
            if not stripped:
                cleaned.append(line)
                continue
            
            count = line_counts.get(line, 1)
            is_short = len(stripped) < 80
            is_repeated = count >= self.min_repetition
            
            if is_repeated and is_short and not self._is_structural(stripped):
                continue  # Skip this artifact
            
            cleaned.append(line)
        return cleaned

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
        """Remove common artifact patterns like 'Confidential', 'Draft', etc."""
        artifact_patterns = [
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
