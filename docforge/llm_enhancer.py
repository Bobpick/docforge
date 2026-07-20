"""LLM-based enhancement for document conversion cleanup using Google Gemini."""

import os
import re
import json
from typing import Optional, Dict, Any

from .utils import truncate_text, chunk_text, clean_whitespace


class LLMEnhancer:
    """Use an LLM (Gemini) to clean up and enhance conversion output."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        """
        Args:
            api_key: Google AI API key. Falls back to GOOGLE_API_KEY env var.
            model: Gemini model name to use.
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Gemini client."""
        if self._client is not None:
            return self._client

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for LLM enhancement. "
                "Install it with: pip install google-generativeai"
            )

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent cleanup
                max_output_tokens=65536,
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )
        return self._client

    def enhance(self, markdown: str, structured: Optional[Dict[str, Any]] = None) -> str:
        """Enhance markdown output using LLM.
        
        Applies multiple enhancement passes:
        1. Table cleanup and restructuring
        2. Math equation formatting
        3. Code block detection and formatting
        4. General formatting improvements
        
        Args:
            markdown: The markdown text to enhance.
            structured: Optional structured data for context.
            
        Returns:
            Enhanced markdown text.
        """
        if not markdown.strip():
            return markdown

        # Check if there's anything that needs enhancement
        needs_enhancement = self._needs_enhancement(markdown)
        if not needs_enhancement:
            return markdown

        try:
            # Enhance tables first (most impactful)
            markdown = self._enhance_tables(markdown)
            
            # Enhance math and code
            markdown = self._enhance_math_and_code(markdown)
            
            # Final cleanup pass
            markdown = self._final_cleanup(markdown)
            
        except Exception as e:
            # If LLM fails, return the original with a comment
            markdown = f"{markdown}\n\n<!-- LLM enhancement failed: {e} -->"

        return markdown

    def _needs_enhancement(self, markdown: str) -> bool:
        """Check if the markdown would benefit from LLM enhancement."""
        # Look for messy tables, unformatted equations, etc.
        indicators = [
            # Messy table indicators
            r'\|.*\|.*\n(?!\|[-\s:|]+)',  # Table row without separator
            r'(\S\t){2,}',                  # Tab-separated data without markdown table
            # Unformatted math
            r'[a-zA-Z]\^[0-9]',            # x2 instead of x^2
            r'[a-zA-Z]_[0-9]',             # x2 instead of x_2
            # Garbled text from OCR
            r'[^\x00-\x7F]{3,}',           # Long non-ASCII runs (OCR artifacts)
        ]
        for pattern in indicators:
            if re.search(pattern, markdown):
                return True
        return True  # Default: always try to enhance

    def _enhance_tables(self, markdown: str) -> str:
        """Use LLM to clean up messy tables."""
        # Extract table sections from the markdown
        table_sections = self._extract_table_sections(markdown)
        if not table_sections:
            return markdown

        result = markdown
        for table_text, start, end in reversed(table_sections):
            enhanced = self._llm_enhance_table(table_text)
            if enhanced:
                result = result[:start] + enhanced + result[end:]

        return result

    def _extract_table_sections(self, markdown: str) -> list:
        """Extract table sections from markdown for individual enhancement."""
        tables = []
        lines = markdown.split("\n")
        in_table = False
        table_start = 0
        table_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("|") and "|" in stripped[1:]:
                if not in_table:
                    in_table = True
                    table_start = sum(len(l) + 1 for l in lines[:i])
                table_lines.append(line)
            elif in_table:
                # End of table
                table_text = "\n".join(table_lines)
                table_end = table_start + len(table_text)
                tables.append((table_text, table_start, table_end))
                in_table = False
                table_lines = []

        if in_table and table_lines:
            table_text = "\n".join(table_lines)
            table_end = table_start + len(table_text)
            tables.append((table_text, table_start, table_end))

        return tables

    def _enhance_math_and_code(self, markdown: str) -> str:
        """Use LLM to improve math equations and code block formatting."""
        # Only process if there are indicators of unformatted math/code
        has_math = bool(re.search(r'[\\^_{}]', markdown))
        has_code = bool(re.search(r'(def |function |class |import |#include )', markdown))

        if not has_math and not has_code:
            return markdown

        # If the text is very long, process in chunks
        if len(markdown) > 15000:
            return self._enhance_in_chunks(markdown)

        prompt = f"""You are a document formatting expert. Clean up the following markdown text by:

1. Formatting any mathematical expressions with proper LaTeX notation:
   - Inline math: $expression$
   - Display math: $$expression$$
   - Fix common OCR errors in math (e.g., "x2" → "$x^2$", "alpha" → "$\\alpha$")

2. Wrapping any detected code in proper fenced code blocks with language tags:
   - ```python ... ```
   - ```javascript ... ```
   - etc.

3. Do NOT change any other formatting, headings, or content.
4. Do NOT add or remove information.
5. Return ONLY the improved markdown text, nothing else.

Markdown to improve:

{markdown}"""

        try:
            client = self._get_client()
            response = client.generate_content(prompt)
            result = response.text.strip()
            # Remove markdown code fences if the LLM wrapped its response
            result = re.sub(r'^```\n?', '', result)
            result = re.sub(r'\n?```$', '', result)
            return result
        except Exception:
            return markdown

    def _enhance_in_chunks(self, markdown: str) -> str:
        """Enhance long documents by processing in chunks."""
        chunks = chunk_text(markdown, chunk_size=12000, overlap=200)
        enhanced_chunks = []

        for chunk in chunks:
            try:
                prompt = f"""You are a document formatting expert. Clean up the following markdown fragment by:
1. Formatting math expressions with LaTeX ($...$ or $$...$$)
2. Wrapping code in fenced code blocks with language tags
3. Fix any OCR artifacts
4. Do NOT change headings or other content
5. Return ONLY the improved markdown, nothing else.

Fragment:
{chunk}"""

                client = self._get_client()
                response = client.generate_content(prompt)
                result = response.text.strip()
                result = re.sub(r'^```\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
                enhanced_chunks.append(result)
            except Exception:
                enhanced_chunks.append(chunk)

        return "\n\n".join(enhanced_chunks)

    def _llm_enhance_table(self, table_text: str) -> Optional[str]:
        """Use LLM to clean up a single table."""
        prompt = f"""You are a table formatting expert. The following markdown table may have issues from document conversion:
- Missing or misaligned columns
- Merged cells that need to be handled
- Garbled text from OCR
- Missing separator rows

Please fix and return ONLY the corrected markdown table. Preserve all data.

Table to fix:
{table_text}"""

        try:
            client = self._get_client()
            response = client.generate_content(prompt)
            result = response.text.strip()
            result = re.sub(r'^```\n?', '', result)
            result = re.sub(r'\n?```$', '', result)
            return result
        except Exception:
            return None

    def _final_cleanup(self, markdown: str) -> str:
        """Final cleanup pass without LLM - pure regex-based."""
        # Fix double-escaped characters
        markdown = markdown.replace("\\\\(", "\\(")
        markdown = markdown.replace("\\\\)", "\\)")
        markdown = markdown.replace("\\\\[", "\\[")
        markdown = markdown.replace("\\\\]", "\\]")
        
        # Fix common math OCR artifacts
        markdown = re.sub(r'(\w)\s*\^\s*(\d)', r'\1^{\2}', markdown)
        markdown = re.sub(r'(\w)\s*_\s*(\d)', r'\1_{\2}', markdown)
        
        # Clean up excessive whitespace
        markdown = clean_whitespace(markdown)
        
        return markdown

    def is_available(self) -> bool:
        """Check if the LLM enhancer is available (has API key and library)."""
        if not self.api_key:
            return False
        try:
            import google.generativeai  # noqa: F401
            return True
        except ImportError:
            return False
