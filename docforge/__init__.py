"""DocForge — Powerful document-to-markdown conversion tool.

Convert PDFs, Word docs, PowerPoints, and images into clean Markdown
and structured JSON, with optional LLM-powered enhancement.
"""

__version__ = "1.0.0"
__author__ = "DocForge"

from .converter import DocForge, ConversionResult

__all__ = ["DocForge", "ConversionResult"]
