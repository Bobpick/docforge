"""Shared utilities for DocForge."""

import os
import re
import hashlib
from pathlib import Path
from typing import Optional


# Supported file extensions and their categories
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".pptx": "pptx",
    ".ppt": "pptx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".tif": "image",
    ".bmp": "image",
    ".webp": "image",
    ".gif": "image",
}


def detect_file_type(file_path: Path) -> str:
    """Detect the type of a file based on its extension.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        Category string: 'pdf', 'docx', 'pptx', 'image', or the extension itself.
    """
    ext = file_path.suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext, ext)


def ensure_output_dir(output_dir: str) -> Path:
    """Create output directory if it doesn't exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """Sanitize a string for use as a filename."""
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r'[^\w\s-]', '_', name)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    sanitized = sanitized.strip('_')
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized or "untitled"


def generate_image_filename(page_num: int, index: int, ext: str = "png") -> str:
    """Generate a consistent filename for an extracted image."""
    return f"page{page_num:03d}_img{index:03d}.{ext}"


def compute_hash(data: bytes) -> str:
    """Compute a short hash for deduplication."""
    return hashlib.md5(data).hexdigest()[:12]


def is_duplicate_image(img_data: bytes, seen_hashes: set) -> bool:
    """Check if an image is a duplicate based on its hash."""
    h = compute_hash(img_data)
    if h in seen_hashes:
        return True
    seen_hashes.add(h)
    return False


def format_table_md(headers: list, rows: list) -> str:
    """Format a table as Markdown.
    
    Args:
        headers: List of header cell strings.
        rows: List of rows, each a list of cell strings.
        
    Returns:
        Markdown table string.
    """
    if not headers and not rows:
        return ""
    
    # Determine column count
    num_cols = max(len(headers), max((len(r) for r in rows), default=0))
    
    # Pad headers and rows
    headers = list(headers) + [""] * (num_cols - len(headers))
    rows = [list(r) + [""] * (num_cols - len(r)) for r in rows]
    
    # Clean cell content (replace newlines with spaces, strip)
    headers = [str(h).replace("\n", " ").strip() for h in headers]
    rows = [[str(c).replace("\n", " ").strip() for c in r] for r in rows]
    
    # Build markdown table
    lines = []
    
    # Header row
    header_line = "| " + " | ".join(headers) + " |"
    lines.append(header_line)
    
    # Separator row
    sep_line = "|" + "|".join([" --- " for _ in range(num_cols)]) + "|"
    lines.append(sep_line)
    
    # Data rows
    for row in rows:
        row_line = "| " + " | ".join(row) + " |"
        lines.append(row_line)
    
    return "\n".join(lines)


def format_table_json(headers: list, rows: list) -> dict:
    """Format a table as a structured JSON object."""
    return {
        "type": "table",
        "headers": [str(h).replace("\n", " ").strip() for h in headers],
        "rows": [[str(c).replace("\n", " ").strip() for c in r] for r in rows],
    }


def detect_math_expression(text: str) -> bool:
    """Heuristic to detect LaTeX or math expressions in text."""
    math_indicators = [
        r'\\frac', r'\\sqrt', r'\\sum', r'\\int', r'\\prod',
        r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', r'\\pi',
        r'\\infty', r'\\partial', r'\\nabla',
        r'\\begin\{equation', r'\\begin\{align',
        r'\\[a-zA-Z]+\{',  # LaTeX commands with braces
        r'\^{[^}]+}',       # Superscripts
        r'_{[^}]+}',        # Subscripts
    ]
    for pattern in math_indicators:
        if re.search(pattern, text):
            return True
    return False


def escape_markdown(text: str) -> str:
    """Escape special Markdown characters in plain text."""
    # Only escape characters that would be misinterpreted
    # Don't escape * or _ within words
    text = re.sub(r'(?<!\w)([#|])', r'\\\1', text)
    return text


def truncate_text(text: str, max_chars: int = 30000) -> str:
    """Truncate text to a maximum number of characters for LLM context limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n... [TRUNCATED due to length] ..."


def chunk_text(text: str, chunk_size: int = 15000, overlap: int = 500) -> list:
    """Split text into overlapping chunks for LLM processing."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at a paragraph boundary
        if end < len(text):
            last_para = chunk.rfind("\n\n")
            if last_para > chunk_size // 2:
                end = start + last_para + 2
                chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def clean_whitespace(text: str) -> str:
    """Clean up excessive whitespace while preserving structure."""
    # Remove trailing whitespace from lines
    text = re.sub(r'[ \t]+\n', '\n', text)
    # Collapse multiple blank lines to max 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace
    return text.strip()


def merge_consecutive_elements(elements: list, same_type: callable = None) -> list:
    """Merge consecutive elements of the same type if a merge function is provided."""
    if not elements:
        return elements
    
    if same_type is None:
        return elements
    
    merged = [elements[0]]
    for elem in elements[1:]:
        if same_type(merged[-1], elem):
            merged[-1] = merge_elements(merged[-1], elem)
        else:
            merged.append(elem)
    return merged


def merge_elements(a: dict, b: dict) -> dict:
    """Merge two content elements."""
    if a.get("type") == "paragraph" and b.get("type") == "paragraph":
        return {
            "type": "paragraph",
            "content": a.get("content", "") + "\n" + b.get("content", ""),
        }
    return b  # Default: just take the second one
