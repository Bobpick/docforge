"""DocForge — main conversion engine that orchestrates all handlers."""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .utils import detect_file_type, ensure_output_dir
from .pdf_handler import PDFHandler
from .docx_handler import DOCXHandler
from .pptx_handler import PPTXHandler
from .image_handler import ImageHandler
from .artifact_remover import ArtifactRemover
from .llm_enhancer import LLMEnhancer


class ConversionResult:
    """Holds the results of a single document conversion."""

    def __init__(
        self,
        markdown: str,
        structured: Dict[str, Any],
        images: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ):
        self.markdown = markdown
        self.structured = structured
        self.images = images
        self.metadata = metadata

    def save(self, output_dir: str) -> Dict[str, str]:
        """Save conversion results to disk.

        Creates:
            - output.md   — the Markdown document
            - output.json  — structured JSON
            - images/      — extracted images (if any)

        Returns:
            Dictionary mapping label → saved file path.
        """
        output_path = ensure_output_dir(output_dir)
        saved: Dict[str, str] = {}

        # --- Markdown ---
        md_path = output_path / "output.md"
        md_path.write_text(self.markdown, encoding="utf-8")
        saved["markdown"] = str(md_path)

        # --- JSON ---
        json_path = output_path / "output.json"
        json_path.write_text(
            json.dumps(self.structured, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        saved["json"] = str(json_path)

        # --- Images ---
        if self.images:
            images_dir = output_path / "images"
            images_dir.mkdir(exist_ok=True)
            for img in self.images:
                img_path = images_dir / img["filename"]
                if "data" in img:
                    img_path.write_bytes(img["data"])
                saved[f"image_{img['filename']}"] = str(img_path)

        return saved

    def __repr__(self) -> str:
        n_images = len(self.images)
        n_sections = len(self.structured.get("sections", []))
        return (
            f"ConversionResult(sections={n_sections}, images={n_images}, "
            f"markdown_len={len(self.markdown)})"
        )


class DocForge:
    """Main document conversion engine.

    Usage::

        forge = DocForge()
        result = forge.convert("report.pdf", output_dir="output/report")
        print(result.markdown)

    For batch processing::

        results = forge.convert_batch("documents/", output_dir="converted/")

    With LLM enhancement::

        forge = DocForge(use_llm=True, llm_api_key="...")
        result = forge.convert("messy.pdf")
    """

    HANDLERS: Dict[str, type] = {
        "pdf": PDFHandler,
        "docx": DOCXHandler,
        "pptx": PPTXHandler,
        "image": ImageHandler,
    }

    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_host: Optional[str] = None,
        # Legacy params kept for backward compatibility
        remove_artifacts: bool = True,
        extract_images: bool = True,
        extract_tables: bool = False,
    ):
        """
        Args:
            use_llm: Enable LLM-based enhancement for formatting cleanup.
            llm_provider: LLM provider — "ollama" (default), "gemini", or "openai-compat".
            llm_model: Model name. Defaults: "cogito:14b" (ollama), "gemini-2.0-flash" (gemini).
            llm_api_key: API key (Gemini only). Falls back to GOOGLE_API_KEY env var.
            llm_host: Ollama host URL (default: http://localhost:11434).
            remove_artifacts: Remove headers, footers, page numbers.
            extract_images: Extract and save embedded images.
            extract_tables: Extract tables (default False — multi-col PDFs shredded badly).
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.remove_artifacts = remove_artifacts
        self.extract_images = extract_images
        self.extract_tables = extract_tables

        self.artifact_remover = ArtifactRemover() if remove_artifacts else None
        self.llm_enhancer: Optional[LLMEnhancer] = None
        if use_llm:
            self.llm_enhancer = LLMEnhancer(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                host=llm_host,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> ConversionResult:
        """Convert a single document to Markdown + JSON.

        Args:
            file_path: Path to the source document.
            output_dir: If given, automatically save results to this directory.
            source_name: Override for the source file name in metadata.
                         Useful when the actual file is a temp file but you
                         want to preserve the original upload name.

        Returns:
            A ConversionResult with markdown, structured JSON, images, metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = detect_file_type(path)
        handler_class = self.HANDLERS.get(file_type)
        if not handler_class:
            raise ValueError(
                f"Unsupported file type: {path.suffix!r}. "
                f"Supported: {self.supported_extensions}"
            )

        start = time.time()
        try:
            handler = handler_class(
                extract_images=self.extract_images,
                extract_tables=self.extract_tables,
            )
        except TypeError:
            # Older handlers without extract_tables kwarg
            handler = handler_class(extract_images=self.extract_images)
            if hasattr(handler, "extract_tables"):
                handler.extract_tables = self.extract_tables
        markdown, structured, images, metadata = handler.convert(path)

        # --- Artifact removal ---
        if self.artifact_remover:
            markdown = self.artifact_remover.remove(markdown, structured)
            structured = self.artifact_remover.remove_from_structured(structured)

        # --- Garbage cleanup (shredded tables, bold spans, hard wraps) ---
        try:
            from .garbage_filter import (
                clean_extracted_markdown,
                clean_section_content,
            )

            markdown = clean_extracted_markdown(markdown)
            if isinstance(structured, dict) and "sections" in structured:
                kept = []
                for s in structured.get("sections") or []:
                    s2 = clean_section_content(s)
                    if s2 is not None:
                        kept.append(s2)
                structured["sections"] = kept
        except Exception:
            pass

        # --- LLM enhancement ---
        if self.llm_enhancer:
            markdown = self.llm_enhancer.enhance(markdown, structured)

        # --- Finalize ---
        structured["markdown"] = markdown
        display_name = source_name or path.name
        structured["metadata"] = {
            **metadata,
            "conversion_date": datetime.now().isoformat(),
            "source_file": display_name,
            "use_llm": self.use_llm,
            "artifacts_removed": self.remove_artifacts,
            "conversion_time_seconds": round(time.time() - start, 2),
        }

        result = ConversionResult(
            markdown=markdown,
            structured=structured,
            images=images,
            metadata=structured["metadata"],
        )

        if output_dir:
            result.save(output_dir)

        return result

    def convert_batch(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """Convert all supported documents in a directory.

        Args:
            input_dir: Directory containing source documents.
            output_dir: Base directory for converted output.
            recursive: Search subdirectories.

        Returns:
            Dict mapping relative file path → ConversionResult (or error string).
        """
        input_path = Path(input_dir)
        if not input_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")

        results: Dict[str, Any] = {}
        pattern = "**/*" if recursive else "*"

        for file_path in sorted(input_path.glob(pattern)):
            if not file_path.is_file():
                continue
            if detect_file_type(file_path) not in self.HANDLERS:
                continue

            rel_path = file_path.relative_to(input_path)
            file_output_dir = Path(output_dir) / rel_path.with_suffix("")

            try:
                result = self.convert(str(file_path), output_dir=str(file_output_dir))
                results[str(rel_path)] = result
            except Exception as e:
                results[str(rel_path)] = f"Error: {e}"

        return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    @property
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions."""
        from .utils import SUPPORTED_EXTENSIONS
        return sorted(SUPPORTED_EXTENSIONS.keys())

    def is_llm_available(self) -> bool:
        """Check whether the LLM enhancer is functional."""
        if not self.llm_enhancer:
            return False
        return self.llm_enhancer.is_available()
