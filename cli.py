#!/usr/bin/env python3
"""DocForge CLI — Batch document conversion from the command line.

Usage:
    # Single file
    python cli.py convert report.pdf -o output/

    # Entire folder
    python cli.py batch documents/ -o converted/

    # With LLM enhancement
    python cli.py convert messy.pdf --llm --api-key YOUR_KEY

    # Recursive folder processing
    python cli.py batch docs/ -o out/ --recursive

    # List supported formats
    python cli.py formats
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import click

from docforge.converter import DocForge, ConversionResult


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def print_banner():
    click.echo(click.style(
        "\n"
        "  ╔═══════════════════════════════════════╗\n"
        "  ║           📄 DocForge v1.0            ║\n"
        "  ║   Document → Markdown & JSON Engine   ║\n"
        "  ╚═══════════════════════════════════════╝\n",
        fg="cyan",
    ))


def format_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def print_result_summary(result: ConversionResult, elapsed: float):
    """Print a summary of a conversion result."""
    stats = result.structured.get("stats", {})
    meta = result.metadata

    click.echo(click.style("\n  ✅ Conversion successful!", fg="green", bold=True))
    click.echo(f"  ⏱  Time: {elapsed:.2f}s")
    click.echo(f"  📄 Source: {meta.get('source_file', 'unknown')}")

    if stats:
        click.echo(f"  📊 Sections: {stats.get('total_sections', 0)} total")
        for key, val in stats.items():
            if key != "total_sections" and val > 0:
                click.echo(f"     • {key.replace('_', ' ').title()}: {val}")

    click.echo(f"  📝 Markdown: {format_size(len(result.markdown.encode('utf-8')))}")
    if result.images:
        click.echo(f"  🖼️  Images: {len(result.images)}")


# ──────────────────────────────────────────────────────────────────────
# CLI commands
# ──────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """DocForge — Document to Markdown & JSON converter."""
    pass


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default=None, help="Output directory (default: ./output/<filename>)")
@click.option("--no-artifacts", is_flag=True, help="Keep headers, footers, page numbers")
@click.option("--no-images", is_flag=True, help="Skip image extraction")
@click.option("--llm", is_flag=True, help="Enable LLM (Gemini) enhancement")
@click.option("--api-key", default=None, help="Google AI API key (or set GOOGLE_API_KEY)")
@click.option("--model", default="gemini-2.0-flash", help="Gemini model name")
@click.option("--quiet", "-q", is_flag=True, help="Suppress banner and summary")
def convert(file_path, output_dir, no_artifacts, no_images, llm, api_key, model, quiet):
    """Convert a single document to Markdown and JSON."""
    if not quiet:
        print_banner()

    # Default output directory
    if output_dir is None:
        stem = Path(file_path).stem
        output_dir = str(Path("output") / stem)

    forge = DocForge(
        use_llm=llm,
        llm_api_key=api_key,
        llm_model=model,
        remove_artifacts=not no_artifacts,
        extract_images=not no_images,
    )

    if llm and not forge.is_llm_available():
        click.echo(click.style("  ⚠️  LLM enhancement requested but not available. "
                               "Set GOOGLE_API_KEY or pass --api-key.", fg="yellow"))

    start = time.time()
    try:
        result = forge.convert(file_path, output_dir=output_dir)
    except Exception as e:
        click.echo(click.style(f"  ❌ Error: {e}", fg="red", bold=True))
        sys.exit(1)
    elapsed = time.time() - start

    if not quiet:
        print_result_summary(result, elapsed)

    click.echo(f"\n  📁 Output saved to: {click.style(output_dir, fg='cyan', bold=True)}")
    click.echo(f"     ├── output.md")
    click.echo(f"     ├── output.json")
    if result.images:
        click.echo(f"     └── images/ ({len(result.images)} files)")


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output-dir", required=True, help="Output base directory")
@click.option("--recursive", "-r", is_flag=True, help="Process subdirectories")
@click.option("--no-artifacts", is_flag=True, help="Keep headers, footers, page numbers")
@click.option("--no-images", is_flag=True, help="Skip image extraction")
@click.option("--llm", is_flag=True, help="Enable LLM (Gemini) enhancement")
@click.option("--api-key", default=None, help="Google AI API key (or set GOOGLE_API_KEY)")
@click.option("--model", default="gemini-2.0-flash", help="Gemini model name")
def batch(input_dir, output_dir, recursive, no_artifacts, no_images, llm, api_key, model):
    """Batch-convert all supported documents in a directory."""
    print_banner()

    forge = DocForge(
        use_llm=llm,
        llm_api_key=api_key,
        llm_model=model,
        remove_artifacts=not no_artifacts,
        extract_images=not no_images,
    )

    click.echo(f"  📂 Input:  {click.style(input_dir, fg='cyan')}")
    click.echo(f"  📁 Output: {click.style(output_dir, fg='cyan')}")
    click.echo(f"  🔄 Recursive: {'Yes' if recursive else 'No'}")
    click.echo()

    start = time.time()
    try:
        results = forge.convert_batch(input_dir, output_dir, recursive=recursive)
    except Exception as e:
        click.echo(click.style(f"  ❌ Error: {e}", fg="red", bold=True))
        sys.exit(1)
    elapsed = time.time() - start

    # Summary
    successes = [r for r in results.values() if isinstance(r, ConversionResult)]
    failures = {k: v for k, v in results.items() if isinstance(v, str)}

    click.echo(click.style(f"\n  ✅ Batch complete in {elapsed:.1f}s", fg="green", bold=True))
    click.echo(f"  📄 Converted: {len(successes)} files")
    if failures:
        click.echo(click.style(f"  ❌ Failed: {len(failures)} files", fg="red"))
        for path, error in failures.items():
            click.echo(f"     • {path}: {error}")

    # Detailed results
    for rel_path, result in results.items():
        if isinstance(result, ConversionResult):
            stats = result.structured.get("stats", {})
            sections = stats.get("total_sections", 0)
            click.echo(f"  ✓ {rel_path} — {sections} sections, {len(result.images)} images")


@cli.command()
def formats():
    """List all supported file formats."""
    print_banner()
    click.echo("  Supported file formats:\n")

    from docforge.utils import SUPPORTED_EXTENSIONS

    # Group by category
    categories = {}
    for ext, cat in SUPPORTED_EXTENSIONS.items():
        categories.setdefault(cat, []).append(ext)

    for cat, exts in sorted(categories.items()):
        ext_list = ", ".join(exts)
        click.echo(f"  {click.style(cat.upper(), fg='cyan', bold=True):12s} {ext_list}")

    click.echo()


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
