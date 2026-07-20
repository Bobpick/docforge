#!/usr/bin/env python3
"""DocForge CLI — Batch document conversion from the command line.

Usage:
    # Single file
    python cli.py convert report.pdf -o output/

    # With Ollama (default LLM provider)
    python cli.py convert messy.pdf --llm
    python cli.py convert messy.pdf --llm --model llama3:8b

    # With Gemini (cloud)
    python cli.py convert messy.pdf --llm --provider gemini --api-key YOUR_KEY

    # Entire folder
    python cli.py batch documents/ -o converted/

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
from docforge.llm_enhancer import OllamaServiceManager


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


def get_default_model(provider: str) -> str:
    """Get the default model for a provider."""
    if provider == "ollama":
        return os.environ.get("DOCFORGE_OLLAMA_MODEL", "cogito:14b")
    elif provider == "gemini":
        return "gemini-2.0-flash"
    else:
        return "local-model"


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
@click.option("--llm", is_flag=True, help="Enable LLM enhancement")
@click.option("--provider", type=click.Choice(["ollama", "gemini", "openai-compat"]), default="ollama",
              help="LLM provider (default: ollama)")
@click.option("--model", default=None, help="LLM model name (default: cogito:14b for ollama)")
@click.option("--api-key", default=None, help="API key for cloud providers (Gemini)")
@click.option("--ollama-host", default=None, help="Ollama host URL (default: http://localhost:11434)")
@click.option("--quiet", "-q", is_flag=True, help="Suppress banner and summary")
def convert(file_path, output_dir, no_artifacts, no_images, llm, provider, model, api_key, ollama_host, quiet):
    """Convert a single document to Markdown and JSON."""
    if not quiet:
        print_banner()

    # Default output directory
    if output_dir is None:
        stem = Path(file_path).stem
        output_dir = str(Path("output") / stem)

    # Resolve model default
    if model is None:
        model = get_default_model(provider)

    forge = DocForge(
        use_llm=llm,
        llm_provider=provider,
        llm_model=model,
        llm_api_key=api_key,
        llm_host=ollama_host,
        remove_artifacts=not no_artifacts,
        extract_images=not no_images,
    )

    if llm:
        if not forge.is_llm_available():
            if provider == "ollama":
                click.echo(click.style(
                    "  ⚠️  Ollama not detected. Make sure it's running:\n"
                    "     ollama serve\n"
                    f"     ollama pull {model}", fg="yellow"))
            elif provider == "gemini":
                click.echo(click.style(
                    "  ⚠️  Gemini not available. Set GOOGLE_API_KEY or pass --api-key.",
                    fg="yellow"))
            else:
                click.echo(click.style(
                    "  ⚠️  LLM provider not available. Check your configuration.",
                    fg="yellow"))
        else:
            if not quiet:
                provider_label = {
                    "ollama": "🦙 Ollama",
                    "gemini": "✨ Gemini",
                    "openai-compat": "🔌 OpenAI-compat",
                }.get(provider, provider)
                click.echo(f"  🤖 LLM: {provider_label} / {model}")

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
@click.option("--llm", is_flag=True, help="Enable LLM enhancement")
@click.option("--provider", type=click.Choice(["ollama", "gemini", "openai-compat"]), default="ollama",
              help="LLM provider (default: ollama)")
@click.option("--model", default=None, help="LLM model name (default: cogito:14b for ollama)")
@click.option("--api-key", default=None, help="API key for cloud providers")
@click.option("--ollama-host", default=None, help="Ollama host URL")
def batch(input_dir, output_dir, recursive, no_artifacts, no_images, llm, provider, model, api_key, ollama_host):
    """Batch-convert all supported documents in a directory."""
    print_banner()

    if model is None:
        model = get_default_model(provider)

    forge = DocForge(
        use_llm=llm,
        llm_provider=provider,
        llm_model=model,
        llm_api_key=api_key,
        llm_host=ollama_host,
        remove_artifacts=not no_artifacts,
        extract_images=not no_images,
    )

    click.echo(f"  📂 Input:  {click.style(input_dir, fg='cyan')}")
    click.echo(f"  📁 Output: {click.style(output_dir, fg='cyan')}")
    click.echo(f"  🔄 Recursive: {'Yes' if recursive else 'No'}")
    if llm:
        click.echo(f"  🤖 LLM: {provider} / {model}")
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


@cli.group()
def ollama():
    """Manage the Ollama LLM service (start, stop, restart, status)."""
    pass


@ollama.command("status")
@click.option("--host", default="http://127.0.0.1:11434", help="Ollama API URL")
def ollama_status(host):
    """Check Ollama service status."""
    mgr = OllamaServiceManager(host=host)
    info = mgr.status()

    if info["running"]:
        click.echo(click.style("  🟢 Ollama is running", fg="green", bold=True))
        if info.get("pid"):
            click.echo(f"  PID: {info['pid']}")
        click.echo(f"  Bind: {info.get('bind')}  (listen_all={info.get('listen_all')})")
        click.echo(f"  API:  {info.get('client')}")
        click.echo(f"  systemd active: {info.get('via_systemd')}")
        if info.get("models"):
            click.echo(f"  Models: {', '.join(info['models'])}")
        if info.get("gpu"):
            click.echo(f"  GPU: Yes ({info.get('memory_mb', 0):.0f} MiB VRAM)")
        else:
            click.echo("  GPU: No (CPU mode)")
    else:
        click.echo(click.style("  🔴 Ollama is not running", fg="red", bold=True))
        click.echo("  Start with: python cli.py ollama start")


@ollama.command("start")
@click.option("--host", default="http://127.0.0.1:11434", help="Ollama API URL")
@click.option("--model", default=None, help="Model to pre-load (e.g. cogito:14b)")
def ollama_start(host, model):
    """Start Ollama on 0.0.0.0:11434."""
    mgr = OllamaServiceManager(host=host)
    result = mgr.start(model=model, force_bind=True)

    if result["success"]:
        click.echo(click.style(f"  ✅ {result['message']}", fg="green", bold=True))
        if result.get("models"):
            click.echo(f"  Models: {', '.join(result['models'])}")
    else:
        click.echo(click.style(f"  ❌ {result['message']}", fg="red", bold=True))
        sys.exit(1)


@ollama.command("stop")
@click.option("--host", default="http://127.0.0.1:11434", help="Ollama API URL")
def ollama_stop(host):
    """Stop the Ollama service (systemd + processes)."""
    mgr = OllamaServiceManager(host=host)
    result = mgr.stop()

    if result["success"]:
        click.echo(click.style(f"  ✅ {result['message']}", fg="green", bold=True))
    else:
        click.echo(click.style(f"  ❌ {result['message']}", fg="red", bold=True))
        sys.exit(1)


@ollama.command("restart")
@click.option("--host", default="http://127.0.0.1:11434", help="Ollama API URL")
@click.option("--model", default=None, help="Model to pre-load after restart")
def ollama_restart(host, model):
    """Stop everything, then start on 0.0.0.0:11434."""
    mgr = OllamaServiceManager(host=host)
    result = mgr.restart(model=model)

    if result["success"]:
        click.echo(click.style(f"  ✅ {result['message']}", fg="green", bold=True))
    else:
        click.echo(click.style(f"  ❌ {result['message']}", fg="red", bold=True))
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
