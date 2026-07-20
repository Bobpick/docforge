#!/usr/bin/env python3
"""DocForge Web App — Batch document conversion powered by Streamlit.

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import time
import tempfile
import zipfile
import io
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure the docforge package is importable
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from docforge.converter import DocForge, ConversionResult
from docforge.llm_enhancer import LLMEnhancer, OllamaServiceManager


# ──────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocForge — Document Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; margin-bottom: 0; }
    .sub-header  { font-size: 1.1rem; color: #888; margin-top: 0; }
    .result-box  {
        background: #0d1117; color: #e6edf3; border-radius: 8px;
        padding: 1.2rem; font-family: 'Courier New', monospace;
        font-size: 0.85rem; white-space: pre-wrap; word-wrap: break-word;
        max-height: 600px; overflow-y: auto; border: 1px solid #30363d;
    }
    .stat-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 1rem; text-align: center;
    }
    .stat-number { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
    .stat-label  { font-size: 0.8rem; color: #8b949e; }
    .image-grid  { display: flex; flex-wrap: wrap; gap: 12px; }
    .image-grid img { max-height: 200px; border-radius: 6px; border: 1px solid #30363d; }
    .provider-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600;
    }
    .badge-ollama { background: #1a3a2a; color: #4ade80; }
    .badge-gemini { background: #1a2a3a; color: #60a5fa; }
    .badge-openai { background: #2a1a3a; color: #c084fc; }
    .file-chip {
        display: inline-flex; align-items: center; gap: 6px;
        background: #161b22; border: 1px solid #30363d; border-radius: 16px;
        padding: 4px 12px; font-size: 0.85rem; margin: 3px;
    }
    .file-chip .name { color: #e6edf3; }
    .file-chip .size { color: #8b949e; font-size: 0.75rem; }
    .file-chip .remove { color: #f85149; cursor: pointer; font-weight: bold; }
    .batch-progress {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 0.75rem 1rem; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────────────────────────────
if "batch_results" not in st.session_state:
    st.session_state.batch_results: Dict[str, Dict[str, Any]] = {}
if "file_queue" not in st.session_state:
    st.session_state.file_queue: list = []  # list of (name, size, type) tuples


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────
def _escape_html(text: str) -> str:
    """Escape HTML special chars for display in a div."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _file_size_str(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _build_zip(results: Dict[str, Dict[str, Any]]) -> bytes:
    """Build a ZIP archive containing all conversion results.

    Each file gets its own subfolder named after the source stem.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, res in results.items():
            stem = Path(filename).stem
            if res.get("error"):
                zf.writestr(f"{stem}/ERROR.txt", res["error"])
                continue
            # Markdown
            if res.get("markdown"):
                zf.writestr(f"{stem}/{stem}.md", res["markdown"])
            # JSON
            if res.get("structured"):
                zf.writestr(
                    f"{stem}/{stem}.json",
                    json.dumps(res["structured"], indent=2, ensure_ascii=False, default=str),
                )
            # Images
            if res.get("image_data"):
                for img_name, img_bytes in res["image_data"].items():
                    zf.writestr(f"{stem}/images/{img_name}", img_bytes)
    return buf.getvalue()


def _convert_one_file(
    uploaded_file,
    forge: DocForge,
    output_dir: str,
) -> Dict[str, Any]:
    """Convert a single uploaded file. Returns a dict with results or error."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        start = time.time()
        result = forge.convert(tmp_path, output_dir=output_dir, source_name=uploaded_file.name)
        elapsed = time.time() - start

        # Collect image bytes for ZIP download
        image_data = {}
        if result.images:
            images_dir = Path(output_dir) / "images"
            for img in result.images:
                img_path = images_dir / img["filename"]
                if img_path.exists():
                    image_data[img["filename"]] = img_path.read_bytes()

        return {
            "markdown": result.markdown,
            "structured": result.structured,
            "images": result.images,
            "image_data": image_data,
            "metadata": result.metadata,
            "elapsed": elapsed,
            "output_dir": output_dir,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────
# Sidebar — configuration
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚙️ Settings")

    remove_artifacts = st.checkbox(
        "Remove headers / footers / page numbers",
        value=True,
        help="Automatically strip repeated headers, footers, and page numbers.",
    )

    extract_images = st.checkbox(
        "Extract embedded images",
        value=True,
        help="Save images found inside the document.",
    )

    st.markdown("---")
    st.markdown("### 🤖 LLM Enhancement")

    use_llm = st.checkbox(
        "Enable LLM cleanup",
        value=False,
        help="Use an LLM to clean up tables, fix OCR errors, and improve formatting.",
    )

    llm_provider = st.selectbox(
        "LLM Provider",
        ["ollama", "gemini", "openai-compat"],
        index=0,
        help="Ollama runs locally (free, private). Gemini is cloud-based. "
             "OpenAI-compat works with LM Studio, vLLM, etc.",
    )

    # Provider-specific settings
    if llm_provider == "ollama":
        st.markdown(
            '<span class="provider-badge badge-ollama">🦙 Ollama — Local & Free</span>',
            unsafe_allow_html=True,
        )
        llm_model = st.text_input(
            "Model",
            value=os.environ.get("DOCFORGE_OLLAMA_MODEL", "cogito:14b"),
            help="Ollama model name. Pull with: ollama pull cogito:14b",
        )
        # Client URL (API). Server always binds 0.0.0.0:11434 via OllamaServiceManager.
        default_client = os.environ.get(
            "DOCFORGE_OLLAMA_CLIENT",
            "http://127.0.0.1:11434",
        )
        llm_host = st.text_input(
            "Ollama API URL",
            value=default_client,
            help="Where DocForge connects. Server is started on 0.0.0.0:11434.",
        )
        llm_api_key = None

        ollama_mgr = OllamaServiceManager(host=llm_host or "http://127.0.0.1:11434")

        # ── Bootstrap once per browser session: kill + start on 0.0.0.0:11434 ──
        if "ollama_bootstrapped" not in st.session_state:
            with st.spinner(
                "Resetting Ollama — stop existing instance, start on 0.0.0.0:11434…"
            ):
                boot = ollama_mgr.ensure_fresh(model=None)
            st.session_state.ollama_bootstrapped = True
            st.session_state.ollama_last_msg = boot.get("message", "")
            st.session_state.ollama_last_ok = bool(boot.get("success"))

        # Show last control message (survives rerun)
        if st.session_state.get("ollama_last_msg"):
            if st.session_state.get("ollama_last_ok", True):
                st.success(st.session_state.ollama_last_msg)
            else:
                st.error(st.session_state.ollama_last_msg)

        # Live status
        ollama_status = ollama_mgr.status()
        if ollama_status["running"]:
            listen = ollama_status.get("listen_all")
            bind_txt = (
                "0.0.0.0:11434"
                if listen
                else ("127.0.0.1 only" if listen is False else "unknown bind")
            )
            st.success(
                f"✅ Ollama running · PID {ollama_status.get('pid', '?')} · {bind_txt}"
            )
            if ollama_status.get("models"):
                with st.expander(f"Available models ({len(ollama_status['models'])})"):
                    for m in ollama_status["models"]:
                        st.text(m)
        else:
            st.warning("⚠️ Ollama not detected")

        # ── Ollama Service Management ──
        st.markdown("#### 🔧 Ollama Service")
        st.caption(
            f"Serve bind: `{ollama_mgr.bind}` · API: `{ollama_mgr.host}` "
            "· App startup always stops then restarts Ollama."
        )

        with st.expander("📋 Ollama Status", expanded=True):
            if ollama_status["running"]:
                st.write(f"**PID:** `{ollama_status.get('pid', '?')}`")
                st.write(f"**Bind:** `{ollama_mgr.bind}` (listen_all={ollama_status.get('listen_all')})")
                st.write(f"**systemd unit active:** `{ollama_status.get('via_systemd')}`")
                if ollama_status.get("models"):
                    st.write(
                        "**Models:** "
                        + ", ".join(f"`{m}`" for m in ollama_status["models"])
                    )
                if ollama_status.get("gpu"):
                    st.info(
                        f"🎮 GPU active  ·  {ollama_status.get('memory_mb', 0):.0f} MiB VRAM"
                    )
                else:
                    st.caption("CPU mode (no GPU compute app detected)")
            else:
                st.error("🔴 Not running")

        def _run_ollama_action(action: str):
            with st.spinner(f"{action.title()} Ollama…"):
                if action == "start":
                    result = ollama_mgr.start(model=llm_model, force_bind=True)
                elif action == "stop":
                    result = ollama_mgr.stop()
                else:
                    result = ollama_mgr.restart(model=llm_model)
            st.session_state.ollama_last_msg = result.get("message", "")
            st.session_state.ollama_last_ok = bool(result.get("success"))
            st.rerun()

        ollama_col1, ollama_col2, ollama_col3 = st.columns(3)
        with ollama_col1:
            if st.button("▶️ Start", key="ollama_start", use_container_width=True):
                _run_ollama_action("start")
        with ollama_col2:
            if st.button("⏹️ Stop", key="ollama_stop", use_container_width=True):
                _run_ollama_action("stop")
        with ollama_col3:
            if st.button("🔄 Restart", key="ollama_restart", use_container_width=True):
                _run_ollama_action("restart")

    elif llm_provider == "gemini":
        st.markdown(
            '<span class="provider-badge badge-gemini">✨ Gemini — Cloud</span>',
            unsafe_allow_html=True,
        )
        llm_model = st.text_input(
            "Model",
            value="gemini-2.0-flash",
            help="Gemini model name.",
        )
        llm_api_key = st.text_input(
            "Google AI API Key",
            type="password",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            help="Your Google AI API key. Also set via GOOGLE_API_KEY env var.",
        )
        llm_host = None

    else:  # openai-compat
        st.markdown(
            '<span class="provider-badge badge-openai">🔌 OpenAI-Compatible</span>',
            unsafe_allow_html=True,
        )
        llm_model = st.text_input("Model", value="local-model")
        llm_api_key = st.text_input("API Key", value="not-needed", type="password")
        llm_host = st.text_input(
            "API Base URL",
            value="http://localhost:8080/v1",
            help="Base URL for OpenAI-compatible API.",
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#666;'>"
        "DocForge v1.0 — PDF · DOCX · PPTX · Images → Markdown & JSON<br>"
        "Batch drag & drop supported 📂"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📄 DocForge</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Convert documents to clean Markdown & JSON — '
    'drag & drop multiple files for batch processing.</p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# File uploader — batch support
# ──────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📂 Upload documents (drag & drop multiple files)",
    type=["pdf", "docx", "doc", "pptx", "ppt", "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp", "gif"],
    accept_multiple_files=True,
    help="Drag and drop one or more files, or click to browse. "
         "Supports PDF, DOCX, PPTX, and images.",
)

# ── File queue display ──
if uploaded_files:
    st.markdown(f"### 📋 File Queue ({len(uploaded_files)} file{'s' if len(uploaded_files) != 1 else ''})")

    # Show files in a clean grid
    file_cols = st.columns(min(len(uploaded_files), 5))
    for i, uf in enumerate(uploaded_files):
        col = file_cols[i % len(file_cols)]
        with col:
            ext = Path(uf.name).suffix.lower().replace(".", "")
            icon = {"pdf": "📕", "docx": "📘", "doc": "📘", "pptx": "📙", "ppt": "📙"}.get(ext, "🖼️")
            st.markdown(
                f'<div class="file-chip">'
                f'<span class="name">{icon} {uf.name}</span>'
                f'<span class="size">{_file_size_str(uf.size)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Batch convert button ──
    st.markdown("---")

    col_convert, col_clear = st.columns([3, 1])
    with col_convert:
        convert_btn = st.button(
            f"🔄 Convert {len(uploaded_files)} File{'s' if len(uploaded_files) != 1 else ''}",
            type="primary",
            use_container_width=True,
        )
    with col_clear:
        clear_btn = st.button(
            "🗑️ Clear Results",
            use_container_width=True,
        )

    if clear_btn:
        st.session_state.batch_results = {}
        st.rerun()

    if convert_btn:
        # Clear previous results
        st.session_state.batch_results = {}

        forge = DocForge(
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_host=llm_host,
            remove_artifacts=remove_artifacts,
            extract_images=extract_images,
        )

        total = len(uploaded_files)
        success_count = 0
        error_count = 0

        # Progress bar + status
        progress_bar = st.progress(0, text=f"Converting 0/{total} files…")
        status_area = st.empty()

        for idx, uf in enumerate(uploaded_files):
            status_area.info(f"📄 Converting **{uf.name}** ({idx + 1}/{total})…")

            output_dir = tempfile.mkdtemp(prefix=f"docforge_{Path(uf.name).stem}_")
            result = _convert_one_file(uf, forge, output_dir)
            st.session_state.batch_results[uf.name] = result

            if result.get("error"):
                error_count += 1
            else:
                success_count += 1

            progress_bar.progress(
                (idx + 1) / total,
                text=f"Converted {idx + 1}/{total} files — ✅ {success_count} · ❌ {error_count}",
            )

        # Final status
        if error_count == 0:
            status_area.success(
                f"✅ All {total} file{'s' if total != 1 else ''} converted successfully "
                f"({success_count} succeeded, {error_count} failed)"
            )
        else:
            status_area.warning(
                f"⚠️ Conversion complete — {success_count} succeeded, {error_count} failed "
                f"out of {total} file{'s' if total != 1 else ''}"
            )

# ──────────────────────────────────────────────────────────────────────
# Results display
# ──────────────────────────────────────────────────────────────────────
if st.session_state.batch_results:
    results = st.session_state.batch_results
    total_files = len(results)
    success_files = {k: v for k, v in results.items() if not v.get("error")}
    error_files = {k: v for k, v in results.items() if v.get("error")}

    # ── Summary stats ──
    st.markdown("## 📊 Results")

    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{total_files}</div>'
            f'<div class="stat-label">Total Files</div></div>',
            unsafe_allow_html=True,
        )
    with stat_cols[1]:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{len(success_files)}</div>'
            f'<div class="stat-label">Succeeded</div></div>',
            unsafe_allow_html=True,
        )
    with stat_cols[2]:
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{len(error_files)}</div>'
            f'<div class="stat-label">Failed</div></div>',
            unsafe_allow_html=True,
        )
    with stat_cols[3]:
        total_md = sum(len(v.get("markdown", "")) for v in success_files.values())
        st.markdown(
            f'<div class="stat-card"><div class="stat-number">{_file_size_str(total_md)}</div>'
            f'<div class="stat-label">Total Markdown</div></div>',
            unsafe_allow_html=True,
        )

    # ── Download All as ZIP ──
    if success_files:
        st.markdown("---")
        zip_bytes = _build_zip(success_files)
        st.download_button(
            "📦 Download All as ZIP",
            data=zip_bytes,
            file_name="docforge_batch.zip",
            mime="application/zip",
            use_container_width=True,
        )

    # ── Per-file results ──
    st.markdown("### 📄 File Results")

    for filename, res in results.items():
        stem = Path(filename).stem
        ext = Path(filename).suffix.lower().replace(".", "")
        icon = {"pdf": "📕", "docx": "📘", "doc": "📘", "pptx": "📙", "ppt": "📙"}.get(ext, "🖼️")

        if res.get("error"):
            with st.expander(f"❌ {icon} {filename}", expanded=False):
                st.error(f"Conversion failed: {res['error']}")
            continue

        elapsed = res.get("elapsed", 0)
        stats = res.get("structured", {}).get("stats", {})
        n_images = len(res.get("images", []))

        # Build a short summary line
        stat_parts = [f"⏱ {elapsed:.1f}s"]
        if stats:
            for k, v in list(stats.items())[:3]:
                stat_parts.append(f"{k.replace('_', ' ').title()}: {v}")
        if n_images:
            stat_parts.append(f"🖼 {n_images} image{'s' if n_images != 1 else ''}")

        summary = " · ".join(stat_parts)

        with st.expander(f"✅ {icon} {filename}  —  {summary}", expanded=(total_files == 1)):
            # Per-file stats
            if stats:
                stat_cards = st.columns(len(stats) if stats else 1)
                for i, (label, value) in enumerate(stats.items()):
                    with stat_cards[i % len(stat_cards)]:
                        st.markdown(
                            f'<div class="stat-card">'
                            f'<div class="stat-number">{value}</div>'
                            f'<div class="stat-label">{label.replace("_", " ").title()}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # LLM info
            if use_llm and res.get("structured", {}).get("metadata", {}).get("use_llm"):
                st.caption("🤖 LLM enhanced")

            # Tabs: Markdown / JSON / Images / Download
            tab_md, tab_json, tab_images, tab_download = st.tabs(
                ["📝 Markdown", "📊 JSON", "🖼️ Images", "⬇️ Download"]
            )

            with tab_md:
                st.markdown("#### Markdown Output")
                st.markdown(
                    f'<div class="result-box">{_escape_html(res["markdown"])}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("📖 Rendered Preview"):
                    st.markdown(res["markdown"])

            with tab_json:
                st.markdown("#### Structured JSON")
                json_str = json.dumps(res["structured"], indent=2, ensure_ascii=False, default=str)
                st.code(json_str, language="json")

            with tab_images:
                st.markdown("#### Extracted Images")
                if res.get("images"):
                    img_cols = st.columns(min(len(res["images"]), 4))
                    for idx, img in enumerate(res["images"]):
                        col = img_cols[idx % len(img_cols)]
                        with col:
                            img_data = res.get("image_data", {}).get(img["filename"])
                            if img_data:
                                st.image(img_data, caption=img["filename"])
                            else:
                                st.text(f"{img['filename']} (not available)")
                else:
                    st.info("No images extracted from this document.")

            with tab_download:
                st.markdown("#### Download Files")
                md_bytes = res["markdown"].encode("utf-8")
                json_bytes = json.dumps(
                    res["structured"], indent=2, ensure_ascii=False, default=str
                ).encode("utf-8")

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "📝 Download Markdown",
                        data=md_bytes,
                        file_name=f"{stem}.md",
                        mime="text/markdown",
                        key=f"dl_md_{filename}",
                        use_container_width=True,
                    )
                with dl_col2:
                    st.download_button(
                        "📊 Download JSON",
                        data=json_bytes,
                        file_name=f"{stem}.json",
                        mime="application/json",
                        key=f"dl_json_{filename}",
                        use_container_width=True,
                    )

                if res.get("image_data"):
                    st.markdown("**Images:**")
                    for img_name, img_bytes in res["image_data"].items():
                        st.download_button(
                            f"🖼️ {img_name}",
                            data=img_bytes,
                            file_name=img_name,
                            mime="image/png",
                            key=f"dl_img_{filename}_{img_name}",
                        )

    # ── Combined Markdown preview ──
    if len(success_files) > 1:
        st.markdown("---")
        st.markdown("### 📑 Combined Output")

        combined_md = "\n\n---\n\n".join(
            f"# {Path(f).stem}\n\n{r['markdown']}"
            for f, r in success_files.items()
        )

        with st.expander("📖 Combined Markdown Preview", expanded=False):
            st.markdown(combined_md)

        combined_json = {
            "batch": True,
            "files": {
                Path(f).stem: r["structured"]
                for f, r in success_files.items()
            },
        }
        combined_json_bytes = json.dumps(
            combined_json, indent=2, ensure_ascii=False, default=str
        ).encode("utf-8")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "📝 Download Combined Markdown",
                data=combined_md.encode("utf-8"),
                file_name="docforge_combined.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "📊 Download Combined JSON",
                data=combined_json_bytes,
                file_name="docforge_combined.json",
                mime="application/json",
                use_container_width=True,
            )

else:
    # ── Empty state ──
    st.markdown(
        """
        <div style='text-align: center; padding: 3rem; color: #8b949e;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>📂</div>
            <div style='font-size: 1.2rem; margin-bottom: 0.5rem;'>
                Drag & drop multiple files above
            </div>
            <div style='font-size: 0.9rem;'>
                Supports PDF, DOCX, PPTX, and images<br>
                All files are processed in batch with a single click
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
