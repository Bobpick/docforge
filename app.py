#!/usr/bin/env python3
"""DocForge Web App — Interactive document conversion powered by Streamlit.

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path

# Ensure the docforge package is importable
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from docforge.converter import DocForge


# ──────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocForge — Document Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished look
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
</style>
""", unsafe_allow_html=True)


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
    st.markdown("### 🤖 LLM Enhancement (Gemini)")

    use_llm = st.checkbox(
        "Enable LLM cleanup",
        value=False,
        help="Use Google Gemini to clean up tables, fix OCR errors, and improve formatting.",
    )

    llm_api_key = st.text_input(
        "Google AI API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Your Google AI API key for Gemini. Also set via GOOGLE_API_KEY env var.",
    )

    llm_model = st.selectbox(
        "Gemini Model",
        ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="Select which Gemini model to use for enhancement.",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#666;'>"
        "DocForge v1.0 — PDF · DOCX · PPTX · Images → Markdown & JSON"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📄 DocForge</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Convert documents to clean Markdown & JSON — '
    'with table, math, and code support.</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "doc", "pptx", "ppt", "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp", "gif"],
    help="Drag and drop or click to browse. Supports PDF, Word, PowerPoint, and image files.",
)

if uploaded_file is not None:
    # Show file info
    file_details = {
        "Filename": uploaded_file.name,
        "Size": f"{uploaded_file.size / 1024:.1f} KB",
        "Type": uploaded_file.type or "unknown",
    }
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"📄 **{file_details['Filename']}**")
    with col2:
        st.info(f"💾 **{file_details['Size']}**")
    with col3:
        st.info(f"🏷️ **{file_details['Type']}**")

    # Convert button
    if st.button("🔄 Convert", type="primary", use_container_width=True):
        # Write uploaded file to a temp location
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        output_dir = tempfile.mkdtemp()

        try:
            # Build the converter with current settings
            forge = DocForge(
                use_llm=use_llm,
                llm_api_key=llm_api_key if llm_api_key else None,
                llm_model=llm_model,
                remove_artifacts=remove_artifacts,
                extract_images=extract_images,
            )

            with st.spinner("Converting document…"):
                start = time.time()
                result = forge.convert(tmp_path, output_dir=output_dir)
                elapsed = time.time() - start

            st.success(f"✅ Conversion complete in {elapsed:.1f}s")

            # ── Stats ──
            stats = result.structured.get("stats", {})
            stat_cols = st.columns(len(stats) if stats else 1)
            for i, (label, value) in enumerate(stats.items()):
                with stat_cols[i % len(stat_cols)]:
                    st.markdown(
                        f'<div class="stat-card">'
                        f'<div class="stat-number">{value}</div>'
                        f'<div class="stat-label">{label.replace("_", " ").title()}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Output tabs ──
            tab_md, tab_json, tab_images, tab_download = st.tabs(
                ["📝 Markdown", "📊 JSON", "🖼️ Images", "⬇️ Download"]
            )

            with tab_md:
                st.markdown("### Markdown Output")
                st.markdown(
                    f'<div class="result-box">{_escape_html(result.markdown)}</div>',
                    unsafe_allow_html=True,
                    )
                # Also render the markdown
                with st.expander("📖 Rendered Preview"):
                    st.markdown(result.markdown)

            with tab_json:
                st.markdown("### Structured JSON")
                json_str = json.dumps(result.structured, indent=2, ensure_ascii=False, default=str)
                st.code(json_str, language="json")

            with tab_images:
                st.markdown("### Extracted Images")
                if result.images:
                    img_cols = st.columns(min(len(result.images), 4))
                    for idx, img in enumerate(result.images):
                        col = img_cols[idx % len(img_cols)]
                        with col:
                            img_path = Path(output_dir) / "images" / img["filename"]
                            if img_path.exists():
                                st.image(str(img_path), caption=img["filename"])
                            else:
                                st.text(f"{img['filename']} (not saved)")
                else:
                    st.info("No images extracted from this document.")

            with tab_download:
                st.markdown("### Download Files")
                md_path = Path(output_dir) / "output.md"
                json_path = Path(output_dir) / "output.json"

                if md_path.exists():
                    st.download_button(
                        "📝 Download Markdown",
                        data=md_path.read_bytes(),
                        file_name=f"{Path(uploaded_file.name).stem}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                if json_path.exists():
                    st.download_button(
                        "📊 Download JSON",
                        data=json_path.read_bytes(),
                        file_name=f"{Path(uploaded_file.name).stem}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                # Individual images
                if result.images:
                    images_dir = Path(output_dir) / "images"
                    for img in result.images:
                        img_path = images_dir / img["filename"]
                        if img_path.exists():
                            st.download_button(
                                f"🖼️ {img['filename']}",
                                data=img_path.read_bytes(),
                                file_name=img["filename"],
                                mime="image/png",
                            )

        except Exception as e:
            st.error(f"❌ Conversion failed: {e}")
            with st.expander("Show traceback"):
                st.exception(e)

        finally:
            # Cleanup temp files
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _escape_html(text: str) -> str:
    """Escape HTML special chars for display in a div."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
