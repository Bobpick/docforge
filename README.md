# 📄 DocForge

**Convert any document into clean Markdown and structured JSON — built for LLM pipelines.**

---

## Why DocForge?

If you're feeding documents into an LLM — for RAG, summarization, code analysis, or any AI workflow — you need the content in a format the model can actually use. Raw PDFs and Word files don't work. Copy-pasting loses structure. Manual conversion is slow and error-prone.

DocForge solves this by converting PDFs, Word docs, PowerPoints, and images into **clean Markdown** and **structured JSON** in one step. Tables stay as tables. Math stays as LaTeX. Code stays as code. Images get extracted and saved. Headers, footers, and page numbers get stripped automatically.

**The result is output that's ready to paste into a prompt, index in a vector database, or process downstream — no manual cleanup required.**

---

## Use Cases

### 1. RAG Pipeline Preparation
You have a knowledge base of PDFs, Word docs, and slide decks. Before embedding them in a vector store, you need clean text. DocForge converts them all into Markdown with proper headings, tables, and structure — far better for chunking and retrieval than raw text extraction.

```python
from docforge import DocForge

forge = DocForge(remove_artifacts=True, extract_images=True)
result = forge.convert("knowledge_base/q4_report.pdf")

# Feed the markdown directly into your chunker/embedder
chunks = split_by_headings(result.markdown)
embed_and_store(chunks)
```

### 2. Feeding Documents to an LLM Agent
Your coding agent needs to read a Word specification or a PDF API reference. Instead of dumping raw text (full of formatting artifacts), convert it first:

```bash
slim run python cli.py convert api_spec.docx -o output/spec
# Output is clean Markdown — tables, code blocks, and all
```

### 3. Batch Processing a Document Corpus
You have hundreds of documents that need conversion. Process them all at once:

```bash
python cli.py batch documents/ -o converted/ --recursive
# Each document gets its own output folder with output.md, output.json, and images/
```

### 4. OCR for Scanned Documents
Scanned PDFs or photos of documents? DocForge uses Tesseract OCR under the hood and adds structure detection:

```bash
python cli.py convert scanned_invoice.png -o output/invoice
```

### 5. LLM-Enhanced Cleanup for Messy Documents
Bad OCR, garbled tables, or formatting artifacts? Enable LLM-powered enhancement. By default, DocForge uses **Ollama** with `cogito:14b` — it runs locally, is free, and keeps your data private. No API key needed.

```bash
# Using Ollama (default, local, free)
ollama pull cogito:14b
python cli.py convert messy_scan.pdf --llm

# Or use Gemini (cloud)
python cli.py convert messy_scan.pdf --llm --provider gemini --api-key YOUR_KEY
```

### 6. Interactive Web App for Quick Conversions
Upload a file, convert it, and download the results — no code needed:

```bash
streamlit run app.py
```

---

## How It Works

DocForge processes documents through a multi-stage pipeline:

```
Input file → Format-specific handler → Artifact removal → (Optional LLM enhancement) → Output
```

1. **Format detection** — Identifies the file type and routes to the correct handler.
2. **Structured extraction** — Each handler (PDF, DOCX, PPTX, Image) extracts content while preserving structure:
   - Headings are detected by font size (PDF) or style (DOCX/PPTX)
   - Tables are extracted via pdfplumber (PDF) or native APIs (DOCX/PPTX)
   - Code blocks are identified by monospace font analysis
   - Math expressions are detected by LaTeX pattern matching
   - Bold, italic, and other formatting is preserved from source styles
3. **Image extraction** — Embedded images are pulled from the document (via ZIP access for DOCX/PPTX, PyMuPDF for PDF) and saved to an `images/` directory.
4. **Artifact removal** — Repeated headers/footers, page numbers, and watermark text are detected and stripped.
5. **LLM enhancement** (optional) — Gemini is called on sections that need cleanup: fixing garbled tables, improving math formatting, and detecting code blocks that font analysis missed.
6. **Output** — Clean Markdown and structured JSON are written alongside extracted images.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-format input** | PDF, DOCX, PPTX, PNG, JPG, TIFF, BMP, WebP |
| **Clean Markdown** | Headings, bold/italic, lists, blockquotes, code fences |
| **Structured JSON** | Sections, metadata, stats, image references |
| **Tables** | Extracted accurately via pdfplumber (PDF) or native APIs (DOCX/PPTX) |
| **Math equations** | Auto-detected LaTeX expressions → `$...$` / `$$...$$` |
| **Code blocks** | Detected by font analysis → fenced code blocks |
| **Image extraction** | Embedded images saved to `images/` subdirectory |
| **Artifact removal** | Strips headers, footers, page numbers, watermarks |
| **LLM enhancement** | Ollama (local), Gemini (cloud), or OpenAI-compatible — cleanup tables, math, code |
| **Web UI** | Upload & convert via Streamlit |
| **CLI** | Batch-process entire folders from the terminal |

---

## 🚀 Quick Start

### Install

```bash
cd docforge
pip install -r requirements.txt

# For OCR (image support), also install Tesseract:
# macOS:   brew install tesseract
# Ubuntu:  sudo apt install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

### Web App

```bash
streamlit run app.py
```

Upload a file, tweak settings in the sidebar, click **Convert**, then preview or download the results.

### CLI — Single File

```bash
python cli.py convert report.pdf -o output/report
```

### CLI — Batch Folder

```bash
python cli.py batch documents/ -o converted/ --recursive
```

### CLI — With LLM Enhancement (Ollama, local and free)

```bash
# Default: Ollama with cogito:14b (runs locally, no API key needed)
ollama pull cogito:14b                 # First time: download the model
python cli.py convert messy_scan.pdf --llm
python cli.py convert messy.pdf --llm --model llama3:8b    # Different Ollama model

# With Google Gemini (cloud, requires API key)
export GOOGLE_API_KEY="your-key-here"
python cli.py convert messy.pdf --llm --provider gemini

# With any OpenAI-compatible API (LM Studio, vLLM, llama.cpp, etc.)
python cli.py convert messy.pdf --llm --provider openai-compat --model local-model
```

### Python API

```python
from docforge import DocForge

forge = DocForge(remove_artifacts=True, extract_images=True)
result = forge.convert("report.pdf", output_dir="output/report")

print(result.markdown)           # → Markdown string
print(result.structured)         # → Dict with sections, stats, metadata
print(len(result.images))        # → Number of extracted images

# Save to disk
result.save("output/report")
# Creates: output/report/output.md
#          output/report/output.json
#          output/report/images/*.png
```

---

## 🏗️ Architecture

```
docforge/
├── docforge/
│   ├── __init__.py           # Package exports
│   ├── converter.py          # DocForge engine + ConversionResult
│   ├── pdf_handler.py        # PDF → MD/JSON (PyMuPDF + pdfplumber)
│   ├── docx_handler.py       # DOCX → MD/JSON (python-docx)
│   ├── pptx_handler.py       # PPTX → MD/JSON (python-pptx)
│   ├── image_handler.py      # Images → MD/JSON (pytesseract OCR)
│   ├── artifact_remover.py   # Header/footer/page-number removal
│   ├── llm_enhancer.py       # LLM enhancement (Ollama / Gemini / OpenAI-compat)
│   └── utils.py              # Shared utilities (tables, math, hashing)
├── app.py                    # Streamlit web application
├── cli.py                    # Click-based CLI tool
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

### Conversion Options

| Option | Default | Description |
|---|---|---|
| `remove_artifacts` | `True` | Strip headers, footers, page numbers |
| `extract_images` | `True` | Extract and save embedded images |
| `use_llm` | `False` | Enable LLM enhancement |
| `llm_provider` | `"ollama"` | LLM provider: `"ollama"`, `"gemini"`, `"openai-compat"` |
| `llm_model` | `"cogito:14b"` | Model name (varies by provider) |
| `llm_api_key` | env `GOOGLE_API_KEY` | API key (Gemini only) |
| `llm_host` | `http://localhost:11434` | Ollama host URL |

### CLI Flags

| Flag | Description |
|---|---|
| `--no-artifacts` | Keep document artifacts |
| `--no-images` | Skip image extraction |
| `--llm` | Enable LLM enhancement |
| `--provider PROVIDER` | LLM provider: ollama, gemini, openai-compat |
| `--model MODEL` | LLM model name (default: cogito:14b for Ollama) |
| `--api-key KEY` | API key for cloud providers (Gemini) |
| `--ollama-host URL` | Ollama host URL |
| `-r / --recursive` | Process subdirectories (batch mode) |
| `-q / --quiet` | Suppress banner and summary |

---

## 📋 Output Format

### Markdown (`output.md`)

```markdown
# Document Title

## Section 1

Some paragraph with **bold** and *italic* text.

| Column 1 | Column 2 | Column 3 |
| --- | --- | --- |
| Data 1   | Data 2   | Data 3   |

$$\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n$$

```python
def hello():
    print("Hello, world!")
```

![Image](images/page001_img001.png)
```

### JSON (`output.json`)

```json
{
  "title": "Document Title",
  "metadata": {
    "source_file": "report.pdf",
    "conversion_date": "2025-01-15T10:30:00",
    "page_count": 5,
    "use_llm": false,
    "artifacts_removed": true
  },
  "sections": [
    { "type": "heading", "level": 1, "content": "Document Title" },
    { "type": "paragraph", "content": "Some paragraph..." },
    { "type": "table", "headers": ["Column 1", "Column 2"], "rows": [["Data 1", "Data 2"]] },
    { "type": "math", "content": "\\sum_{i=1}^{n} x_i" },
    { "type": "code", "content": "def hello(): ..." },
    { "type": "image", "path": "images/page001_img001.png" }
  ],
  "stats": {
    "total_sections": 42,
    "headings": 8,
    "paragraphs": 25,
    "tables": 3,
    "code_blocks": 2,
    "math_blocks": 4,
    "images": 5
  }
}
```

---

## 🤖 LLM Enhancement

DocForge supports **three LLM providers** for intelligent cleanup:

| Provider | How it works | Cost | Privacy |
|---|---|---|---|
| **Ollama** (default) | Runs locally on your machine | Free | ✅ Fully private |
| **Google Gemini** | Cloud API | Pay per token | ❌ Data sent to Google |
| **OpenAI-compatible** | Works with LM Studio, vLLM, llama.cpp server | Varies | Depends on deployment |

### Setup

**Ollama (recommended):**
```bash
# Install Ollama: https://ollama.com
ollama pull cogito:14b       # Download the model
ollama serve                  # Start the server (runs on localhost:11434)

# Now use DocForge with --llm
python cli.py convert document.pdf --llm
```

**Gemini:**
```bash
export GOOGLE_API_KEY="your-key"
python cli.py convert document.pdf --llm --provider gemini
```

**OpenAI-compatible (LM Studio, vLLM, etc.):**
```bash
python cli.py convert document.pdf --llm --provider openai-compat --model local-model
```

### What the LLM does

When `use_llm=True`, DocForge sends the converted Markdown to the LLM
for a multi-pass cleanup:

1. **Table restructuring** — fixes misaligned columns, merged cells, OCR garble
2. **Math formatting** — wraps equations in proper LaTeX delimiters
3. **Code detection** — identifies and fences code blocks with language tags
4. **General cleanup** — fixes OCR artifacts, formatting inconsistencies

The LLM is called only on sections that need enhancement, and gracefully
degrades if the provider is unavailable (falls back to the rule-based conversion).

### Environment Variables

| Variable | Description |
|---|---|
| `DOCFORGE_LLM_PROVIDER` | Default provider: ollama, gemini, openai-compat |
| `DOCFORGE_OLLAMA_MODEL` | Default Ollama model (default: cogito:14b) |
| `OLLAMA_HOST` | Ollama server URL (default: http://localhost:11434) |
| `GOOGLE_API_KEY` | Gemini API key |

---

## 📄 License

MIT License — use freely in personal and commercial projects.
