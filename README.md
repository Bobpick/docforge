# 📄 DocForge

**Powerful document-to-Markdown & JSON conversion tool.**

Convert PDFs, Word docs, PowerPoints, and images into clean, structured
Markdown and JSON — with first-class support for tables, math equations,
code blocks, and embedded images.

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
| **LLM enhancement** | Optional Gemini-powered cleanup for tricky tables & OCR fixes |
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

Upload a file, tweak settings in the sidebar, click **Convert**, then preview
or download the results.

### CLI — Single File

```bash
python cli.py convert report.pdf -o output/report
```

### CLI — Batch Folder

```bash
python cli.py batch documents/ -o converted/ --recursive
```

### CLI — With LLM Enhancement

```bash
export GOOGLE_API_KEY="your-key-here"
python cli.py convert messy_scan.pdf --llm
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
│   ├── llm_enhancer.py       # Gemini-based table & formatting cleanup
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
| `use_llm` | `False` | Enable Gemini LLM enhancement |
| `llm_api_key` | env `GOOGLE_API_KEY` | Google AI API key |
| `llm_model` | `gemini-2.0-flash` | Gemini model for enhancement |

### CLI Flags

| Flag | Description |
|---|---|
| `--no-artifacts` | Keep document artifacts |
| `--no-images` | Skip image extraction |
| `--llm` | Enable LLM enhancement |
| `--api-key KEY` | Pass API key directly |
| `--model MODEL` | Choose Gemini model |
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

When `use_llm=True`, DocForge sends the converted Markdown to Google Gemini
for a multi-pass cleanup:

1. **Table restructuring** — fixes misaligned columns, merged cells, OCR garble
2. **Math formatting** — wraps equations in proper LaTeX delimiters
3. **Code detection** — identifies and fences code blocks with language tags
4. **General cleanup** — fixes OCR artifacts, formatting inconsistencies

The LLM is called only on sections that need enhancement, and gracefully
degrades if the API is unavailable.

---

## 📄 License

MIT License — use freely in personal and commercial projects.
