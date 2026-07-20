# DocForge PDF Quality — Before vs After

## What was wrong (from your 5-6 PDF batch output)

| Problem | Example Before | Example After | Status |
|---|---|---|---|
| **Ligatures dropped** | `ight duration in electried aircraft` | `flight duration in electrified aircraft` | ✅ Fixed |
| **Spaced form text** | `P r o p o s a l   S u m m a r y` | `Proposal Summary` | ✅ Fixed |
| **Spaced multi-word** | `C o s t   B r e a k d o w n` | `Cost Breakdown` | ✅ Fixed |
| **Source filenames** | `tmpr3u_472g.pdf` | Your original filename | ✅ Fixed |
| **Patent drawing noise** | `` ``` Z \| ``` `` | Filtered out | ✅ Fixed |
| **Web UI chrome** | `Download PDF`, `Find Prior Art`, `Similar` | Removed | ✅ Fixed |
| **Page counters** | `Sheet 1 of 3`, `1 of 5` | Removed | ✅ Fixed |
| **URL footers** | `https://patents.google.com/...` | Removed | ✅ Fixed |

## Remaining issues — now improved without requiring LLM

| Problem | Example | What we do now | Residual risk |
|---|---|---|---|
| **Table extraction** | SBIR budget tables mangled into 1-cell rows | Multi-strategy pdfplumber (lines/text mixes) + PyMuPDF `find_tables` + score & pick best + reconstruct multi-col from space-aligned 1-cell rows | Complex merged cells / multi-line form fields may still need `--llm` |
| **Google Patents print-to-PDF** | Repeating headers, citation tables x5 | Stronger chrome patterns, patent-print detection, aggressive repeated-line strip, identical table dedupe | Novel chrome strings still need a one-line pattern add |
| **Scanned patent drawings** | `OUTPUT BANDPASS FILTER` as code | Mono font alone is no longer enough; drawing-label heuristic keeps ALL-CAPS callouts as paragraphs | Real code in mono ALL-CAPS (rare) may be demoted |
| **Partial ligature / OCR** | `electried` ✅, `SLENT` stayed | Expanded high-confidence OCR map (`SLENT`→`SILENT`, patent vocab, etc.) + Unicode ligatures | Open-vocabulary OCR still needs `--llm` or a spellchecker |

## Implementation map

| Area | Module |
|---|---|
| Table strategies / 1-col rebuild / table dedupe | `docforge/table_utils.py` |
| Ligatures, OCR words, code vs drawing | `docforge/text_cleanup.py` |
| PDF pipeline wiring | `docforge/pdf_handler.py` |
| Patent print chrome | `docforge/artifact_remover.py` |
| Unit tests | `tests/test_quality_fixes.py` |

## New: Ollama Service Management

### Streamlit App
When you select Ollama as provider, the sidebar now shows:
- **📋 Status** expander: running/stopped, PID, models, GPU/VRAM
- **▶️ Start** button: starts Ollama, optionally pre-loads a model
- **⏹️ Stop** button: stops Ollama, frees GPU/RAM
- **🔄 Restart** button: stop + start with model reload

### CLI
```bash
# Check if Ollama is running
docforge ollama status

# Start Ollama and pre-load cogito:14b
docforge ollama start --model cogito:14b

# Stop Ollama (frees GPU memory)
docforge ollama stop

# Restart (recover from stuck state)
docforge ollama restart --model cogito:14b
```

### Recommended batch workflow

1. **Stop Ollama first** (frees GPU for faster non-LLM conversion):
   ```
   docforge ollama stop
   ```

2. **Run batch conversion without LLM** (fast, all files at once):
   ```
   docforge batch documents/ -o converted/
   ```

3. **Restart Ollama** for LLM cleanup on remaining problem files:
   ```
   docforge ollama start --model cogito:14b
   docforge convert messy.pdf --llm
   ```

## Bottom line

Most of the previously “needs LLM” list is now handled deterministically:
tables, patent chrome, drawing labels, and common OCR/ligature words.

Use `--llm --provider ollama --model cogito:14b` only for the long tail
(weird form layouts, open-vocabulary OCR, heavy noise).
