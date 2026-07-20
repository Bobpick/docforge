# Table extraction test fixture: ADL5542

## Why this document

From the **tables-on** batch (`~/Downloads/docforge_combined.md` before no-tables).

Analog Devices **ADL5542** RF/IF gain-block datasheet is the best single local test:

| Has | Why it matters |
|-----|----------------|
| Real ruled **Table 1** specs (Parameter / Min / Typ / Max / Unit) | Success = recover these |
| Plot/figure axes misread as 20–40 column tables | Success = **reject** these |
| FEATURES as multi-col layout soup | Success = keep as **prose**, not tables |
| Manageable size (~13 pages, not 386-table monster) | Fast iterate |

Worse alternatives (not recommended as sole fixture):
- `Design and Modeling of an UWB…` — pure shred chaos (386 tables), slow, little ground truth
- `2511.16472v2` — academic multi-col; few clean target grids
- Army AOS — form/title soup, weak true data tables

## Files

| File | Role |
|------|------|
| `ADL5542.pdf` | Source PDF (copy from local RAG_data) |
| `ADL5542_BAD_tables_on.md` | DocForge output **with** tables on (baseline failure) |
| `ADL5542_BAD_tables_on.meta.json` | Stats + success criteria |

## Local test

```bash
cd ~/projects/forks/docforge
source .venv/bin/activate
# optional for Camelot lattice (Ghostscript already needed system-wide):
# pip install camelot-py opencv-python-headless

# Default (tables off — fast prose)
python cli.py convert fixtures/table_test/ADL5542.pdf -o /tmp/adl5542_notables

# Quality tables ON (uses docforge/table_extractor.py)
python -c "
from docforge import DocForge
r = DocForge(extract_tables=True, extract_images=False).convert(
    'fixtures/table_test/ADL5542.pdf', output_dir='/tmp/adl5542_tables'
)
print(r.structured.get('stats'))
tabs = [s for s in r.structured.get('sections',[]) if s.get('type')=='table']
print('tables', len(tabs))
for t in tabs:
    print(' -', t.get('source'), 'cols', len(t.get('headers') or []), 'rows', len(t.get('rows') or []), 'score', t.get('score'))
"
```

## Pass bar

1. At least one clean spec table with Gain / Frequency / return-loss-style numbers  
2. Table count **≪ 75** (old bad run) — ideally under ~10  
3. No letter-split title rows (`20 | MH | z t | o | 6 GHz`)  
4. GENERAL DESCRIPTION still normal paragraphs  
