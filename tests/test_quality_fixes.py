"""Unit tests for DocForge quality fixes (tables, patents, ligatures, code)."""

from docforge.table_utils import (
    reconstruct_single_column_table,
    score_table,
    pick_best_table,
    tables_are_duplicates,
    dedupe_table_list,
    normalize_table,
)
from docforge.text_cleanup import (
    fix_ligatures_and_ocr,
    looks_like_code,
    looks_like_drawing_label,
    collapse_spaced_text,
)
from docforge.artifact_remover import ArtifactRemover


class TestTables:
    def test_reconstruct_sbir_style_one_col(self):
        headers = ["Item  Year1  Year2  Total"]
        rows = [
            ["Personnel  100  200  300"],
            ["Equipment  50  50  100"],
            ["Total  150  250  400"],
        ]
        # Normalize to single-col lists as pdfplumber often returns
        headers = [headers[0]]
        rows = [[r[0]] for r in rows]
        nh, nr = reconstruct_single_column_table(headers, rows)
        assert len(nh) >= 3
        assert len(nr) == 3
        assert all(len(r) == len(nh) for r in nr)
        assert score_table(nh, nr) > score_table(headers, rows)

    def test_pick_best_prefers_multi_col(self):
        bad = (["all jammed"], [["a b c"], ["d e f"]])
        good = (["A", "B", "C"], [["1", "2", "3"], ["4", "5", "6"]])
        best = pick_best_table([bad, good])
        assert best is not None
        assert best[0] == ["A", "B", "C"]

    def test_dedupe_tables(self):
        t1 = {"headers": ["Pub", "Date"], "rows": [["US1", "2020"], ["US2", "2021"]]}
        t2 = {"headers": ["Pub", "Date"], "rows": [["US1", "2020"], ["US2", "2021"]]}
        t3 = {"headers": ["Name"], "rows": [["Alice"]]}
        out = dedupe_table_list([t1, t2, t3])
        assert len(out) == 2

    def test_normalize_drops_empty(self):
        assert normalize_table([[None, None], ["", ""]]) is None


class TestTextCleanup:
    def test_slent_and_electried(self):
        out = fix_ligatures_and_ocr("SLENT mode in electried aircraft")
        assert "SILENT" in out
        assert "electrified" in out

    def test_unicode_ligatures(self):
        out = fix_ligatures_and_ocr("the \ufb01rst \ufb01nding")
        assert "first" in out
        assert "finding" in out or "ﬁnding" not in out

    def test_spaced_form_text(self):
        out = collapse_spaced_text("P r o p o s a l   S u m m a r y")
        assert "Proposal" in out
        assert "Summary" in out

    def test_drawing_label_not_code(self):
        label = "OUTPUT BANDPASS FILTER"
        assert looks_like_drawing_label(label)
        assert not looks_like_code(label, is_mono_font=True)

    def test_real_code_detected(self):
        code = "def foo(x):\n    return x + 1\n"
        assert looks_like_code(code, is_mono_font=True)

    def test_patent_ocr_words(self):
        out = fix_ligatures_and_ocr("The inventlon includes a circnit and amplifer")
        assert "invention" in out
        assert "circuit" in out
        assert "amplifier" in out


class TestArtifacts:
    def test_google_patents_chrome(self):
        md = "\n".join(
            [
                "Download PDF",
                "Find Prior Art",
                "Google Patents",
                "Useful claim text about a circuit",
                "Download PDF",
                "Find Prior Art",
                "https://patents.google.com/patent/US123",
                "",
                "| Pub | Date |",
                "| --- | --- |",
                "| US1 | 2020 |",
                "",
                "| Pub | Date |",
                "| --- | --- |",
                "| US1 | 2020 |",
            ]
        )
        cleaned = ArtifactRemover().remove(md)
        assert "Download PDF" not in cleaned
        assert "Find Prior Art" not in cleaned
        assert "patents.google.com" not in cleaned
        assert "Useful claim text" in cleaned
        # Table appears once
        assert cleaned.count("| Pub | Date |") == 1
