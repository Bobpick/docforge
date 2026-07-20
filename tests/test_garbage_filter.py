"""Garbage filter: reject shredded tables, clean bold/wraps."""

from docforge.garbage_filter import (
    is_garbage_table,
    filter_tables,
    strip_garbage_markdown_tables,
    defragment_bold,
    reflow_hard_wraps,
    clean_extracted_markdown,
)
from docforge.table_utils import score_table


def test_rejects_letter_soup_title():
    headers = ["A", "RMY", "OP", "E", "N", "S", "O", "LI", "CI", "TA", "TI", "ON"]
    rows = [["W", "9", "128", "Z", "-", "25", "-", "S", "-", "A", "00", "2"]]
    assert is_garbage_table(headers, rows)


def test_rejects_prose_shred_wide():
    headers = [""] * 12
    rows = [
        ["3–20 G", "Hz Wideband Tightl", "y-", "Co", "up", "led", "D", "ual-", "P", "ola", "rized", ""],
        ["", "Vivaldi Ant", "en", "na", "Ar", "ray", "", "", "", "", "", ""],
    ]
    assert is_garbage_table(headers, rows)


def test_keeps_sane_spec_table():
    headers = ["Parameter", "Min", "Typ", "Max", "Unit"]
    rows = [
        ["Frequency Range", "20", "", "6000", "MHz"],
        ["Gain (S21)", "", "19.7", "", "dB"],
        ["Noise Figure", "", "3.0", "", "dB"],
    ]
    assert not is_garbage_table(headers, rows)
    assert score_table(headers, rows) >= 8.0


def test_filter_tables_drops_garbage():
    good = {
        "headers": ["A", "B"],
        "rows": [["1", "2"], ["3", "4"]],
    }
    bad = {
        "headers": list("ABCDEFGHIJKL"),
        "rows": [list("abcdefghijkl")],
    }
    out = filter_tables([good, bad])
    assert len(out) == 1
    assert out[0]["headers"] == ["A", "B"]


def test_strip_garbage_md_tables():
    md = """# Title

Hello world paragraph.

| A | RMY | OP | E | N | SOL |
| --- | --- | --- | --- | --- | --- |
| x | y | z | q | r | s |

| Param | Typ | Unit |
| --- | --- | --- |
| Gain | 20 | dB |

Done.
"""
    cleaned = strip_garbage_markdown_tables(md)
    assert "Hello world" in cleaned
    assert "Gain" in cleaned
    assert "RMY" not in cleaned


def test_defragment_bold():
    s = "**Student** **Member,** **IEEE**"
    out = defragment_bold(s)
    assert out.count("**") == 2
    assert "Student" in out and "IEEE" in out


def test_reflow_hard_wraps():
    text = (
        "This is a long enough prose line that looks like a PDF wrap\n"
        "continuing onto the next line without a blank between them.\n"
        "\n"
        "# Heading stays\n"
    )
    out = reflow_hard_wraps(text)
    assert "wrap continuing" in out or "wrapcontinuing" in out.replace(" ", "")
    assert "# Heading stays" in out


def test_clean_pipeline_smoke():
    md = "**direction** **of** **arrival** (DoA)\n\n| A | B | C | D | E | F | G | H |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n| a | b | c | d | e | f | g | h |\n"
    out = clean_extracted_markdown(md)
    assert "direction" in out
    # shredded wide table should be gone or rare
    assert out.count("|") < md.count("|")
