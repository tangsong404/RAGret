from __future__ import annotations

from pathlib import Path

import pytest

from ragret.preprocess import preprocess_file


def test_preprocess_txt_file(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello\nworld", encoding="utf-8")
    assert preprocess_file(p) == "hello\nworld"


def test_preprocess_pdf_single_stream(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    pdf_path = tmp_path / "two.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with pdf_path.open("wb") as f:
        writer.write(f)
    text = preprocess_file(pdf_path)
    assert "--- page 1 ---" in text
    assert "--- page 2 ---" in text
