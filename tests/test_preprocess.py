from __future__ import annotations

from pathlib import Path

import pytest

from ragret.preprocess import preprocess_file


def test_preprocess_txt_file(tmp_path: Path) -> None:
    p = tmp_path / "note.txt"
    p.write_text("hello\nworld", encoding="utf-8")
    assert preprocess_file(p) == "hello\nworld"


def test_preprocess_txt_file_gb18030(tmp_path: Path) -> None:
    p = tmp_path / "note-gbk.txt"
    p.write_bytes("中文说明".encode("gb18030"))
    assert "中文说明" in preprocess_file(p)


def test_preprocess_xlsx_openpyxl_style_error_uses_xml_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "book.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "数据"
    ws["A1"] = "单元格内容"
    wb.save(p)
    wb.close()

    def boom(*_args, **_kwargs):
        raise TypeError("expected Fill")

    monkeypatch.setattr(openpyxl, "load_workbook", boom)
    text = preprocess_file(p)
    assert "单元格内容" in text
    assert "sheet 数据" in text or "sheet" in text


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
