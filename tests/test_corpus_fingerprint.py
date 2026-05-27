from __future__ import annotations

from pathlib import Path

import pytest

from ragret.loader import corpus_fingerprint_map


def test_corpus_fingerprint_includes_office_and_images(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.pdf").write_bytes(b"pdf")
    (tmp_path / "docs" / "b.docx").write_bytes(b"docx")
    (tmp_path / "sheet.xlsx").write_bytes(b"xlsx")
    (tmp_path / "pic.png").write_bytes(b"png")
    (tmp_path / "notes.txt").write_bytes(b"txt")
    fp = corpus_fingerprint_map(tmp_path)
    assert set(fp.keys()) == {"docs/a.pdf", "docs/b.docx", "sheet.xlsx", "pic.png", "notes.txt"}
    assert all(len(v) == 64 for v in fp.values())


def test_corpus_fingerprint_excludes_derived_store_dirs(tmp_path: Path) -> None:
    (tmp_path / "kb_parents" / "x").mkdir(parents=True)
    (tmp_path / "kb_parents" / "x" / "fake.pdf.txt").write_text("derived")
    (tmp_path / "real.pdf").write_bytes(b"pdf")
    fp = corpus_fingerprint_map(tmp_path)
    assert list(fp.keys()) == ["real.pdf"]
