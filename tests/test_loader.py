from __future__ import annotations

from pathlib import Path

import pytest

from ragret.loader import is_ignored_corpus_filename, iter_indexable_files, iter_raw_corpus_files, relative_source_key


def test_iter_indexable_files_raises_on_nonexistent(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        iter_indexable_files(nonexistent)


def test_iter_indexable_files_single_file(tmp_path: Path) -> None:
    f = tmp_path / "test.md"
    f.write_text("hello")
    result = iter_indexable_files(f)
    assert result == [f.resolve()]


def test_iter_indexable_files_skips_non_indexable(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("a")
    (tmp_path / "data.csv").write_text("b")
    (tmp_path / "doc.md").write_text("c")
    result = iter_indexable_files(tmp_path)
    names = {p.name for p in result}
    assert names == {"notes.txt", "doc.md"}


def test_relative_source_key_within_dir(tmp_path: Path) -> None:
    f = tmp_path / "sub" / "doc.md"
    f.parent.mkdir(parents=True)
    f.write_text("test")
    assert relative_source_key(tmp_path, str(f)) == "sub/doc.md"


def test_relative_source_key_outside_dir() -> None:
    p = Path("/etc/passwd")
    result = relative_source_key(Path("/tmp"), str(p))
    assert "etc" in result or "passwd" in result


def test_relative_source_key_empty() -> None:
    result = relative_source_key(Path("/tmp/work"), "")
    assert result


def test_is_ignored_corpus_filename_word_lock() -> None:
    assert is_ignored_corpus_filename("~$report.docx") is True
    assert is_ignored_corpus_filename("report.docx") is False


def test_iter_raw_corpus_files_skips_word_lock_files(tmp_path: Path) -> None:
    (tmp_path / "real.docx").write_bytes(b"PK\x03\x04")
    (tmp_path / "~$real.docx").write_bytes(b"not-a-docx")
    names = {p.name for p in iter_raw_corpus_files(tmp_path)}
    assert names == {"real.docx"}
