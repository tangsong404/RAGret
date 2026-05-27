from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from ragret.indexer import (
    build_index,
    connect,
    get_meta,
    index_workdir,
    init_schema,
    try_incremental_update_workdir,
)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / "test.sqlite"
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    init_schema(c)
    yield c
    c.close()


class MockEmbedModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * 4 for _ in texts]


def test_build_index_creates_chunks(conn: sqlite3.Connection, tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("hello world\n\nsecond paragraph\n\nthird one")
    n = build_index(conn, tmp_path, MockEmbedModel())
    assert n > 0
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == n


def test_build_index_sets_meta(conn: sqlite3.Connection, tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("some content")
    build_index(conn, tmp_path, MockEmbedModel())
    assert get_meta(conn, "embedding_model") is not None
    assert get_meta(conn, "embed_dim") == "4"


def test_build_index_writes_parent_and_line_metadata(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    import json

    parents = tmp_path / "parents"
    (tmp_path / "doc.md").write_text("alpha\nbeta\n" * 400)
    build_index(conn, tmp_path, MockEmbedModel(), parents_dir=parents)
    assert (parents / "doc.md.txt").is_file()
    row = conn.execute("SELECT metadata_json FROM chunks LIMIT 1").fetchone()
    meta = json.loads(row[0])
    assert "line_start" in meta
    assert "line_end" in meta


def test_build_index_stores_corpus_fingerprints(conn: sqlite3.Connection, tmp_path: Path) -> None:
    import json

    (tmp_path / "notes.txt").write_text("some content")
    (tmp_path / "diagram.png").write_bytes(b"png")
    build_index(conn, tmp_path, MockEmbedModel())
    raw = get_meta(conn, "corpus_fingerprints")
    assert raw is not None
    fp = json.loads(raw)
    assert set(fp.keys()) == {"notes.txt", "diagram.png"}


def test_try_incremental_no_file_changes(tmp_path: Path) -> None:
    work = tmp_path / "corpus"
    work.mkdir()
    (work / "notes.txt").write_text("some content")
    db = tmp_path / "test.sqlite"
    conn = connect(db)
    build_index(conn, work, MockEmbedModel())
    conn.close()

    assert try_incremental_update_workdir(work, db) is True
    conn2 = connect(db)
    assert get_meta(conn2, "indexed_work_dir") == str(work.resolve())
    conn2.close()


def test_try_incremental_missing_db(tmp_path: Path) -> None:
    work = tmp_path / "corpus"
    work.mkdir()
    (work / "notes.txt").write_text("x")
    assert try_incremental_update_workdir(work, tmp_path / "missing.sqlite") is False


def test_index_workdir_full_build(tmp_path: Path) -> None:
    work = tmp_path / "corpus"
    work.mkdir()
    (work / "doc.md").write_text("hello world with enough text for chunks")
    db = tmp_path / "idx.sqlite"
    with patch("ragret.indexer.make_embed_model", return_value=MockEmbedModel()):
        with patch("ragret.indexer.resolve_device", return_value="cuda:0"):
            n = index_workdir(work, db)
    assert n > 0
    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == n
    conn.close()
