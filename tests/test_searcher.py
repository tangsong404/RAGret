from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ragret.cache import IndexCache, ModelCache
from ragret.indexer import init_schema, set_meta
from ragret.searcher import search_db


def _make_index(db_path: Path, texts: list[str], dim: int = 4) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    set_meta(conn, "embedding_model", "maidalun1020/bce-embedding-base_v1")
    set_meta(conn, "embed_dim", str(dim))
    for i, t in enumerate(texts):
        vec = np.full(dim, 0.5 if i == 0 else 0.1, dtype=np.float32)
        conn.execute(
            "INSERT INTO chunks(source, chunk_index, content, metadata_json, embedding) VALUES(?,?,?,?,?)",
            ("test.md", i, t, "{}", vec.tobytes()),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def model_cache() -> MagicMock:
    m = MagicMock(spec=ModelCache)
    m.embed_query.return_value = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    m.rerank.return_value = []
    return m


def test_search_returns_empty_for_no_match(tmp_path: Path, model_cache: MagicMock) -> None:
    db = tmp_path / "test.sqlite"
    _make_index(db, ["alpha", "beta", "gamma"])
    ic = IndexCache(max_entries=4)
    results = search_db(db, "query", model_cache=model_cache, index_cache=ic)
    assert results == []


def test_search_caches_index(tmp_path: Path, model_cache: MagicMock) -> None:
    db = tmp_path / "test.sqlite"
    _make_index(db, ["hello world"])
    ic = IndexCache(max_entries=4)
    search_db(db, "test", model_cache=model_cache, index_cache=ic)
    search_db(db, "test", model_cache=model_cache, index_cache=ic)
    assert ic.get(db) is not None


def test_search_handles_empty_index(tmp_path: Path, model_cache: MagicMock) -> None:
    db = tmp_path / "empty.sqlite"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    set_meta(conn, "embedding_model", "test")
    set_meta(conn, "embed_dim", "4")
    conn.close()
    ic = IndexCache(max_entries=4)
    with pytest.raises(ValueError, match="Index is empty"):
        search_db(db, "test", model_cache=model_cache, index_cache=ic)
