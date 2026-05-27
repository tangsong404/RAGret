from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document

from ragret.cache import IndexCache, ModelCache
from ragret.indexer import init_schema, set_meta
from ragret.searcher import search_db


def _make_index_with_lines(db_path: Path, dim: int = 4) -> None:
    conn = sqlite3.connect(str(db_path))
    meta = json.dumps({"line_start": 10, "line_end": 12, "source": "doc.md"})
    vec = np.full(dim, 0.9, dtype=np.float32)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            metadata_json TEXT,
            embedding BLOB NOT NULL
        );
        """
    )
    conn.execute("INSERT INTO meta(key, value) VALUES('embed_dim', ?)", (str(dim),))
    conn.execute(
        "INSERT INTO chunks(source, chunk_index, content, metadata_json, embedding) VALUES(?,?,?,?,?)",
        ("doc.md", 0, "hit text", meta, vec.tobytes()),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def model_cache() -> MagicMock:
    m = MagicMock(spec=ModelCache)
    m.embed_query.return_value = np.array([0.9, 0.1, 0.1, 0.1], dtype=np.float32)
    doc = Document(
        page_content="hit text",
        metadata={
            "source": "doc.md",
            "chunk_index": 0,
            "line_start": 10,
            "line_end": 12,
            "vector_score": 0.9,
            "relevance_score": 0.95,
        },
    )
    m.rerank.return_value = [doc]
    return m


def test_search_db_includes_parent_citation(tmp_path: Path, model_cache: MagicMock) -> None:
    db = tmp_path / "idx.sqlite"
    _make_index_with_lines(db)
    ic = IndexCache(max_entries=4)
    results = search_db(
        db,
        "query",
        model_cache=model_cache,
        index_cache=ic,
        kb_name="mykb",
        public_host="https://ragret.example.com",
    )
    assert len(results) == 1
    hit = results[0]
    assert hit["line_start"] == 10
    assert hit["line_end"] == 12
    assert hit["parent_url"] == "https://ragret.example.com/api/kb/mykb/parents/doc.md.txt"
