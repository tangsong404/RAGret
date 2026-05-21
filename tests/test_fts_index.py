from __future__ import annotations

import sqlite3

import pytest

from ragret.fts_index import (
    bm25_ranked_chunk_ids,
    chunks_fts_ready,
    fts5_match_query,
    reciprocal_rank_fusion,
    try_init_chunks_fts,
)


def test_reciprocal_rank_fusion_merges_two_lists() -> None:
    fused = reciprocal_rank_fusion([[10, 20], [20, 30]], k=60)
    ids = [x[0] for x in fused]
    assert ids[0] == 20
    assert 10 in ids and 30 in ids


def test_fts5_match_query_or_terms() -> None:
    q = fts5_match_query("hello world test")
    assert q is not None
    assert "hello" in q and "world" in q
    assert " OR " in q


def test_bm25_roundtrip_when_fts5_available() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            metadata_json TEXT,
            embedding BLOB NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO chunks(id, source, chunk_index, content, metadata_json, embedding) VALUES (1, 'a', 0, 'alpha beta gamma', '{}', X'00');"
    )
    conn.execute(
        "INSERT INTO chunks(id, source, chunk_index, content, metadata_json, embedding) VALUES (2, 'a', 1, 'delta epsilon', '{}', X'00');"
    )
    conn.commit()
    if not try_init_chunks_fts(conn):
        pytest.skip("SQLite build lacks FTS5")
    assert chunks_fts_ready(conn)
    hits = bm25_ranked_chunk_ids(conn, "beta", limit=5)
    assert 1 in hits
