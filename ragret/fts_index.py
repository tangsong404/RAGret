"""SQLite FTS5 BM25 index helpers and reciprocal rank fusion (RRF)."""
from __future__ import annotations

import os
import re
import sqlite3
import sys
from typing import Any

import numpy as np

RRF_K = max(1, int(os.environ.get("RAGRET_RRF_K", "60")))
RRF_DENSE_POOL = max(8, int(os.environ.get("RAGRET_RRF_DENSE_POOL", "48")))
RRF_BM25_POOL = max(8, int(os.environ.get("RAGRET_RRF_BM25_POOL", "48")))
RRF_FUSE_TOP = max(8, int(os.environ.get("RAGRET_RRF_FUSE_TOP", "32")))


def chunks_fts_ready(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts' LIMIT 1",
    ).fetchone()
    return row is not None


def try_init_chunks_fts(conn: sqlite3.Connection) -> bool:
    """Create FTS5 external-content index over chunks (no-op if already present)."""
    if chunks_fts_ready(conn):
        return True
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 0'
            );
            """
        )
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');")
        conn.commit()
        return True
    except sqlite3.OperationalError as e:
        print(
            f"Note: FTS5 BM25 index unavailable ({e}); using dense retrieval only.",
            file=sys.stderr,
        )
        conn.rollback()
        return False


def chunks_fts_rebuild(conn: sqlite3.Connection) -> None:
    if not chunks_fts_ready(conn):
        return
    try:
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');")
        conn.commit()
    except sqlite3.OperationalError:
        conn.rollback()


def fts5_match_query(query: str, *, max_terms: int = 16) -> str | None:
    """Build an FTS5 MATCH string (OR of quoted terms)."""
    terms: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"[\w\u0080-\U0010ffff]+", query, flags=re.UNICODE):
        t = (m.group(0) or "").strip()
        if len(t) < 1:
            continue
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(t)
        if len(terms) >= max_terms:
            break
    if not terms:
        return None
    parts = [f'"{t.replace(chr(34), chr(34) * 2)}"' for t in terms]
    return " OR ".join(parts)


def bm25_ranked_chunk_ids(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = RRF_BM25_POOL,
) -> list[int]:
    mq = fts5_match_query(query)
    if not mq:
        return []
    try:
        cur = conn.execute(
            """
            SELECT rowid
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts) ASC
            LIMIT ?
            """,
            (mq, int(limit)),
        )
        return [int(r[0]) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        return []


def reciprocal_rank_fusion(
    ranked_id_lists: list[list[int]],
    *,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ids in ranked_id_lists:
        if not ids:
            continue
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def dense_chunk_ids_for_rrf(
    scores: np.ndarray,
    records: list[dict[str, Any]],
    *,
    threshold: float,
    cap: int = RRF_DENSE_POOL,
) -> list[int]:
    """Ordered chunk ids for dense arm: prefer cosine >= threshold, else top scores."""
    order = np.argsort(-scores)
    out: list[int] = []
    seen: set[int] = set()
    for j in order:
        if len(out) >= cap:
            break
        s = float(scores[int(j)])
        if s < threshold:
            continue
        rid = int(records[int(j)]["id"])
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    if not out:
        for j in order[:cap]:
            rid = int(records[int(j)]["id"])
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
    return out
