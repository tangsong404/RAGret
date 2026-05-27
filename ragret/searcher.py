from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from ragret.cache import IndexCache, ModelCache
from ragret.fts_index import (
    RRF_BM25_POOL,
    RRF_DENSE_POOL,
    RRF_FUSE_TOP,
    RRF_K,
    bm25_ranked_chunk_ids,
    chunks_fts_ready,
    dense_chunk_ids_for_rrf,
    reciprocal_rank_fusion,
)
from ragret.citation_urls import build_parent_url
from ragret.indexer import get_meta

EMBEDDING_MODEL = "maidalun1020/bce-embedding-base_v1"


def _load_index_snapshot(db_path: Path) -> tuple[np.ndarray, list[dict], str | None]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        stored_model = get_meta(conn, "embedding_model")
        dim_s = get_meta(conn, "embed_dim")
        if not dim_s:
            raise ValueError("Missing embed_dim in meta table.")
        dim = int(dim_s)
        rows = conn.execute(
            "SELECT id, source, chunk_index, content, metadata_json, embedding FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            raise ValueError("Index is empty. Build it first.")
        embs = []
        records = []
        for rid, source, cidx, content, meta_json, emb_blob in rows:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            if vec.size != dim:
                raise ValueError(f"Embedding size mismatch for id={rid}")
            embs.append(vec)
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except json.JSONDecodeError:
                meta = {}
            records.append(
                {
                    "id": rid,
                    "source": source,
                    "chunk_index": cidx,
                    "content": content,
                    "metadata": meta,
                }
            )
        matrix = np.stack(embs, axis=0)
        return matrix, records, stored_model
    finally:
        conn.close()


def search_db(
    db_path: Path,
    query: str,
    *,
    model_cache: ModelCache,
    index_cache: IndexCache,
    k: int = 10,
    score_threshold: float = 0.3,
    rerank_top_n: int = 5,
    kb_name: str | None = None,
    public_host: str | None = None,
) -> list[dict]:
    db_path = db_path.resolve()

    cached = index_cache.get(db_path)
    if cached is not None:
        matrix, records, stored_model = cached
    else:
        matrix, records, stored_model = _load_index_snapshot(db_path)
        index_cache.set(db_path, matrix, records, stored_model)

    if stored_model and stored_model != EMBEDDING_MODEL:
        import sys

        print(
            f"Warning: index was built with {stored_model}, this build expects {EMBEDDING_MODEL}.",
            file=sys.stderr,
        )

    q = model_cache.embed_query(query)
    scores = matrix @ q
    by_id = {int(r["id"]): r for r in records}
    dense_scores_by_id = {int(records[j]["id"]): float(scores[j]) for j in range(len(records))}
    dense_ids = dense_chunk_ids_for_rrf(
        scores,
        records,
        threshold=float(score_threshold),
        cap=RRF_DENSE_POOL,
    )

    bm25_ids: list[int] = []
    fts5_indexed = False
    conn_ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        fts5_indexed = chunks_fts_ready(conn_ro)
        if fts5_indexed:
            bm25_ids = bm25_ranked_chunk_ids(conn_ro, query, limit=RRF_BM25_POOL)
    finally:
        conn_ro.close()

    fused = reciprocal_rank_fusion([dense_ids, bm25_ids], k=RRF_K)
    fuse_take = min(RRF_FUSE_TOP, max(k * 3, int(rerank_top_n) * 4), len(fused))
    fused = fused[:fuse_take]

    bm25_rank_by_id = {cid: rank for rank, cid in enumerate(bm25_ids, start=1)}
    dense_rank_by_id = {cid: rank for rank, cid in enumerate(dense_ids, start=1)}

    candidates: list[Document] = []
    for chunk_id, rrf_s in fused:
        r = by_id.get(int(chunk_id))
        if r is None:
            continue
        meta = dict(r["metadata"])
        meta["source"] = meta.get("source") or r["source"]
        meta["chunk_index"] = r["chunk_index"]
        meta["vector_score"] = float(dense_scores_by_id.get(int(chunk_id), 0.0))
        meta["rrf_score"] = float(rrf_s)
        dr = dense_rank_by_id.get(int(chunk_id))
        br = bm25_rank_by_id.get(int(chunk_id))
        if dr is not None:
            meta["dense_rank"] = dr
        if br is not None:
            meta["bm25_rank"] = br
        candidates.append(Document(page_content=r["content"], metadata=meta))

    if not candidates:
        return []

    ranked = model_cache.rerank(query, candidates)
    ranked = ranked[:rerank_top_n]

    results = []
    for d in ranked:
        source = str(d.metadata.get("source", ""))
        row: dict = {
            "content": d.page_content,
            "source": source,
            "chunk_index": int(d.metadata.get("chunk_index", 0)),
            "vector_score": float(d.metadata.get("vector_score", 0.0)),
            "relevance_score": float(d.metadata.get("relevance_score", 0.0)),
        }
        if "rrf_score" in d.metadata:
            row["rrf_score"] = float(d.metadata["rrf_score"])
        if "dense_rank" in d.metadata:
            row["dense_rank"] = int(d.metadata["dense_rank"])
        if "bm25_rank" in d.metadata:
            row["bm25_rank"] = int(d.metadata["bm25_rank"])
        line_start = d.metadata.get("line_start")
        line_end = d.metadata.get("line_end")
        if line_start is not None:
            row["line_start"] = int(line_start)
        if line_end is not None:
            row["line_end"] = int(line_end)
        if kb_name and source:
            row["parent_url"] = build_parent_url(
                kb_name=kb_name,
                source_key=source,
                public_host=public_host,
            )
        results.append(row)
    return results
