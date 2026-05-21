from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from ragret.cache import IndexCache, ModelCache
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
    order = np.argsort(-scores)

    candidates: list[Document] = []
    for idx in order:
        s = float(scores[idx])
        if s < score_threshold:
            continue
        r = records[int(idx)]
        meta = dict(r["metadata"])
        meta["source"] = meta.get("source") or r["source"]
        meta["chunk_index"] = r["chunk_index"]
        meta["vector_score"] = s
        candidates.append(Document(page_content=r["content"], metadata=meta))
        if len(candidates) >= k:
            break

    if not candidates:
        return []

    ranked = model_cache.rerank(query, candidates)
    ranked = ranked[:rerank_top_n]

    results = []
    for d in ranked:
        results.append(
            {
                "content": d.page_content,
                "source": str(d.metadata.get("source", "")),
                "chunk_index": int(d.metadata.get("chunk_index", 0)),
                "vector_score": float(d.metadata.get("vector_score", 0.0)),
                "relevance_score": float(d.metadata.get("relevance_score", 0.0)),
            }
        )
    return results
