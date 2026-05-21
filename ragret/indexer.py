"""SQLite index build: schema helpers and full workdir indexing."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from langchain_core.documents import Document

from ragret.embedder import (
    EMBEDDING_MODEL,
    BuildCancelledError,
    embed_batch,
    make_embed_model,
    resolve_device,
    reraise_if_missing_hf_weights,
)
from ragret.loader import (
    chunk_documents,
    fingerprint_map,
    load_documents_from_dir,
    load_one_file,
    relative_source_key,
)

IndexProgressFn = Callable[[str, int, str | None], None]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    metadata_json TEXT,
    embedding BLOB NOT NULL,
    UNIQUE(source, chunk_index)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_schema(conn: Any) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def set_meta(conn: Any, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def get_meta(conn: Any, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def clear_chunks(conn: Any) -> None:
    conn.execute("DELETE FROM chunks;")
    conn.commit()


def build_index(
    conn: Any,
    work_dir: Path,
    embed_model: Any,
    *,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    device: str | None = None,  # noqa: ARG001 — reserved for callers logging device
    progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    work_dir = work_dir.resolve()

    documents = load_documents_from_dir(work_dir)
    texts = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    contents = [d.page_content for d in texts]
    vectors = embed_batch(
        embed_model,
        contents,
        on_batch=progress,
        cancel_check=cancel_check,
    )
    if not vectors:
        raise RuntimeError("Embedding returned empty.")
    dim = len(vectors[0])
    arr = np.asarray(vectors, dtype=np.float32)

    init_schema(conn)
    clear_chunks(conn)
    set_meta(conn, "schema_version", "1")
    set_meta(conn, "embedding_model", EMBEDDING_MODEL)
    set_meta(conn, "embed_dim", str(dim))
    set_meta(conn, "indexed_work_dir", str(work_dir))
    set_meta(conn, "indexed_at", str(int(time.time())))

    last_src: str | None = None
    local_i = 0
    for i, doc in enumerate(texts):
        meta = json.dumps(doc.metadata, ensure_ascii=False)
        blob = arr[i].tobytes()
        src = relative_source_key(work_dir, str(doc.metadata.get("source", "") or ""))
        if src != last_src:
            local_i = 0
            last_src = src
        conn.execute(
            "INSERT INTO chunks(source, chunk_index, content, metadata_json, embedding) VALUES(?,?,?,?,?)",
            (src, local_i, doc.page_content, meta, blob),
        )
        local_i += 1

    fp_map = fingerprint_map(work_dir)
    set_meta(conn, "source_fingerprints", json.dumps(fp_map, sort_keys=True, ensure_ascii=False))
    set_meta(conn, "chunk_size", str(chunk_size))
    set_meta(conn, "chunk_overlap", str(chunk_overlap))
    conn.commit()
    return len(texts)


def _embed_on_batch(progress: IndexProgressFn | None) -> Callable[[int, int], None] | None:
    if progress is None:
        return None

    def on_batch(done: int, total: int) -> None:
        progress("embed", 15 + int(69 * done / max(total, 1)), f"{done}/{total}")

    return on_batch


def index_workdir(
    work_dir: Path,
    db_path: Path,
    *,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    device: str | None = None,
    progress: IndexProgressFn | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> int:
    """Full rebuild: open SQLite DB, embed corpus, write chunks. Legacy-compatible API."""
    work_dir = work_dir.resolve()
    db_path = db_path.resolve()
    device = device or resolve_device()

    if cancel_check is not None and cancel_check():
        raise BuildCancelledError("cancelled")

    try:
        embed_model = make_embed_model(device)
    except Exception as e:
        reraise_if_missing_hf_weights(e)

    conn = connect(db_path)
    try:
        return build_index(
            conn,
            work_dir,
            embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            device=device,
            progress=_embed_on_batch(progress),
            cancel_check=cancel_check,
        )
    finally:
        conn.close()


def try_incremental_update_workdir(
    work_dir: Path,
    db_path: Path,
    *,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    device: str | None = None,
    progress: IndexProgressFn | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> bool:
    """Apply minimal SQLite changes from a new corpus. Returns False if full rebuild is required."""
    work_dir = work_dir.resolve()
    db_path = db_path.resolve()
    if not db_path.is_file():
        return False

    device = device or resolve_device()

    def report(phase: str, pct: int, detail: str | None = None) -> None:
        if progress is not None:
            progress(phase, max(0, min(100, pct)), detail)

    conn = connect(db_path)
    try:
        init_schema(conn)
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if int(n_chunks or 0) == 0:
            return False
        raw_fp = get_meta(conn, "source_fingerprints")
        if not raw_fp:
            return False
        try:
            old_fp: dict[str, str] = json.loads(raw_fp)
        except json.JSONDecodeError:
            return False
        if not isinstance(old_fp, dict):
            return False
        stored_cs = get_meta(conn, "chunk_size")
        stored_co = get_meta(conn, "chunk_overlap")
        if stored_cs and int(stored_cs) != chunk_size:
            return False
        if stored_co and int(stored_co) != chunk_overlap:
            return False
        dim_s = get_meta(conn, "embed_dim")
        if not dim_s:
            return False
        dim = int(dim_s)
        emb_model_name = get_meta(conn, "embedding_model")
        if emb_model_name and emb_model_name != EMBEDDING_MODEL:
            return False
    finally:
        conn.close()

    try:
        new_fp = fingerprint_map(work_dir)
    except ValueError:
        return False
    if not new_fp:
        return False

    old_keys = set(old_fp.keys())
    new_keys = set(new_fp.keys())
    removed = old_keys - new_keys
    added_or_changed = {k for k in new_keys if k not in old_fp or old_fp[k] != new_fp[k]}

    if not removed and not added_or_changed:
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("cancelled")
        conn = connect(db_path)
        try:
            set_meta(conn, "indexed_work_dir", str(work_dir))
            set_meta(conn, "indexed_at", str(int(time.time())))
            set_meta(conn, "source_fingerprints", json.dumps(new_fp, sort_keys=True, ensure_ascii=False))
        finally:
            conn.close()
        report("done", 100, "no file changes")
        return True

    report("load", 8, f"+{len(added_or_changed)} ~{len(removed)} removed")
    if cancel_check is not None and cancel_check():
        raise BuildCancelledError("cancelled")

    try:
        embed_model = make_embed_model(device)
    except Exception as e:
        reraise_if_missing_hf_weights(e)

    conn = connect(db_path)
    try:
        for rel in sorted(removed):
            conn.execute("DELETE FROM chunks WHERE source = ?", (rel,))
        conn.commit()

        to_embed: list[tuple[str, Document]] = []
        for rel in sorted(added_or_changed):
            conn.execute("DELETE FROM chunks WHERE source = ?", (rel,))
            fp = (work_dir / rel).resolve()
            try:
                fp.relative_to(work_dir)
            except ValueError:
                conn.rollback()
                return False
            if not fp.is_file():
                conn.rollback()
                return False
            docs = load_one_file(fp)
            for d in docs:
                d.metadata["source"] = str(fp)
            parts = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for d in parts:
                to_embed.append((rel, d))
        conn.commit()

        if not to_embed:
            set_meta(conn, "source_fingerprints", json.dumps(new_fp, sort_keys=True, ensure_ascii=False))
            set_meta(conn, "indexed_work_dir", str(work_dir))
            set_meta(conn, "indexed_at", str(int(time.time())))
            report("done", 100, "deleted only")
            return True

        contents = [d.page_content for _, d in to_embed]
        n_emb = len(contents)
        report("chunk", 12, f"{n_emb} new chunks")

        try:
            vectors = embed_batch(
                embed_model,
                contents,
                on_batch=_embed_on_batch(progress),
                cancel_check=cancel_check,
            )
        except Exception as e:
            reraise_if_missing_hf_weights(e)
        if not vectors or len(vectors) != n_emb:
            raise RuntimeError("Embedding returned empty or wrong count.")
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.shape[1] != dim:
            raise RuntimeError("Embedding dimension mismatch; full rebuild required.")

        last_src: str | None = None
        local_i = 0
        for row_i, (rel, doc) in enumerate(to_embed):
            if rel != last_src:
                local_i = 0
                last_src = rel
            meta = json.dumps(doc.metadata, ensure_ascii=False)
            blob = arr[row_i].tobytes()
            conn.execute(
                "INSERT INTO chunks(source, chunk_index, content, metadata_json, embedding) VALUES(?,?,?,?,?)",
                (rel, local_i, doc.page_content, meta, blob),
            )
            local_i += 1
            step = max(1, n_emb // 20)
            if (row_i + 1) % step == 0 or row_i + 1 == n_emb:
                if progress is not None:
                    pct = 85 + int(13 * (row_i + 1) / max(n_emb, 1))
                    report("sqlite", min(99, pct), f"{row_i + 1}/{n_emb}")

        set_meta(conn, "source_fingerprints", json.dumps(new_fp, sort_keys=True, ensure_ascii=False))
        set_meta(conn, "indexed_work_dir", str(work_dir))
        set_meta(conn, "indexed_at", str(int(time.time())))
        set_meta(conn, "chunk_size", str(chunk_size))
        set_meta(conn, "chunk_overlap", str(chunk_overlap))
        conn.commit()
    finally:
        conn.close()

    report("done", 100, None)
    return True
