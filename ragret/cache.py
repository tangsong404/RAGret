from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document

from ragret.rerank import RagretBCERerank


class ModelCache:
    """Injectable embedding + reranker cache. Replaces module-level _search_embed_models etc."""

    def __init__(self, device: str, rerank_top_n: int = 256) -> None:
        self._device = device
        self._rerank_top_n = rerank_top_n
        self._embed_lock = threading.Lock()
        self._rerank_lock = threading.Lock()
        self._embed_model: Any = None
        self._rerank_model: RagretBCERerank | None = None

    def get_embed_model(self) -> Any:
        if self._embed_model is None:
            with self._embed_lock:
                if self._embed_model is None:
                    from ragret.embedder import make_embed_model

                    self._embed_model = make_embed_model(self._device)
        return self._embed_model

    def get_rerank_model(self) -> RagretBCERerank:
        if self._rerank_model is None:
            with self._rerank_lock:
                if self._rerank_model is None:
                    from ragret.embedder import make_reranker

                    self._rerank_model = make_reranker(self._device, self._rerank_top_n)
        return self._rerank_model

    def embed_query(self, text: str) -> np.ndarray:
        model = self.get_embed_model()
        with self._embed_lock:
            return np.asarray(model.embed_query(text), dtype=np.float32)

    def rerank(self, query: str, candidates: list[Document]) -> list[Document]:
        want = max(1, self._rerank_top_n)
        from ragret.embedder import make_reranker

        if want > 256:
            reranker = make_reranker(self._device, top_n=want)
            with self._rerank_lock:
                return list(reranker.compress_documents(candidates, query))
        reranker = self.get_rerank_model()
        with self._rerank_lock:
            return list(reranker.compress_documents(candidates, query))


class IndexCache:
    """LRU cache for in-memory index snapshots (vectors + records). Thread-safe."""

    def __init__(self, max_entries: int = 64) -> None:
        self._max = max_entries
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[int, int, np.ndarray, list[dict[str, Any]], str | None]] = (
            OrderedDict()
        )

    def _file_stat_sig(self, path: Path) -> tuple[int, int]:
        st = path.stat()
        ns = getattr(st, "st_mtime_ns", None)
        if ns is None:
            ns = int(st.st_mtime * 1_000_000_000)
        return int(ns), int(st.st_size)

    def get(self, db_path: Path) -> tuple[np.ndarray, list[dict[str, Any]], str | None] | None:
        key = str(db_path.resolve())
        sig = self._file_stat_sig(db_path)
        with self._lock:
            ent = self._cache.get(key)
            if ent is not None and ent[0] == sig[0] and ent[1] == sig[1]:
                self._cache.move_to_end(key)
                return ent[2], ent[3], ent[4]
        return None

    def set(
        self,
        db_path: Path,
        matrix: np.ndarray,
        records: list[dict[str, Any]],
        stored_model: str | None,
    ) -> None:
        key = str(db_path.resolve())
        sig = self._file_stat_sig(db_path)
        with self._lock:
            while len(self._cache) >= self._max:
                self._cache.popitem(last=False)
            self._cache[key] = (sig[0], sig[1], matrix, records, stored_model)

    def invalidate(self, db_path: Path) -> None:
        key = str(db_path.resolve())
        with self._lock:
            self._cache.pop(key, None)
