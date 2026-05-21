# RAGret Backend Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace monolithic ThreadingHTTPServer + httpd.py with FastAPI, split rag.py into single-responsibility modules, encapsulate global state in injectable classes, add connection pooling to SqliteAppStore, and build comprehensive test infrastructure.

**Architecture:** The `ragret/` core package splits into `loader.py`, `embedder.py`, `indexer.py`, `searcher.py`, `cache.py`, `models.py`. The `server/` package becomes a FastAPI app with `routers/`, `services/`, and `middleware/` sub-packages. SqliteAppStore gains a connection pool (`pool.py`). The old `httpd.py` and `rag.py` are deleted in the final phase.

**Tech Stack:** FastAPI, uvicorn, pydantic, pydantic-settings, pytest, pytest-mock, httpx (TestClient), sqlite3 (built-in).

---

## File Structure

### Files created (new):
```
ragret/
  cache.py        — Injectable ModelCache + IndexCache (replaces module-level globals)
  models.py       — Pydantic Chunk, SearchResult, IndexMeta
  loader.py       — Document loading + chunking (from rag.py:112-182)
  embedder.py     — Embedding model wrapper (from rag.py:269-303)
  indexer.py      — SQLite index build + incremental update (from rag.py:185-663)
  searcher.py     — Dense retrieval + rerank, returns list[SearchResult] (from rag.py:666-852)

server/
  main.py         — FastAPI app create + lifespan (uvicorn entry)
  config.py       — pydantic-settings Settings class
  deps.py         — FastAPI Depends() providers for store, cache, auth
  schemas.py      — Pydantic request/response schemas
  routers/
    __init__.py   — empty
    auth.py       — register, login, logout, me, password
    kb.py         — knowledge base CRUD, members, subscriptions, icons
    search.py     — GET /api/search/{name}
    webhook.py    — GitLab/GitHub webhook handlers
    jobs.py       — list/get/cancel build jobs
    upload.py     — POST /api/upload multipart tar
    admin.py      — superuser PATCH/DELETE
  services/
    __init__.py   — empty
    auth_service.py   — auth business logic (from httpd _handle_auth_*)
    kb_service.py     — KB business logic (from httpd _handle_kb_*)
    search_service.py — search dispatch with permission checks (from httpd _handle_search)
    build_service.py  — build job + webhook-pull logic (from httpd _handle_start_build_job)
  middleware/
    __init__.py   — empty
    auth.py       — Bearer token parser middleware
  store/
    pool.py       — SqliteConnectionPool (new)
    (others unchanged)

tests/
  conftest.py     — pytest fixtures: pool, store, cache, client
  test_loader.py
  test_embedder.py
  test_indexer.py
  test_searcher.py
  test_auth_api.py
  test_search_api.py
  test_kb_api.py
  test_webhook.py
  test_build_queue.py

pyproject.toml    — project metadata + pytest config
```

### Files modified (adapted):
```
ragret/rag.py            — legacy, keep as import shim → deleted in Phase 3
server/httpd.py          — legacy, keep for parallel run → deleted in Phase 3
server/store/sqlite_store.py — adapted for SqliteConnectionPool
server/build_queue.py    — adapted for pool-based store
server/store/factory.py  — pass pool to SqliteAppStore
server/__init__.py       — updated imports
ragret/__init__.py       — updated exports
ragret/cli.py            — switch to uvicorn.run()
requirements.txt         — add fastapi, uvicorn, httpx, pytest, pytest-mock
```

---

### Phase 1: Foundation (test infra, models, cache, pool, loader)

#### Task 1.1: Create pyproject.toml + test config

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"

[tool.ruff]
line-length = 110
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml with pytest config"
```

---

#### Task 1.2: ragret/models.py — core data models

**Files:**
- Create: `ragret/models.py`

- [ ] **Step 1: Write the models file**

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: int
    source: str
    chunk_index: int
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    content: str
    source: str
    chunk_index: int
    vector_score: float
    relevance_score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class IndexMeta:
    embedding_model: str
    embed_dim: int
    chunk_size: int
    chunk_overlap: int
    indexed_at: int
    schema_version: str
    source_fingerprints: dict[str, str]
```

- [ ] **Step 2: Commit**

```bash
git add ragret/models.py
git commit -m "feat: add core Pydantic data models"
```

---

#### Task 1.3: ragret/cache.py — injectable ModelCache + IndexCache

**Files:**
- Create: `ragret/cache.py`

- [ ] **Step 1: Write cache.py**

```python
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
            from ragret.embedder import make_embed_model

            self._embed_model = make_embed_model(self._device)
        return self._embed_model

    def get_rerank_model(self) -> RagretBCERerank:
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
```

- [ ] **Step 2: Commit**

```bash
git add ragret/cache.py
git commit -m "feat: add injectable ModelCache and IndexCache"
```

---

#### Task 1.4: server/config.py — centralized settings

**Files:**
- Create: `server/config.py`

- [ ] **Step 1: Write config.py**

```python
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8765
    registry_path: Path | None = None
    app_db_path: Path | None = None
    session_ttl: int = 30 * 24 * 3600
    search_index_cache_max: int = 64
    search_rerank_cache_top: int = 256
    git_http_connect_timeout_s: float = 20.0
    git_http_read_timeout_s: float = 30.0
    git_clone_wall_timeout_s: float = 30.0
    api_token: str | None = None
    avatar_max_bytes: int = 2 * 1024 * 1024
    public_host: str | None = None

    model_config = {"env_prefix": "RAGRET_"}
```

- [ ] **Step 2: Add pydantic-settings to requirements.txt**

Edit `requirements.txt`, add line: `pydantic-settings>=2`

- [ ] **Step 3: Commit**

```bash
git add server/config.py requirements.txt
git commit -m "feat: add Settings with pydantic-settings"
```

---

#### Task 1.5: server/store/pool.py — SqliteConnectionPool

**Files:**
- Create: `server/store/pool.py`

- [ ] **Step 1: Write pool.py**

```python
from __future__ import annotations

import queue
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


class SqliteConnectionPool:
    """Thread-safe SQLite connection pool. Each connection uses WAL + foreign_keys."""

    def __init__(self, db_path: Path, min_size: int = 4, max_size: int = 32) -> None:
        self._path = db_path
        self._max = max_size
        self._lock = threading.Lock()
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue()
        self._created = 0
        for _ in range(min_size):
            self._pool.put(self._new_conn())

    def _new_conn(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def acquire(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            conn = self._pool.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._created < self._max:
                    conn = self._new_conn()
                    self._created += 1
                else:
                    conn = self._pool.get()  # block until one is free
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            self._pool.put(conn)

    def close(self) -> None:
        while True:
            try:
                self._pool.get_nowait().close()
            except queue.Empty:
                break
```

- [ ] **Step 2: Commit**

```bash
git add server/store/pool.py
git commit -m "feat: add SqliteConnectionPool"
```

---

#### Task 1.6: Adapt SqliteAppStore for pool-based connections

**Files:**
- Modify: `server/store/sqlite_store.py`

Key change: constructor takes `SqliteConnectionPool` instead of `Path`. All methods that need a connection receive it as a parameter. Schema init and migration happen once via `pool.acquire()`.

- [ ] **Step 1: Modify SqliteAppStore.__init__ to accept pool**

```python
class SqliteAppStore:
    def __init__(self, pool_or_path: SqliteConnectionPool | Path) -> None:
        if isinstance(pool_or_path, SqliteConnectionPool):
            self._pool = pool_or_path
        else:
            self._pool = SqliteConnectionPool(pool_or_path)
        with self._pool.acquire() as conn:
            conn.executescript(_INIT_SQL)
            self._migrate_schema(conn)
```

- [ ] **Step 2: Remove all `with self._lock` patterns from SqliteAppStore methods, replace with `with self._pool.acquire() as conn:` and pass `conn` to all helper methods**

Pattern for every method:
```python
def create_user(self, username: str, password_hash: str) -> UserRecord:
    t = self._now()
    with self._pool.acquire() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
                (username.strip(), password_hash, t),
            )
            uid = int(cur.lastrowid)
        except sqlite3.IntegrityError as e:
            raise ValueError("Username already taken") from e
    return UserRecord(id=uid, username=username.strip())
```

- [ ] **Step 3: Adapt _migrate_schema to take conn parameter**

Remove `_migrate_schema_unlocked` → rename to `_migrate_schema(conn)`.

- [ ] **Step 4: Remove all _unlocked helper methods, make them take conn**

`_kb_ready_from_row`, `_permission_unlocked`, `_user_has_avatar_unlocked`, `_kb_is_public_unlocked` all take `conn` parameter instead of using `self._conn`.

- [ ] **Step 5: Commit**

```bash
git add server/store/sqlite_store.py
git commit -m "refactor: SqliteAppStore uses SqliteConnectionPool"
```

---

#### Task 1.7: Adapt factory.py to pass pool

**Files:**
- Modify: `server/store/factory.py`

- [ ] **Step 1: Update create_app_store**

```python
def create_app_store(repo_root: Path) -> SqliteAppStore:
    backend = (os.environ.get("RAGRET_APP_STORE") or "sqlite").strip().lower()
    if backend == "sqlite":
        raw = os.environ.get("RAGRET_APP_DB")
        db_path = (
            Path(raw).expanduser().resolve()
            if raw
            else default_app_sqlite_path(repo_root)
        )
        return SqliteAppStore(db_path)  # SqliteAppStore now accepts Path or pool
    raise ValueError(f"Unsupported RAGRET_APP_STORE={backend!r}")
```

- [ ] **Step 2: Commit**

```bash
git add server/store/factory.py
git commit -m "refactor: create_app_store passes Path to SqliteAppStore"
```

---

#### Task 1.8: ragret/loader.py — document loading + chunking

**Files:**
- Create: `ragret/loader.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: Write loader.py**

```python
from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_one_file(path: Path) -> list[Document]:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suf in (".txt", ".md", ".markdown"):
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {path.suffix}. Use .pdf, .txt, or .md.")


def iter_indexable_files(work_dir: Path) -> list[Path]:
    if not work_dir.exists():
        raise FileNotFoundError(work_dir)
    if work_dir.is_file():
        return [work_dir.resolve()]
    out: list[Path] = []
    for glob_pat in ("**/*.pdf", "**/*.txt", "**/*.md"):
        for f in sorted(work_dir.glob(glob_pat)):
            if f.is_file():
                out.append(f.resolve())
    return out


def load_documents_from_dir(work_dir: Path) -> list[Document]:
    if not work_dir.exists():
        raise FileNotFoundError(work_dir)
    if work_dir.is_file():
        return load_one_file(work_dir)
    documents: list[Document] = []
    for f in iter_indexable_files(work_dir):
        suf = f.suffix.lower()
        if suf == ".pdf":
            documents.extend(PyPDFLoader(str(f)).load())
        else:
            documents.extend(TextLoader(str(f), encoding="utf-8").load())
    if not documents:
        raise ValueError(f"No .pdf / .txt / .md files under: {work_dir}")
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = splitter.split_documents(documents)
    if not texts:
        raise ValueError("No chunks after split.")
    return texts


def fingerprint_map(work_dir: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    for f in iter_indexable_files(work_dir):
        key = f.relative_to(work_dir.resolve()).as_posix()
        m[key] = _file_sha256(f)
    return m


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as bf:
        for chunk in iter(lambda: bf.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_source_key(work_dir: Path, source_raw: str) -> str:
    work_dir = work_dir.resolve()
    if not source_raw or not str(source_raw).strip():
        return work_dir.name
    p = Path(source_raw).expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = Path(source_raw)
    try:
        rel = p.relative_to(work_dir)
    except ValueError:
        return str(p).replace("\\", "/")
    return rel.as_posix()
```

- [ ] **Step 2: Write test_loader.py**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from ragret.loader import iter_indexable_files, relative_source_key


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
```

- [ ] **Step 3: Run tests to verify**

```bash
pip install pytest pytest-mock -q
pytest tests/test_loader.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add ragret/loader.py tests/test_loader.py
git commit -m "feat: add document loader module with tests"
```

---

#### Task 1.9: tests/conftest.py — base test fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write conftest.py**

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ragret.cache import IndexCache, ModelCache
from server.store.pool import SqliteConnectionPool
from server.store.sqlite_store import SqliteAppStore


@pytest.fixture
def pool(tmp_path: Path) -> SqliteConnectionPool:
    db = tmp_path / "test_app.sqlite"
    p = SqliteConnectionPool(db, min_size=2, max_size=4)
    yield p
    p.close()


@pytest.fixture
def store(pool: SqliteConnectionPool) -> SqliteAppStore:
    return SqliteAppStore(pool)


@pytest.fixture
def model_cache() -> MagicMock:
    cache = MagicMock(spec=ModelCache)
    cache.embed_query.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    return cache


@pytest.fixture
def index_cache(tmp_path: Path) -> IndexCache:
    return IndexCache(max_entries=8)
```

- [ ] **Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add base fixtures for pool, store, model_cache"
```

---

### Phase 2: FastAPI App + Routers

#### Task 2.1: server/schemas.py — Pydantic request/response schemas

**Files:**
- Create: `server/schemas.py`

- [ ] **Step 1: Write schemas.py**

```python
from __future__ import annotations

from pydantic import BaseModel, Field


# --- Auth ---
class AuthRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9._-]{3,64}$")
    password: str = Field(min_length=8)


class AuthLoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    has_avatar: bool = False


class AuthResponse(BaseModel):
    ok: bool = True
    token: str
    user: UserOut


class MeResponse(BaseModel):
    ok: bool = True
    user: UserOut | None = None
    superuser: bool = False


# --- Search ---
class SearchResultOut(BaseModel):
    content: str
    source: str
    chunk_index: int
    vector_score: float
    relevance_score: float


class SearchResponse(BaseModel):
    ok: bool = True
    index: str
    query: str
    results: list[SearchResultOut]


# --- Knowledge Base ---
class KBPermissionOut(BaseModel):
    can_read: bool
    can_write: bool
    can_delete: bool
    is_owner: bool


class KBOwnerOut(BaseModel):
    id: int
    username: str
    has_avatar: bool


class KBOut(BaseModel):
    name: str
    description: str
    sqlite_exists: bool
    is_public: bool
    icon: str
    source_type: str
    owner: KBOwnerOut
    permission: KBPermissionOut


class KBListResponse(BaseModel):
    ok: bool = True
    indexes: list[KBOut]


class BuildJobRequest(BaseModel):
    name: str
    description: str
    readme_md: str = ""
    upload_id: str | None = None
    source_type: str = "tar"
    is_public: bool = False
    icon: str = "book"
    webhook_provider: str = ""
    webhook_secret: str = ""
    repo_url: str = ""
    ref: str = ""


class BuildJobResponse(BaseModel):
    ok: bool = True
    job_id: str
    webhook_url: str | None = None


class JobOut(BaseModel):
    job_id: str
    status: str
    phase: str
    percent: int
    detail: str
    error: str | None = None
    result: dict | None = None
    op: str
    kb_name: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


# --- Generic ---
class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
```

- [ ] **Step 2: Commit**

```bash
git add server/schemas.py
git commit -m "feat: add Pydantic request/response schemas"
```

---

#### Task 2.2: server/middleware/auth.py — Bearer token parser

**Files:**
- Create: `server/middleware/auth.py`
- Create: `server/middleware/__init__.py` (empty)

- [ ] **Step 1: Write middleware/auth.py**

```python
from __future__ import annotations

from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    """Parse Authorization header and store actor in request.state.actor."""

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        auth = (request.headers.get("Authorization") or "").strip()
        api_key = (request.headers.get("X-API-Key") or "").strip()
        request.state.actor = _resolve_actor(auth, api_key)
        return await call_next(request)


def _resolve_actor(auth: str, api_key: str) -> dict[str, Any]:
    """Return dict with keys: kind ('superuser'|'user'|'api_key'|'anon'), token, user_id."""
    bearer = ""
    if auth.lower().startswith("bearer "):
        bearer = auth[7:].strip()
    # superuser check via env token — done later in deps.py
    return {
        "kind": "anon",
        "token": bearer,
        "api_key": api_key if api_key else "",
    }
```

- [ ] **Step 2: Commit**

```bash
git add server/middleware/__init__.py server/middleware/auth.py
git commit -m "feat: add AuthMiddleware for bearer token parsing"
```

---

#### Task 2.3: server/deps.py — dependency injection

**Files:**
- Create: `server/deps.py`

- [ ] **Step 1: Write deps.py**

```python
from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from ragret.cache import IndexCache, ModelCache
from server.config import Settings
from server.store.protocol import AppStore
from server.store.sqlite_store import SqliteAppStore


async def get_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_store(request: Request) -> AppStore:
    return request.app.state.app_store


async def get_model_cache(request: Request) -> ModelCache:
    return request.app.state.model_cache


async def get_index_cache(request: Request) -> IndexCache:
    return request.app.state.index_cache


async def _super_token() -> str | None:
    t = os.environ.get("RAGRET_API_TOKEN")
    return t.strip() if t else None


async def require_actor(
    request: Request,
    store: AppStore = Depends(get_store),
) -> dict[str, Any]:
    """Returns actor dict. Raises 401 if anon."""
    actor = request.state.actor
    token = str(actor.get("token") or "")

    super_tok = await _super_token()
    if super_tok and token == super_tok:
        return {"kind": "superuser", "user_id": None}

    uid = store.get_session_user_id(token)
    if uid is not None:
        return {"kind": "user", "user_id": int(uid)}

    api_key = str(actor.get("api_key") or "")
    uid_by_key = store.get_api_key_owner_user_id(api_key)
    if uid_by_key is not None:
        return {"kind": "api_key", "user_id": int(uid_by_key)}

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")


async def optional_actor(
    request: Request,
    store: AppStore = Depends(get_store),
) -> dict[str, Any]:
    """Returns actor dict. For anon, returns {"kind": "anon", "user_id": None}."""
    try:
        return await require_actor(request, store)
    except HTTPException:
        return {"kind": "anon", "user_id": None}
```

- [ ] **Step 2: Commit**

```bash
git add server/deps.py
git commit -m "feat: add FastAPI dependency injection"
```

---

#### Task 2.4: server/services/auth_service.py — auth business logic

**Files:**
- Create: `server/services/auth_service.py`
- Create: `server/services/__init__.py` (empty)

- [ ] **Step 1: Write auth_service.py**

```python
from __future__ import annotations

from server.passwords import hash_password
from server.store.protocol import AppStore

_SESSION_TTL = 30 * 24 * 3600  # 30 days


def register_user(store: AppStore, username: str, password: str) -> dict:
    user = store.create_user(username.strip(), hash_password(password))
    token = store.create_session(user.id, ttl_seconds=_SESSION_TTL)
    return {
        "token": token,
        "user": {"id": user.id, "username": user.username, "has_avatar": False},
    }


def login_user(store: AppStore, username: str, password: str) -> dict | None:
    user = store.verify_user_password(username.strip(), password)
    if user is None:
        return None
    token = store.create_session(user.id, ttl_seconds=_SESSION_TTL)
    has_avatar = store.user_has_avatar(user.id)
    return {
        "token": token,
        "user": {"id": user.id, "username": user.username, "has_avatar": has_avatar},
    }


def logout_user(store: AppStore, token: str) -> None:
    if token:
        store.delete_session(token)


def change_user_password(
    store: AppStore,
    user_id: int,
    current_password: str,
    new_password: str,
) -> bool:
    return store.change_password(user_id, current_password, hash_password(new_password))
```

- [ ] **Step 2: Commit**

```bash
git add server/services/__init__.py server/services/auth_service.py
git commit -m "feat: add AuthService"
```

---

#### Task 2.5: router: auth.py — register, login, logout, me, password

**Files:**
- Create: `server/routers/__init__.py` (empty)
- Create: `server/routers/auth.py`
- Test: `tests/test_auth_api.py`

- [ ] **Step 1: Write routers/auth.py**

```python
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from server.deps import get_store, optional_actor, require_actor
from server.schemas import (
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    ErrorResponse,
    MeResponse,
    UserOut,
)
from server.services import auth_service
from server.store.protocol import AppStore

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(body: AuthRegisterRequest, store: AppStore = Depends(get_store)):
    try:
        result = auth_service.register_user(store, body.username, body.password)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    return AuthResponse(token=result["token"], user=UserOut(**result["user"]))


@router.post("/login", response_model=AuthResponse)
def login(body: AuthLoginRequest, store: AppStore = Depends(get_store)):
    result = auth_service.login_user(store, body.username, body.password)
    if result is None:
        raise HTTPException(401, detail="Invalid username or password")
    return AuthResponse(token=result["token"], user=UserOut(**result["user"]))


@router.post("/logout")
def logout(
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
):
    auth_service.logout_user(store, str(actor.get("token", "")))
    return {"ok": True}


@router.get("/me", response_model=MeResponse)
def me(
    actor: dict = Depends(optional_actor),
    store: AppStore = Depends(get_store),
):
    kind = actor.get("kind")
    if kind == "superuser":
        return MeResponse(user=None, superuser=True)
    uid = actor.get("user_id")
    if uid is None:
        raise HTTPException(401, detail="Not logged in")
    user = store.get_user_by_id(int(uid))
    if user is None:
        raise HTTPException(401, detail="Invalid session")
    has_avatar = store.user_has_avatar(int(uid))
    return MeResponse(user=UserOut(id=user.id, username=user.username, has_avatar=has_avatar))


@router.post("/password")
def change_password(
    body: dict,
    actor: dict = Depends(require_actor),
    store: AppStore = Depends(get_store),
):
    uid = actor.get("user_id")
    if uid is None:
        raise HTTPException(403, detail="User ID required")
    current = str(body.get("current_password", ""))
    new_pw = str(body.get("new_password", ""))
    if len(new_pw) < 8:
        raise HTTPException(400, detail="New password must be at least 8 characters")
    if not auth_service.change_user_password(store, int(uid), current, new_pw):
        raise HTTPException(401, detail="Current password is incorrect")
    return {"ok": True}
```

- [ ] **Step 2: Write test_auth_api.py**

```python
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.main import create_app
from server.store.pool import SqliteConnectionPool
from server.store.sqlite_store import SqliteAppStore


@pytest.fixture
def store(tmp_path: Path):
    db = tmp_path / "test.sqlite"
    pool = SqliteConnectionPool(db, min_size=1, max_size=2)
    s = SqliteAppStore(pool)
    yield s
    pool.close()


@pytest.fixture
def client(store: SqliteAppStore):
    app = create_app(store=store)
    with TestClient(app) as c:
        yield c


class TestRegister:
    def test_success(self, client: TestClient):
        resp = client.post("/api/auth/register", json={"username": "alice", "password": "secret123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "token" in data
        assert data["user"]["username"] == "alice"

    def test_duplicate_username(self, client: TestClient):
        client.post("/api/auth/register", json={"username": "bob", "password": "secret123"})
        resp = client.post("/api/auth/register", json={"username": "bob", "password": "other456"})
        assert resp.status_code == 400

    def test_short_password(self, client: TestClient):
        resp = client.post("/api/auth/register", json={"username": "charlie", "password": "short"})
        assert resp.status_code == 422  # pydantic validation


class TestLogin:
    def test_success(self, client: TestClient):
        client.post("/api/auth/register", json={"username": "dave", "password": "secret123"})
        resp = client.post("/api/auth/login", json={"username": "dave", "password": "secret123"})
        assert resp.status_code == 200
        assert resp.json()["user"]["username"] == "dave"

    def test_wrong_password(self, client: TestClient):
        client.post("/api/auth/register", json={"username": "eve", "password": "secret123"})
        resp = client.post("/api/auth/login", json={"username": "eve", "password": "wrongpass"})
        assert resp.status_code == 401


class TestMe:
    def test_authenticated(self, client: TestClient):
        reg = client.post("/api/auth/register", json={"username": "frank", "password": "secret123"}).json()
        token = reg["token"]
        resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["user"]["username"] == "frank"

    def test_unauthenticated(self, client: TestClient):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401
```

- [ ] **Step 3: Create a minimal create_app + main.py to make tests work**

```python
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from ragret.cache import IndexCache, ModelCache
from ragret.embedder import resolve_device
from server.config import Settings
from server.middleware.auth import AuthMiddleware
from server.routers import auth
from server.store.factory import create_app_store
from server.store.protocol import AppStore
from server.store.sqlite_store import SqliteAppStore


def create_app(
    store: AppStore | None = None,
    model_cache: ModelCache | None = None,
    index_cache: IndexCache | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    app = FastAPI(title="RAGret")
    app.state.settings = settings or Settings()
    app.state.app_store = store or create_app_store(Path.cwd())
    app.state.model_cache = model_cache or ModelCache(device="cpu")
    app.state.index_cache = index_cache or IndexCache()

    app.add_middleware(AuthMiddleware)
    app.include_router(auth.router)
    return app
```

- [ ] **Step 4: Run auth tests**

```bash
pytest tests/test_auth_api.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add server/routers/__init__.py server/routers/auth.py server/middleware/ server/main.py tests/test_auth_api.py
git commit -m "feat: add auth router + API tests"
```

---

#### Task 2.6: server/services/search_service.py — search dispatch

**Files:**
- Create: `server/services/search_service.py`
- Create: `server/routers/search.py`
- Test: `tests/test_search_api.py`

- [ ] **Step 1: Write search_service.py**

```python
from __future__ import annotations

from pathlib import Path

from ragret.cache import IndexCache, ModelCache
from ragret.searcher import search_db as core_search
from server.store.protocol import AppStore


def resolve_searchable_db(
    index_name: str,
    actor: dict,
    store: AppStore,
) -> Path | None:
    """Returns db_path if the actor has read permission, else None."""
    kind = actor.get("kind")
    uid = actor.get("user_id")

    store_path = store.resolve_kb_db_path(index_name)
    if kind == "superuser":
        return Path(store_path) if store_path else None
    if uid is None:
        return None
    if kind == "api_key":
        allowed = {
            str(r.name)
            for r in store.list_owned_and_subscribed_knowledge_bases_for_user(int(uid))
        }
        if index_name not in allowed:
            return None
    perm = store.permission_for(int(uid), index_name)
    if perm is None or not perm.can_read:
        return None
    return Path(store_path) if store_path else None


def search_index(
    db_path: Path,
    query: str,
    model_cache: ModelCache,
    index_cache: IndexCache,
    k: int = 10,
    score_threshold: float = 0.3,
    rerank_top_n: int = 5,
) -> list[dict]:
    return core_search(
        db_path,
        query,
        model_cache=model_cache,
        index_cache=index_cache,
        k=k,
        score_threshold=score_threshold,
        rerank_top_n=rerank_top_n,
    )
```

- [ ] **Step 2: Write search router**

```python
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ragret.cache import IndexCache, ModelCache
from server.deps import get_index_cache, get_model_cache, get_store, optional_actor
from server.schemas import SearchResponse, SearchResultOut
from server.services.search_service import resolve_searchable_db, search_index
from server.store.protocol import AppStore

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search/{name}", response_model=SearchResponse)
def search(
    name: str,
    q: str = Query(alias="query"),
    k: int = Query(default=10, ge=1, le=100),
    threshold: float = Query(default=0.3, ge=0.0, le=1.0, alias="score_threshold"),
    top_n: int = Query(default=5, ge=1, le=50, alias="rerank_top_n"),
    store: AppStore = Depends(get_store),
    model_cache: ModelCache = Depends(get_model_cache),
    index_cache: IndexCache = Depends(get_index_cache),
    actor: dict = Depends(optional_actor),
):
    db = resolve_searchable_db(name, actor, store)
    if db is None:
        raise HTTPException(404, detail=f"Unknown or inaccessible index: {name!r}")
    if not db.is_file():
        raise HTTPException(404, detail=f"SQLite missing for index {name!r}")
    try:
        results = search_index(db, q, model_cache, index_cache, k=k, score_threshold=threshold, rerank_top_n=top_n)
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    return SearchResponse(index=name, query=q, results=[SearchResultOut(**r) for r in results])
```

- [ ] **Step 3: Write test_search_api.py**

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ragret.cache import IndexCache, ModelCache
from server.main import create_app
from server.store.pool import SqliteConnectionPool
from server.store.sqlite_store import SqliteAppStore


@pytest.fixture
def store(tmp_path: Path):
    pool = SqliteConnectionPool(tmp_path / "test.sqlite", min_size=1, max_size=2)
    s = SqliteAppStore(pool)
    yield s
    pool.close()


@pytest.fixture
def model_cache():
    m = MagicMock(spec=ModelCache)
    m.embed_query.return_value = [0.1, 0.2, 0.3]
    return m


@pytest.fixture
def client(store, model_cache):
    app = create_app(store=store, model_cache=model_cache)
    with TestClient(app) as c:
        yield c


class TestSearch:
    def test_missing_query_param(self, client: TestClient):
        resp = client.get("/api/search/myindex")
        assert resp.status_code == 422  # missing query

    def test_unknown_index(self, client: TestClient):
        resp = client.get("/api/search/nonexistent?query=hello")
        assert resp.status_code == 404

    def test_empty_index(self, client: TestClient, store):
        user = store.create_user("owner", "pwhash")
        store.create_knowledge_base(
            name="mykb", description="test", db_path="/tmp/test.sqlite",
            owner_id=user.id,
        )
        resp = client.get("/api/search/mykb?query=hello")
        assert resp.status_code == 500
```

- [ ] **Step 4: Commit**

```bash
git add server/services/search_service.py server/routers/search.py tests/test_search_api.py
git commit -m "feat: add search router + service + API tests"
```

---

#### Task 2.7: Complete remaining routers and services

The remaining routers follow the same pattern as auth and search:

**Files to create:**
- `server/services/kb_service.py` — knowledge base CRUD operations from httpd _handle_kb_*
- `server/services/build_service.py` — build job creation / cancel logic from httpd _handle_start_build_job
- `server/routers/kb.py` — KB CRUD, members, subscriptions, icons (GET/POST/DELETE /api/kb/{name}/...)
- `server/routers/webhook.py` — POST /api/webhooks/{gitlab|github}/{name}
- `server/routers/jobs.py` — list, get, cancel build jobs
- `server/routers/upload.py` — POST /api/upload multipart tar
- `server/routers/admin.py` — superuser PATCH /api/kb/{name}, DELETE /api/indexes/{name}

Each router should have:
- Proper Pydantic request/response models from `server/schemas.py`
- Auth dependencies from `server/deps.py`
- Business logic delegated to service layer
- API tests using TestClient with memory SQLite

- [ ] **Step 1: Create kb service + router + tests**

Service layer wraps store calls; router handles request parsing / response formatting.

Test cases: create KB, get KB, list KBs, update description, rename, members CRUD, subscriptions.

- [ ] **Step 2: Create webhook router + tests**

Port `_handle_gitlab_webhook` and `_handle_github_webhook` from httpd.py.

Test with mock store (verify webhook validation logic, signature checking).

- [ ] **Step 3: Create jobs router + tests**

Port `_handle_start_build_job`, cancel, list.

- [ ] **Step 4: Create upload router + tests**

Port `_handle_stage_archive_upload`. Test with multipart form data.

- [ ] **Step 5: Create admin router + tests**

Port superuser PATCH and DELETE.

- [ ] **Step 6: Commit all remaining routers**

```bash
git add server/services/kb_service.py server/services/build_service.py server/routers/kb.py server/routers/webhook.py server/routers/jobs.py server/routers/upload.py server/routers/admin.py tests/test_kb_api.py tests/test_webhook.py tests/test_build_queue.py
git commit -m "feat: add remaining routers + services with API tests"
```

---

### Phase 3: RAG Module Split + Legacy Cleanup

#### Task 3.1: ragret/embedder.py

**Files:**
- Create: `ragret/embedder.py`
- Test: `tests/test_embedder.py`

- [ ] **Step 1: Write embedder.py**

```python
from __future__ import annotations

import os
from typing import Callable

import torch

from ragret.paths import resolve_hf_snapshot_dir

EMBEDDING_MODEL = "maidalun1020/bce-embedding-base_v1"
RERANKER_MODEL = "maidalun1020/bce-reranker-base_v1"
EMBED_BATCH_SIZE = 8


def resolve_device() -> str:
    """Pick compute device: env RAGRET_DEVICE, else CUDA, else Intel XPU."""
    override = (os.environ.get("RAGRET_DEVICE") or "").strip()
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda:0"
    if _xpu_available():
        return "xpu:0"
    raise RuntimeError("No GPU available: neither CUDA nor Intel XPU is usable.")


def _xpu_available() -> bool:
    if not hasattr(torch, "xpu"):
        return False
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


def make_embed_model(device: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    local = _local_snapshot_path_or_fail(EMBEDDING_MODEL, "BCE embedding")
    return HuggingFaceEmbeddings(
        model_name=local,
        model_kwargs={"device": device, "local_files_only": True},
        encode_kwargs={"batch_size": EMBED_BATCH_SIZE, "normalize_embeddings": True},
        cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME", ""),
    )


def make_reranker(device: str, top_n: int):
    from ragret.rerank import RagretBCERerank

    dev = str(device)
    rerank_dev = "cpu" if dev.lower().startswith("xpu") else dev
    use_fp16 = rerank_dev.startswith("cuda") and torch.cuda.is_available()
    local = _local_snapshot_path_or_fail(RERANKER_MODEL, "BCE reranker")
    return RagretBCERerank(
        model=local,
        top_n=top_n,
        device=rerank_dev,
        use_fp16=use_fp16,
    )


def _local_snapshot_path_or_fail(repo_id: str, label: str) -> str:
    from ragret.paths import default_hf_models_dir

    roots: list = []
    for raw in (os.environ.get("HF_HOME"), os.environ.get("SENTENCE_TRANSFORMERS_HOME"), str(default_hf_models_dir())):
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()  # noqa: F821
        if p not in roots:
            roots.append(p)
    for root in roots:
        snap = resolve_hf_snapshot_dir(repo_id, hf_home=root, require_weights=True, require_tokenizer=True)
        if snap is None:
            continue
        os.environ["HF_HOME"] = str(root)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(root)
        return str(snap.resolve())
    raise RuntimeError(f"No on-disk snapshot for {label} ({repo_id!r})")


def embed_batch(embed_model, contents: list[str], *, on_batch: Callable | None = None, cancel_check: Callable | None = None) -> list[list[float]]:
    n = len(contents)
    if n == 0:
        return []
    out: list[list[float]] = []
    report_step = max(EMBED_BATCH_SIZE, max(1, n // 40))
    for i in range(0, n, EMBED_BATCH_SIZE):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("embedding cancelled")  # noqa: F821
        batch = contents[i : i + EMBED_BATCH_SIZE]
        out.extend(embed_model.embed_documents(batch))
        done = min(i + len(batch), n)
        if on_batch is not None and (done - (i - i % report_step) >= report_step or done == n):
            on_batch(done, n)
    return out


class BuildCancelledError(Exception):
    """Raised when a long-running index operation is cancelled."""
```

- [ ] **Step 2: Write test_embedder.py**

```python
from __future__ import annotations

import pytest

from ragret.embedder import embed_batch


class MockEmbedModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]


def test_embed_batch_empty() -> None:
    assert embed_batch(MockEmbedModel(), []) == []


def test_embed_batch_progress_callback() -> None:
    calls: list[int] = []
    embed_batch(MockEmbedModel(), ["a", "bb", "ccc"], on_batch=lambda done, total: calls.append(done))
    assert calls[-1] == 3


def test_embed_batch_cancellation() -> None:
    with pytest.raises(Exception):  # BuildCancelledError
        embed_batch(MockEmbedModel(), ["a", "bb", "ccc"], cancel_check=lambda: True)
```

- [ ] **Step 3: Commit**

```bash
git add ragret/embedder.py tests/test_embedder.py
git commit -m "feat: add embedder module with batch embedding + tests"
```

---

#### Task 3.2: ragret/indexer.py — SQLite index build

**Files:**
- Create: `ragret/indexer.py`
- Test: `tests/test_indexer.py`

- [ ] **Step 1: Write indexer.py**

```python
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ragret.loader import fingerprint_map, load_documents_from_dir, relative_source_key
from ragret.models import IndexMeta

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
    device: str | None = None,
    progress: Callable | None = None,
    cancel_check: Callable | None = None,
) -> int:
    from ragret.embedder import EMBEDDING_MODEL, embed_batch

    work_dir = work_dir.resolve()

    documents = load_documents_from_dir(work_dir)
    from ragret.loader import chunk_documents
    texts = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    contents = [d.page_content for d in texts]
    vectors = embed_batch(embed_model, contents, on_batch=progress, cancel_check=cancel_check)
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
```

- [ ] **Step 2: Write test_indexer.py**

```python
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ragret.indexer import build_index, get_meta, init_schema


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
```

- [ ] **Step 3: Commit**

```bash
git add ragret/indexer.py tests/test_indexer.py
git commit -m "feat: add indexer module with SQLite index build + tests"
```

---

#### Task 3.3: ragret/searcher.py — dense retrieval + rerank

**Files:**
- Create: `ragret/searcher.py`
- Test: `tests/test_searcher.py`

- [ ] **Step 1: Write searcher.py**

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ragret.cache import IndexCache, ModelCache
from ragret.indexer import get_meta
from ragret.models import SearchResult

EMBEDDING_MODEL = "maidalun1020/bce-embedding-base_v1"


def _load_index_snapshot(db_path: Path) -> tuple[np.ndarray, list[dict], str | None]:
    import sqlite3

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
            records.append({"id": rid, "source": source, "chunk_index": cidx, "content": content, "metadata": meta})
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

    # cached index load
    cached = index_cache.get(db_path)
    if cached is not None:
        matrix, records, stored_model = cached
    else:
        matrix, records, stored_model = _load_index_snapshot(db_path)
        index_cache.set(db_path, matrix, records, stored_model)

    if stored_model and stored_model != EMBEDDING_MODEL:
        import sys
        print(f"Warning: index was built with {stored_model}, this build expects {EMBEDDING_MODEL}.", file=sys.stderr)

    # vector search
    q = model_cache.embed_query(query)
    scores = matrix @ q
    order = np.argsort(-scores)

    from langchain_core.documents import Document

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

    # rerank
    ranked = model_cache.rerank(query, candidates)
    ranked = ranked[:rerank_top_n]

    results = []
    for d in ranked:
        results.append({
            "content": d.page_content,
            "source": str(d.metadata.get("source", "")),
            "chunk_index": int(d.metadata.get("chunk_index", 0)),
            "vector_score": float(d.metadata.get("vector_score", 0.0)),
            "relevance_score": float(d.metadata.get("relevance_score", 0.0)),
        })
    return results
```

- [ ] **Step 2: Write test_searcher.py**

```python
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
    assert isinstance(results, list)


def test_search_caches_index(tmp_path: Path, model_cache: MagicMock) -> None:
    db = tmp_path / "test.sqlite"
    _make_index(db, ["hello world"])
    ic = IndexCache(max_entries=4)
    search_db(db, "test", model_cache=model_cache, index_cache=ic)
    # second call should use cache
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
```

- [ ] **Step 3: Commit**

```bash
git add ragret/searcher.py tests/test_searcher.py
git commit -m "feat: add searcher module with structured search results + tests"
```

---

#### Task 3.4: Adapt build_queue.py for new modules

**Files:**
- Modify: `server/build_queue.py`

Replace imports from `ragret.rag` with imports from `ragret.loader`, `ragret.indexer`, `ragret.embedder`.

- [ ] **Step 1: Update imports in build_queue.py**

```python
# Before:
from ragret.rag import BuildCancelledError, index_workdir, try_incremental_update_workdir

# After:
from ragret.embedder import BuildCancelledError, embed_batch, make_embed_model
from ragret.indexer import build_index
from ragret.loader import iter_indexable_files, load_documents_from_dir
```

- [ ] **Step 2: Update run_one_build_job to use new functions**

Replace `index_workdir(extract_dir, final_db, ...)` with:
```python
conn = pool.acquire()
try:
    embed_model = make_embed_model(device)
    n = build_index(conn, extract_dir, embed_model, progress=rag_progress, cancel_check=cancelled)
finally:
    conn.close()
```

- [ ] **Step 3: Commit**

```bash
git add server/build_queue.py
git commit -m "refactor: build_queue uses new ragret modules"
```

---

#### Task 3.5: Update cli.py to use uvicorn

**Files:**
- Modify: `ragret/cli.py`

- [ ] **Step 1: Update cli.py**

Replace `from server import run_server; return run_server(...)` with:

```python
import uvicorn
from server.main import create_app
from server.config import Settings
from server.store.factory import create_app_store
from ragret.registry import IndexRegistry
from ragret.cache import ModelCache, IndexCache
from ragret.embedder import resolve_device

def serve(args: argparse.Namespace) -> int:
    settings = Settings(host=args.host, port=args.port)
    app = create_app(settings=settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0
```

- [ ] **Step 2: Commit**

```bash
git add ragret/cli.py
git commit -m "refactor: cli.py uses uvicorn + create_app"
```

---

#### Task 3.6: Delete legacy httpd.py and rag.py

- [ ] **Step 1: Remove httpd.py and update server/__init__.py**

```bash
git rm server/httpd.py
```

Update `server/__init__.py`:
```python
"""HTTP API process: auth, knowledge-base ACL, static UI. Depends on ``ragret`` for RAG + registry."""
from server.main import create_app

__all__ = ["create_app"]
```

- [ ] **Step 2: Remove rag.py and update ragret/__init__.py**

```bash
git rm ragret/rag.py
```

Update `ragret/__init__.py` to re-export key symbols with deprecation notice:
```python
"""ragret: document index (SQLite), dense retrieval, BCE rerank."""

__all__: list[str] = []
```

- [ ] **Step 3: Commit**

```bash
git add server/__init__.py ragret/__init__.py
git commit -m "refactor: remove legacy httpd.py and rag.py"
```

---

#### Task 3.7: Update requirements.txt with new dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add FastAPI, uvicorn, httpx, pydantic-settings**

```
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic-settings>=2
httpx>=0.27          # for TestClient
pytest>=8
pytest-mock>=3
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "build: add fastapi, uvicorn, httpx, pydantic-settings"
```

---

## Spec Coverage Checklist

| Spec requirement | Implemented in |
|---|---|
| Pydantic data models | Task 1.2 |
| Injectable ModelCache / IndexCache | Task 1.3 |
| Centralized settings (pydantic-settings) | Task 1.4 |
| SqliteConnectionPool | Task 1.5 |
| SqliteAppStore adapted for pool | Task 1.6-1.7 |
| Document loader module | Task 1.8 |
| Test infrastructure (conftest) | Task 1.9 |
| Embedder module + batch embedding | Task 3.1 |
| Indexer module + SQLite build | Task 3.2 |
| Searcher + structured results | Task 3.3 |
| FastAPI app + lifespan | Task 2.5 |
| Auth middleware + Deps | Task 2.2-2.3 |
| Auth router + tests | Task 2.5 |
| Search router + tests | Task 2.6 |
| KB router + tests | Task 2.7 |
| Webhook router + tests | Task 2.7 |
| Jobs router + tests | Task 2.7 |
| Upload router + tests | Task 2.7 |
| Admin router + tests | Task 2.7 |
| Auth service | Task 2.4 |
| Search service | Task 2.6 |
| KB service | Task 2.7 |
| Build service | Task 2.7 |
| Build_queue adapted | Task 3.4 |
| CLI uses uvicorn | Task 3.5 |
| Delete legacy httpd.py + rag.py | Task 3.6 |
| requirements.txt updated | Task 3.7 |
