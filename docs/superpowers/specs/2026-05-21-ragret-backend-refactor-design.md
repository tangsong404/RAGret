# RAGret Backend Refactoring Design

**Date**: 2026-05-21
**Status**: Approved design
**Priority order**: Testability > Maintainability > Performance

## Architecture Overview

Replace the monolithic `http.server.ThreadingHTTPServer` + inline-if/else routing
with **FastAPI + uvicorn**. The core RAG library (`ragret/` package) is split into
single-responsibility modules. All global mutable state (model caches, index caches)
is encapsulated in injectable classes.

### Before vs After

```
Before:                                 After:
ragret/                                 ragret/
  rag.py          ~850 lines (all)        loader.py       document loading + chunking
  registry.py                            embedder.py     embedding model wrapper
  paths.py                               indexer.py      SQLite index build + incremental
  compat.py                              searcher.py     dense retrieval + rerank (structured)
  bce_...patch.py                        cache.py        injectable ModelCache / IndexCache
  rerank.py                              models.py       Pydantic data models
  cli.py                                 registry.py     (unchanged)
                                         paths.py        (unchanged)
server/                                 compat.py       (unchanged)
  httpd.py        ~2200 lines (all)     bce_...patch.py (unchanged)
  build_queue.py                         rerank.py       (unchanged)
  archive_util.py                        cli.py          (simplified)
  passwords.py
  runtime_paths.py                      server/
  data_cleanup.py                        main.py         FastAPI app + lifespan
  store/                                 config.py       pydantic-settings
    protocol.py                          deps.py         Depends() providers
    factory.py                           schemas.py      Pydantic request/response schemas
    sqlite_store.py                      routers/
  (frontend/ unchanged)                    __init__.py
                                          auth.py
tests/          (none)                    kb.py
                                          search.py
                                         webhook.py
                                          jobs.py
                                          upload.py
                                          admin.py
                                         services/
                                           auth_service.py
                                           kb_service.py
                                           search_service.py
                                           build_service.py
                                         middleware/
                                           auth.py
                                         store/
                                           pool.py         SqliteConnectionPool
                                           protocol.py    (unchanged)
                                           factory.py     (unchanged)
                                           sqlite_store.py (adapted for pool)
                                         archive_util.py  (unchanged)
                                         build_queue.py   (adapted for pool)
                                         passwords.py     (unchanged)
                                         runtime_paths.py (unchanged)
                                        tests/
                                          conftest.py
                                          test_loader.py
                                          test_indexer.py
                                          test_searcher.py
                                          test_embedder.py
                                          test_auth_api.py
                                          test_search_api.py
                                          test_kb_api.py
                                          test_webhook.py
                                          test_build_queue.py
```

## Detailed Design

### 1. Core Modules (`ragret/` package)

#### `ragret/models.py` — Pydantic data models
```
Chunk(id, source, chunk_index, content, metadata, embedding: list[float] | None)
SearchResult(content, source, chunk_index, vector_score, relevance_score, metadata)
IndexMeta(embedding_model, embed_dim, chunk_size, chunk_overlap, indexed_at, source_fingerprints, schema_version)
```

#### `ragret/loader.py` — Document loading + chunking
Extracted from `rag.py` lines 112-182. Functions:
- `load_document(path: Path) -> list[Document]`
- `load_documents_from_dir(work_dir: Path) -> list[Document]`
- `chunk_documents(docs: list[Document], chunk_size, chunk_overlap) -> list[Document]`
- `fingerprint_map(work_dir: Path) -> dict[str, str]`
- `relative_source_key(work_dir, source_raw) -> str`

Testable without GPU: file I/O + langchain splitter only.

#### `ragret/embedder.py` — Embedding model wrapper
Extracted from `rag.py` lines 269-303. Functions:
- `make_embed_model(device: str) -> HuggingFaceEmbeddings`
- `make_reranker(device: str, top_n: int) -> RagretBCERerank`
- `resolve_device() -> str`
- `embed_batch(contents, *, on_batch, cancel_check) -> list[list[float]]`

#### `ragret/indexer.py` — SQLite index build + incremental update
Extracted from `rag.py` lines 185-663. Functions:
- `connect(db_path) -> sqlite3.Connection`   (pool-aware)
- `init_schema(conn)` and `clear_chunks(conn)`  (stateless helper)
- `set_meta / get_meta`  (stateless helper)
- `index_workdir(...)` → takes a connection instead of opening its own
- `try_incremental_update_workdir(...)` → same

#### `ragret/searcher.py` — Dense retrieval + rerank (returns structured data)
Extracted from `rag.py` lines 666-852. Key changes:
- `search_db(...)` returns `list[SearchResult]` instead of `str`
- Takes `ModelCache` and `IndexCache` as parameters (injectable)
- No global module-level state
- `_resolve_search_index()` → moved into `IndexCache` class

#### `ragret/cache.py` — Injectable caches replacing global module state
```
ModelCache(device: str, rerank_top_n: int = 256)
  - get_embed_model() -> HuggingFaceEmbeddings
  - get_rerank_model() -> RagretBCERerank
  - embed_query(text: str) -> np.ndarray
  - rerank(query: str, candidates: list[Document]) -> list[Document]

IndexCache(max_entries: int = 64)
  - get_or_load(db_path: Path) -> tuple[np.ndarray, list[dict], str | None]
  - invalidate(db_path: Path)
```

### 2. Server Package — FastAPI Migration

#### `server/config.py` — Centralized configuration
```python
class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8765
    registry_path: Path | None = None
    app_db_path: Path | None = None
    session_ttl: int = 30 * 24 * 3600
    search_index_cache_max: int = 64
    search_rerank_cache_top: int = 256
    git_http_connect_timeout: float = 20.0
    git_http_read_timeout: float = 30.0
    git_clone_wall_timeout: float = 30.0
    api_token: str | None = None
    avatar_max_bytes: int = 2 * 1024 * 1024
    public_host: str | None = None
```

#### `server/main.py` — FastAPI app + lifespan
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    pool = SqliteConnectionPool(settings.app_db_path or default_path)
    store = SqliteAppStore(pool)
    cache = ModelCache(device=resolve_device())
    registry = IndexRegistry(settings.registry_path or default_registry)
    registry.load()
    cleanup_orphan_kb_sqlite_files(...)
    stop = threading.Event()
    worker = start_build_worker(store=store, registry=registry, ..., stop_event=stop)
    wake_build_worker()
    yield
    stop.set()
    pool.close()
    store.close()
```

#### `server/deps.py` — Dependency injection
```python
def get_settings(request: Request) -> Settings
def get_pool(request: Request) -> SqliteConnectionPool
def get_store(request: Request) -> AppStore
def get_cache(request: Request) -> ModelCache
def get_current_user(..., store = Depends(get_store)) -> UserRecord | None
def require_user(...) -> UserRecord
```

#### `server/middleware/auth.py` — Bearer token parser
Converts `Authorization: Bearer <token>` into session user / API key user / superuser.
Stores result in `request.state.actor`.

#### `server/routers/` — Route modules
Each router is a standard `APIRouter`. Examples:

```python
# server/routers/search.py
router = APIRouter(prefix="/api")

@router.get("/search/{name}")
def search_index(
    name: str,
    q: str = Query(alias="query"),
    k: int = 10,
    threshold: float = 0.3,
    top_n: int = 5,
    format: str = "json",
    store = Depends(get_store),
    cache = Depends(get_cache),
    actor = Depends(require_actor_or_anon),
):
    db_path = resolve_searchable_db(name, actor, store)
    results = search_db(db_path, q, model_cache=cache, k=k, ...)
    if format == "text":
        return PlainTextResponse(render_search_text(results))
    return {"ok": True, "results": [r.model_dump() for r in results]}
```

`server/routers/auth.py`: register, login, logout, me, password change
`server/routers/kb.py`: CRUD, members, subscriptions, icons, PATCH (superuser)
`server/routers/webhook.py`: POST /api/webhooks/{gitlab|github}/{name}
`server/routers/jobs.py`: list, get, cancel
`server/routers/upload.py`: POST /api/upload (multipart tar staging)
`server/routers/admin.py`: DELETE /api/indexes/{name}, superuser PATCH

#### `server/services/` — Business logic layer
- **AuthService**: register, login, logout, session mgmt (from httpd _handle_auth_*)
- **KBService**: CRUD, members, subscriptions, icons, webhook config (from httpd _handle_kb_*)
- **SearchService**: permission-gated search dispatch (from httpd _handle_search + _db_for_search)
- **BuildService**: job creation, cancel, webhook-pull (from httpd _handle_start_build_job, _handle_kb_webhook_pull)

### 3. Store Layer — Connection Pool

#### `server/store/pool.py`
```
SqliteConnectionPool(db_path, min_size=4, max_size=32)
  - acquire() -> contextmanager yielding sqlite3.Connection
  - close()
```

#### `SqliteAppStore` adapted:
- Constructor takes `SqliteConnectionPool` instead of `Path`
- Methods gain `conn: sqlite3.Connection` parameter
- The pool is managed by FastAPI lifespan
- Thread safety: each thread gets its own connection from pool

### 4. Test Strategy

#### `tests/conftest.py` fixtures
```python
@pytest.fixture
def pool():
    p = SqliteConnectionPool(":memory:", min_size=1, max_size=1)
    yield p
    p.close()

@pytest.fixture
def store(pool):
    return SqliteAppStore(pool)

@pytest.fixture
def cache():
    return MagicMock(spec=ModelCache)

@pytest.fixture
def client(store, cache):
    app = create_app(store=store, cache=cache)
    with TestClient(app) as c:
        yield c
```

#### Test coverage target per module:
| Module | Type | Coverage target |
|--------|------|----------------|
| loader.py | unit | 90%+ |
| indexer.py | unit | 85%+ |
| searcher.py | unit | 90%+ |
| embedder.py | unit (mock torch) | 80%+ |
| SqliteAppStore | unit (:memory:) | 90%+ |
| API routers | integration (TestClient) | 80%+ |
| build_queue.py | integration | 70%+ |

### 5. Migration Phasing

**Phase 1 — Foundation**:
1. Create `ragret/models.py` — Pydantic models
2. Create `ragret/loader.py` — extract from rag.py
3. Create `ragret/cache.py` — ModelCache + IndexCache
4. Create `server/config.py` — Settings
5. Create `server/store/pool.py` — SqliteConnectionPool
6. Adapt `SqliteAppStore` for pool
7. Create `tests/conftest.py` + base fixtures
8. Write tests for loader, store, cache

**Phase 2 — FastAPI app + routers**:
1. Create `server/main.py` — FastAPI app + lifespan
2. Create `server/deps.py` — dependency injection
3. Create `server/middleware/auth.py`
4. Create all `routers/` one by one, each with API tests
5. Run FastAPI on a separate port alongside old server

**Phase 3 — RAG module split**:
1. Create `ragret/embedder.py`, `ragret/indexer.py`, `ragret/searcher.py`
2. Adapt `build_queue.py` to use new modules
3. Adapt `routers/search.py` to use new `search_db` returning structured data
4. Write tests for each new module
5. Once all endpoints migrated: delete `httpd.py`, delete `rag.py` (keep backward import shims for one release)

### 6. Performance Considerations (Priority 3)

- Connection pool replaces `threading.Lock` serialization → true concurrent reads
- `IndexCache` (LRU, max 64 entries) keeps hot indexes in memory, same as current
- Embedding/rerank model caches remain per-device singleton (via `ModelCache`)
- Search returns structured data → API responses no longer include string formatting overhead
- No async for embedding calls (torch is synchronous and GIL-bound anyway)

## Self-Review Notes

- No TODOs/placeholders in this spec
- Architecture matches feature descriptions
- Scope focused on backend refactoring only; no frontend changes
- Each requirement is explicit (pool min/max sizes, test coverage targets, module boundaries)
- Migration phasing ensures no "big bang" cutover
