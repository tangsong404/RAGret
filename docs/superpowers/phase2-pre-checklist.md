# Phase 2 前 Checklist（开发者）

在进入 **Phase 2（FastAPI + routers）** 之前完成下列项。假设 Phase 1 代码已在工作区就绪（1.5–1.9 等），且设计见 [spec](specs/2026-05-21-ragret-backend-refactor-design.md)、[plan](plans/2026-05-21-ragret-backend-refactor-plan.md)。

---

## 1. 提交 Phase 1 剩余 commit（4 或 5 个）

按依赖顺序提交（`factory.py` 无需改动）：

| 顺序 | 建议 message | 文件 |
|------|----------------|------|
| 1 | `feat: add SqliteConnectionPool` | `server/store/pool.py` |
| 2 | `refactor: SqliteAppStore uses SqliteConnectionPool` | `server/store/sqlite_store.py` |
| 3 | `feat: add document loader module with tests` | `ragret/loader.py`, `tests/test_loader.py` |
| 4 | `test: add base fixtures for pool, store, model_cache` | `tests/conftest.py`, `tests/__init__.py`, `tests/test_store_pool_smoke.py` |
| 5（可选） | `docs: add backend refactoring implementation plan` | `docs/superpowers/plans/2026-05-21-ragret-backend-refactor-plan.md` |

提交后 `main` 相对 `origin/main` 应包含 Phase 1 全部已提交模块。

---

## 2. （推荐）修复 `SqliteConnectionPool._created` 初始化

**问题：** `__init__` 预创建 `min_size` 条连接时，`_created` 仍为 `0`，导致 `acquire()` 在池空时可能超额创建连接（超过 `max_size` 的本意）。

**修复：** 预创建后设置 `self._created = min_size`（或在循环内递增）。见 `server/store/pool.py` 当前实现。

**验证：** 单元/冒烟测试中多次 `acquire` 并发，连接总数不超过 `max_size`。

---

## 3. （推荐）pytest：`pytest-asyncio` 或移除 `asyncio_mode`

**现象：** 未安装 `pytest-asyncio` 时，`pyproject.toml` 中 `asyncio_mode = "auto"` 会触发 `PytestConfigWarning: Unknown config option`.

**二选一：**

- `pip install pytest-asyncio`（bcrag / CI 环境），保留 `asyncio_mode = "auto"`；或
- 从 `pyproject.toml` 删除 `asyncio_mode`（Phase 2 尚无 async 测试时可先删）。

---

## 4. 明确双轨 + 提交 1.6 后的生产 Store 行为

### 4.1 当前 HEAD（仅 1.1–1.4 已提交时）

```
httpd.py → create_app_store() → SqliteAppStore(db_path)
  → 单条 sqlite3 连接 + threading.Lock
```

### 4.2 提交 1.5 + 1.6 之后（**无需改 `httpd.py`**）

```
httpd.py → create_app_store() → SqliteAppStore(db_path)   # 签名不变
  → 内部 SqliteConnectionPool
  → 各方法 with pool.acquire() as conn
```

**结论：** 同一条生产 HTTP 路径会自动使用连接池；这不是「旧 Store vs 新 FastAPI Store」两套实现。

### 4.3 真正的双轨（Phase 2/3 结束前）

| 轨道 | 路径 | 状态 |
|------|------|------|
| 旧 | 路由 / 鉴权 / KB / 构建 → `server/httpd.py` | 生产主路径 |
| 旧 | 建索引 / 搜索 / 模型 → `ragret/rag.py` | 生产 RAG |
| 新 | `loader.py`, `cache.py`, `models.py`；未来 `embedder` / `searcher` | **未接入** httpd / build_queue |

### 4.4 （推荐）手工 smoke：比 loader 单测更贴近生产

在 **bcrag** 环境、**1.6 已提交** 后执行一次短流程（确认池化 Store + httpd 路径）：

```powershell
conda activate bcrag
cd <repo-root>

# 启动服务（默认 127.0.0.1:8765，可用临时 DB 目录）
$env:RAGRET_APP_DB = "$pwd\runtime\data\smoke_app.sqlite"
python -m ragret serve --host 127.0.0.1 --port 8765
```

另开终端：

```powershell
# 注册
curl -s -X POST http://127.0.0.1:8765/api/auth/register `
  -H "Content-Type: application/json" `
  -d '{"username":"smoke_user","password":"secret12345"}'

# 登录拿 token 后 /api/auth/me
# 若有 KB 创建 API：创建一条 KB 或触发与 store 相关的只读接口（列表 KB 等）
```

**通过标准：** 无 500；`smoke_app.sqlite` 可写；重复请求无 SQLite 锁死/明显变慢。

自动化补充（bcrag）：

```powershell
& "$env:USERPROFILE\.conda\envs\bcrag\python.exe" -m pytest tests/test_loader.py tests/test_store_pool_smoke.py -v
```

---

## 5. Phase 2 开始时：Settings 与 legacy 环境变量名兼容

`server/config.py` 使用 `pydantic-settings`，前缀 **`RAGRET_`**。`httpd.py` 仍直接读 `os.environ`，命名不完全一致。

**Phase 2 启动任务：** 在 `Settings` / `deps` / `lifespan` 中统一或做别名映射，避免双份配置。

| Legacy（httpd / 其它） | Settings 字段（建议） | 备注 |
|------------------------|----------------------|------|
| `RAGRET_SESSION_TTL` | `session_ttl` | 已对齐 |
| `RAGRET_AVATAR_MAX_BYTES` | `avatar_max_bytes` | 已对齐 |
| `RAGRET_PUBLIC_HOST` | `public_host` | 已对齐 |
| `RAGRET_REGISTRY` | `registry_path` | httpd 用字符串路径 |
| `RAGRET_API_TOKEN` | `api_token` | 已对齐 |
| `RAGRET_APP_DB` | `app_db_path` | factory 仍可读 env；Settings 宜同步 |
| `RAGRET_APP_STORE` | （无） | 仅 factory，Phase 2 lifespan 需决定 |
| Git 超时等 | `git_http_*_s`, `git_clone_wall_timeout_s` | httpd 内硬编码或散落 env 时需对照 spec |

**建议实现方式（Phase 2 Task 2.1 附近）：**

- `Settings` 增加 `model_config` 的 `Field(validation_alias=...)` 或自定义 `settings_customise_sources`，兼容旧名；或
- `create_app` / lifespan 从 `Settings()` 读入后写回 `os.environ` 供尚未迁移的 httpd 并行期使用（短期）。

**勿在 Phase 3 前：** 在生产搜索路径使用 `ModelCache` → `ragret.embedder`（模块尚未存在）。

---

## 6. 其它风险（确认项）

| 项 | 处理 |
|----|------|
| `pytest` / `conftest` 导入链重 | `conftest` → `ragret.cache` → BCEmbedding；`cgi` 警告来自间接加载 `httpd` |
| `claim_next_queued_build_job` | 单 `acquire` 事务；多 worker 与 `build_queue` 联调放在 **Phase 3** |
| `ModelCache` / 新 searcher | Phase 3 前勿接入 httpd 搜索 |

---

## 7. Checklist 勾选

- [ ] 上述 4（或 5）个 commit 已 push / 合并目标分支
- [ ] `SqliteConnectionPool._created` 已修复并验证
- [ ] `pytest-asyncio` 或已移除 `asyncio_mode`
- [ ] 1.6 提交后完成 httpd 手工 smoke（注册 / 建 KB 或等价 Store 操作）
- [ ] bcrag 下 `pytest tests/test_loader.py tests/test_store_pool_smoke.py` 通过
- [ ] Phase 2 首任务已登记：Settings ↔ legacy `RAGRET_*` 兼容表

全部完成后即可开始 **Phase 2 Task 2.1**（`server/schemas.py`）。
