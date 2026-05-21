"""Argparse entry for ``serve``."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_hf_env() -> None:
    if "HF_HOME" not in os.environ:
        from ragret.paths import default_hf_models_dir

        d = default_hf_models_dir()
        os.environ["HF_HOME"] = str(d)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(d)
        d.mkdir(parents=True, exist_ok=True)


def serve(args: argparse.Namespace) -> int:
    import uvicorn

    from ragret.cache import IndexCache, ModelCache
    from ragret.embedder import resolve_device
    from server.build_queue import start_global_build_worker, wake_build_worker
    from server.config import Settings
    from server.data_cleanup import cleanup_orphan_kb_sqlite_files
    from server.main import create_app
    from server.runtime_paths import default_registry_path

    _ensure_hf_env()
    settings = Settings(host=args.host, port=args.port)
    settings.apply_legacy_environ()

    device = resolve_device()
    model_cache = ModelCache(
        device=device,
        rerank_top_n=settings.search_rerank_cache_top,
    )
    index_cache = IndexCache(max_entries=settings.search_index_cache_max)
    app = create_app(
        settings=settings,
        model_cache=model_cache,
        index_cache=index_cache,
        repo_root=REPO_ROOT,
    )

    reg_path = settings.registry_path or default_registry_path(REPO_ROOT)
    registry = app.state.registry
    app_store = app.state.app_store
    upload_base = app.state.upload_base

    n_orphan = cleanup_orphan_kb_sqlite_files(REPO_ROOT, registry=registry, app_store=app_store)
    if n_orphan:
        print(
            f"Removed {n_orphan} orphan KB sqlite file(s) under runtime/data or legacy data/.",
            flush=True,
        )

    _thread, stop = start_global_build_worker(
        root=REPO_ROOT,
        registry=registry,
        app_store=app_store,
        upload_base=upload_base,
    )
    wake_build_worker()

    db_path = getattr(app_store, "_path", None)
    extra = f" app_db={db_path}" if db_path is not None else ""
    print(
        f"ragret server http://{settings.host}:{settings.port}/  registry={reg_path}{extra}",
        flush=True,
    )
    print(
        "API: auth /api/auth/* | GET /api/indexes | GET /api/kb/{name} | "
        "GET /api/search/{index}?query=... | POST/DELETE /api/indexes",
        flush=True,
    )

    try:
        uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    finally:
        stop.set()
    return 0


def main() -> int:
    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

    p = argparse.ArgumentParser(
        prog="ragret",
        description=(
            "RAGret — local RAG index (BCE embedding + rerank) stored in SQLite. "
            "Subcommands: serve (HTTP API)."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser(
        "serve",
        help="HTTP API service (FastAPI + uvicorn)",
    )
    pv.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: loopback)")
    pv.add_argument("--port", type=int, default=8765)
    pv.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry JSON path (default: ./runtime/ragret_registry.json or env RAGRET_REGISTRY)",
    )

    args = p.parse_args()
    if args.registry is not None:
        os.environ["RAGRET_REGISTRY"] = str(args.registry.expanduser().resolve())

    if args.cmd == "serve":
        return serve(args)
    return 1
