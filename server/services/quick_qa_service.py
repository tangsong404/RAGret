from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

from ragret.cache import IndexCache, ModelCache
from ragret.quick_qa_agent import quick_qa_llm_configured, run_quick_qa
from ragret.registry import IndexRegistry
from server.services import search_service
from server.store.protocol import AppStore


def _normalize_messages(raw: Any, *, max_items: int = 24) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content})
    if len(out) > max_items:
        out = out[-max_items:]
    return out


def _allowed_kb_names(store: AppStore, uid: int) -> list[str]:
    rows = store.list_owned_and_subscribed_knowledge_bases_for_user(uid)
    return [str(r.name) for r in rows]


def _make_search_fn(
    *,
    store: AppStore,
    registry: IndexRegistry,
    model_cache: ModelCache,
    index_cache: IndexCache,
    allowed_kbs: list[str],
    actor: dict[str, Any],
    public_host: str | None = None,
) -> Callable[[str, str], str]:
    def search_in_kb(kb_name: str, query: str) -> str:
        kb = str(kb_name or "").strip()
        if kb not in allowed_kbs:
            return f"知识库 {kb} 不存在或无权限访问。"
        db_path = search_service.resolve_searchable_db(kb, actor, store, registry)
        if db_path is None or not db_path.is_file():
            return f"知识库 {kb} 无可用索引。"
        results = search_service.search_index(
            db_path,
            str(query or ""),
            model_cache,
            index_cache,
            k=8,
            score_threshold=0.3,
            rerank_top_n=5,
            kb_name=kb,
            public_host=public_host,
        )
        return search_service.format_search_text(str(query or ""), results)

    return search_in_kb


def run_direct_index_answer(
    question: str,
    *,
    store: AppStore,
    registry: IndexRegistry,
    model_cache: ModelCache,
    index_cache: IndexCache,
    uid: int,
    actor: dict[str, Any],
    max_kbs: int = 3,
    public_host: str | None = None,
) -> dict[str, Any]:
    allowed = _allowed_kb_names(store, uid)
    search_in_kb = _make_search_fn(
        store=store,
        registry=registry,
        model_cache=model_cache,
        index_cache=index_cache,
        allowed_kbs=allowed,
        actor=actor,
        public_host=public_host,
    )
    blocks: list[str] = []
    for kb_name in allowed:
        try:
            rtxt = search_in_kb(kb_name, question)
        except Exception:
            continue
        txt = str(rtxt or "").strip()
        if not txt:
            continue
        blocks.append(f"[{kb_name}]\n{txt}")
        if len(blocks) >= max_kbs:
            break
    if blocks:
        answer = "\n\n---\n\n".join(blocks)
    else:
        answer = "未配置完整 LLM 参数，已切换为索引直返模式；当前未检索到匹配内容。"
    return {
        "ok": True,
        "answer": answer,
        "used_tool": False,
        "tool_name": "",
        "mode": "direct_index",
    }


def run_quick_qa_request(
    *,
    question: str,
    store: AppStore,
    registry: IndexRegistry,
    model_cache: ModelCache,
    index_cache: IndexCache,
    uid: int,
    actor: dict[str, Any],
    messages: list[dict[str, str]] | None = None,
    lang: str = "zh",
    on_tool_event: Callable[[str], None] | None = None,
    public_host: str | None = None,
) -> dict[str, Any]:
    allowed = _allowed_kb_names(store, uid)
    search_in_kb = _make_search_fn(
        store=store,
        registry=registry,
        model_cache=model_cache,
        index_cache=index_cache,
        allowed_kbs=allowed,
        actor=actor,
        public_host=public_host,
    )

    if not quick_qa_llm_configured():
        return run_direct_index_answer(
            question,
            store=store,
            registry=registry,
            model_cache=model_cache,
            index_cache=index_cache,
            uid=uid,
            actor=actor,
            public_host=public_host,
        )

    result = run_quick_qa(
        question,
        list_available_kbs=lambda: list(allowed),
        search_in_kb=search_in_kb,
        messages=messages,
        on_tool_event=on_tool_event,
        lang=lang,
    )
    return {
        "ok": True,
        "answer": str(result.get("answer") or ""),
        "used_tool": bool(result.get("used_tool")),
        "tool_name": str(result.get("tool_name") or ""),
        "tool_events": list(result.get("tool_events") or []),
        "mode": "llm",
    }


def stream_quick_qa_events(
    *,
    question: str,
    store: AppStore,
    registry: IndexRegistry,
    model_cache: ModelCache,
    index_cache: IndexCache,
    uid: int,
    actor: dict[str, Any],
    messages: list[dict[str, str]] | None = None,
    lang: str = "zh",
    public_host: str | None = None,
) -> Iterator[bytes]:
    out_q: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def on_tool_event(msg: str) -> None:
        out_q.put({"type": "tool_event", "text": str(msg or "")})

    def worker() -> None:
        try:
            payload = run_quick_qa_request(
                question=question,
                store=store,
                registry=registry,
                model_cache=model_cache,
                index_cache=index_cache,
                uid=uid,
                actor=actor,
                messages=messages,
                lang=lang,
                on_tool_event=on_tool_event,
                public_host=public_host,
            )
            out_q.put({"type": "final", **payload})
        except Exception as e:
            out_q.put({"type": "error", "ok": False, "error": str(e)})
        finally:
            out_q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = out_q.get()
        if item is None:
            break
        yield (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")
