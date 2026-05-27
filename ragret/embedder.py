"""Embedding model wrapper: device resolution, batch embedding, reranker factory."""
from __future__ import annotations

import ragret.compat  # noqa: F401 — multiprocess patch before torch / langchain

import os
import threading
from pathlib import Path
from typing import Any, Callable

import torch

try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
except ImportError:
    pass

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from ragret.paths import default_hf_models_dir, resolve_hf_snapshot_dir
from ragret.rerank import RagretBCERerank

EMBEDDING_MODEL = "maidalun1020/bce-embedding-base_v1"
RERANKER_MODEL = "maidalun1020/bce-reranker-base_v1"
EMBED_BATCH_SIZE = 8


def _ensure_hf_cache_env() -> None:
    """Single cache root; force offline Hub for index/search."""
    default_root = default_hf_models_dir()
    raw_hf = os.environ.get("HF_HOME")
    raw_st = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    if raw_hf:
        root = Path(raw_hf).expanduser().resolve()
    elif raw_st:
        root = Path(raw_st).expanduser().resolve()
    else:
        root = default_root
    s = str(root)
    os.environ["HF_HOME"] = s
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = s
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


_ensure_hf_cache_env()


def _xpu_available() -> bool:
    if not hasattr(torch, "xpu"):
        return False
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


def resolve_device() -> str:
    """Pick compute device: env RAGRET_DEVICE, else CUDA, else Intel XPU. CPU is not supported."""
    override = (os.environ.get("RAGRET_DEVICE") or "").strip()
    if override:
        if override.lower() == "cpu":
            raise RuntimeError(
                "ragret does not support a CPU backend; use an NVIDIA GPU (CUDA) or "
                "Intel GPU (torch.xpu). See README (Dockerfile / Dockerfile.xpu).",
            )
        return override
    if torch.cuda.is_available():
        return "cuda:0"
    if _xpu_available():
        return "xpu:0"
    raise RuntimeError(
        "No GPU available: neither CUDA nor Intel XPU is usable. Use the NVIDIA "
        "Dockerfile with --gpus all, or Dockerfile.xpu with Intel device passthrough, "
        "or install CUDA or PyTorch-with-XPU locally (see README).",
    )


class BuildCancelledError(Exception):
    """Raised when a long-running index operation is cancelled (e.g. user abort)."""


def _hf_weights_hint() -> str:
    return (
        "BCE weights are missing on disk. "
        f"HF_HOME (where ragret looks): {os.environ.get('HF_HOME', '')}. "
        "From the RAGret repo root with network run: python warmup_hf_models.py "
        "(or set HF_HOME to the directory that already contains the Hub cache)."
    )


def _looks_like_hf_cache_miss(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "couldn't connect",
            "could not connect",
            "cached files",
            "local_files_only",
            "find them in the cached",
        )
    )


def reraise_if_missing_hf_weights(exc: BaseException) -> None:
    if _looks_like_hf_cache_miss(exc):
        raise RuntimeError(
            f"{_hf_weights_hint()}\n\nOriginal: {type(exc).__name__}: {exc}",
        ) from exc
    raise exc


def _local_snapshot_path_or_fail(repo_id: str, label: str) -> str:
    roots: list[Path] = []
    for raw in (
        os.environ.get("HF_HOME"),
        os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
        str(default_hf_models_dir()),
    ):
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if p not in roots:
            roots.append(p)

    for root in roots:
        snap = resolve_hf_snapshot_dir(
            repo_id,
            hf_home=root,
            require_weights=True,
            require_tokenizer=True,
        )
        if snap is None:
            continue
        s = str(root)
        os.environ["HF_HOME"] = s
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = s
        return str(snap.resolve())

    root = os.environ.get("HF_HOME", "")
    checked = ", ".join(str(p) for p in roots) or "(none)"
    raise RuntimeError(
        f"No on-disk snapshot for {label} ({repo_id!r}) under {root}. "
        "Expected …/hub/models--<org>--<name>/snapshots/<hash>/ or "
        "…/models--<org>--<name>/snapshots/<hash>/ (flat cache). "
        f"Checked roots: {checked}. "
        "Run from repo root with network: python warmup_hf_models.py",
    )


_embed_model_lock = threading.Lock()


def make_embed_model(device: str) -> HuggingFaceEmbeddings:
    """Load BCE embedding model (serialized — concurrent loads break on meta tensors)."""
    with _embed_model_lock:
        return _make_embed_model_unlocked(device)


def _make_embed_model_unlocked(device: str) -> HuggingFaceEmbeddings:
    local = _local_snapshot_path_or_fail(EMBEDDING_MODEL, "BCE embedding")
    return HuggingFaceEmbeddings(
        model_name=local,
        model_kwargs={
            "device": device,
            "local_files_only": True,
            "model_kwargs": {"low_cpu_mem_usage": False},
        },
        encode_kwargs={"batch_size": EMBED_BATCH_SIZE, "normalize_embeddings": True},
        cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
    )


def make_reranker(device: str, top_n: int) -> RagretBCERerank:
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


def embed_batch(
    embed_model: Any,
    contents: list[str],
    *,
    on_batch: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[list[float]]:
    n = len(contents)
    if n == 0:
        return []
    out: list[list[float]] = []
    report_step = max(EMBED_BATCH_SIZE, max(1, n // 40))
    last_reported = 0
    for i in range(0, n, EMBED_BATCH_SIZE):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("embedding cancelled")
        batch = contents[i : i + EMBED_BATCH_SIZE]
        out.extend(embed_model.embed_documents(batch))
        done = min(i + len(batch), n)
        if done - last_reported >= report_step or done == n:
            if on_batch is not None:
                on_batch(done, n)
            last_reported = done
    return out
