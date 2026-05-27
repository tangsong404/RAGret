from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document

try:
    from docx.opc.exceptions import PackageNotFoundError as DocxPackageNotFoundError
except ImportError:  # pragma: no cover - python-docx optional at import time
    DocxPackageNotFoundError = type("_MissingDocxPackageNotFoundError", (Exception,), {})

logger = logging.getLogger(__name__)

from ragret.citation_urls import build_asset_url
from ragret.chunk_parent import chunk_parent_text
from ragret.loader import iter_raw_corpus_files, relative_source_key
from ragret.image_convert import convert_image_to_png
from ragret.media_assets import save_asset_binary
from ragret.parent_store import write_parent_text
from ragret.preprocess import is_preprocess_supported, preprocess_file
from ragret.preprocess.xlsx import UnreadableXlsxError
from ragret.vision_caption import caption_image_png, format_image_enrichment
from ragret.vision_config import VisionSettings

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
DEFAULT_BUILD_WORKERS = 4
_BUILD_PROGRESS_START = 15
_BUILD_PROGRESS_SPAN = 54
_PARSE_SOURCE_SUFFIXES = frozenset({".docx", ".pdf"})

ChunkProgressFn = Callable[[str, int, str | None], None]


class _BuildProgressState:
    """Thread-safe build progress: parse while preprocessing/captioning, chunk only when splitting."""

    def __init__(self, progress: ChunkProgressFn | None, total: int) -> None:
        self._progress = progress
        self._total = max(1, total)
        self._done = 0
        self._lock = threading.Lock()

    def report(self, phase: str, *, detail: str | None = None, done: int | None = None) -> None:
        if self._progress is None:
            return
        d = self._done if done is None else done
        pct = _BUILD_PROGRESS_START + int(_BUILD_PROGRESS_SPAN * d / self._total)
        line = f"{d}/{self._total}"
        if detail:
            line = f"{line} · {detail}"
        self._progress(phase, pct, line)

    def mark_source_done(self, source_key: str) -> None:
        with self._lock:
            self._done += 1
            d = self._done
        self.report("load", detail=source_key, done=d)


def resolve_build_workers(workers: int | None = None) -> int:
    if workers is not None:
        return max(1, int(workers))
    raw = os.environ.get("RAGRET_BUILD_WORKERS", str(DEFAULT_BUILD_WORKERS))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_BUILD_WORKERS


def iter_index_source_files(work_dir: Path, *, image_ingest_enabled: bool) -> list[Path]:
    root = work_dir.resolve()
    out: list[Path] = []
    for f in iter_raw_corpus_files(work_dir):
        suf = f.suffix.lower()
        if suf in _IMAGE_SUFFIXES:
            if image_ingest_enabled:
                out.append(f)
            continue
        if is_preprocess_supported(f):
            out.append(f)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def build_chunks_from_workdir(
    work_dir: Path,
    *,
    kb_name: str | None = None,
    parents_dir: Path | None = None,
    assets_dir: Path | None = None,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    image_ingest_enabled: bool = False,
    public_host: str | None = None,
    vision_settings: VisionSettings | None = None,
    resume_cache_dir: Path | None = None,
    build_workers: int | None = None,
    progress: ChunkProgressFn | None = None,
) -> list[Document]:
    work_dir = work_dir.resolve()
    sources = iter_index_source_files(work_dir, image_ingest_enabled=image_ingest_enabled)
    if not sources:
        raise ValueError(f"No indexable files under: {work_dir}")
    if image_ingest_enabled and vision_settings is None:
        raise RuntimeError("Missing vision settings while image ingest enabled")

    workers = min(resolve_build_workers(build_workers), len(sources))
    total_sources = len(sources)
    prog = _BuildProgressState(progress, total_sources)
    prog.report("load", detail=None, done=0)

    def _build_one(raw: Path) -> list[Document]:
        rel = relative_source_key(work_dir, str(raw))
        parts = build_chunks_for_source(
            raw_path=raw,
            source_key=rel,
            kb_name=kb_name,
            parents_dir=parents_dir,
            assets_dir=assets_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            image_ingest_enabled=image_ingest_enabled,
            public_host=public_host,
            vision_settings=vision_settings,
            resume_cache_dir=resume_cache_dir,
            progress_state=prog,
        )
        prog.mark_source_done(rel)
        return parts

    texts: list[Document] = []
    if workers <= 1:
        per_source = [_build_one(raw) for raw in sources]
    else:
        logger.info("Building chunks with %d workers for %d sources", workers, total_sources)
        slots: list[list[Document] | None] = [None] * total_sources
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ragret-build") as pool:
            future_to_idx = {pool.submit(_build_one, raw): idx for idx, raw in enumerate(sources)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                slots[idx] = fut.result()
        per_source = [parts if parts is not None else [] for parts in slots]
    for parts in per_source:
        texts.extend(parts)
    if not texts:
        raise ValueError("No chunks after split.")
    return texts


def build_chunks_for_source(
    *,
    raw_path: Path,
    source_key: str,
    kb_name: str | None,
    parents_dir: Path | None,
    assets_dir: Path | None,
    chunk_size: int,
    chunk_overlap: int,
    image_ingest_enabled: bool,
    public_host: str | None,
    vision_settings: VisionSettings | None,
    resume_cache_dir: Path | None = None,
    progress_state: _BuildProgressState | None = None,
) -> list[Document]:
    def _is_empty_text(value: str) -> bool:
        return not str(value or "").strip()

    def _file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as bf:
            for chunk in iter(lambda: bf.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _cache_path_for(source_id: str) -> Path | None:
        if resume_cache_dir is None:
            return None
        resume_cache_dir.mkdir(parents=True, exist_ok=True)
        sid = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:32]
        return (resume_cache_dir / f"{sid}.json").resolve()

    def _cache_context() -> dict[str, object]:
        return {
            "source_key": source_key,
            "source_sha256": _file_sha256(raw_path),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "image_ingest_enabled": bool(image_ingest_enabled),
            "vision_provider": str(vision_settings.provider) if vision_settings is not None else "",
            "vision_model": str(vision_settings.model) if vision_settings is not None else "",
        }

    def _load_cache(cache_path: Path, cache_ctx: dict[str, object]) -> tuple[str, list[Document]] | None:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        for key, value in cache_ctx.items():
            if payload.get(key) != value:
                return None
        parent_text = str(payload.get("parent_text") or "")
        docs_raw = payload.get("docs")
        if not isinstance(docs_raw, list):
            return None
        docs: list[Document] = []
        for item in docs_raw:
            if not isinstance(item, dict):
                return None
            page_content = str(item.get("page_content") or "")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            docs.append(Document(page_content=page_content, metadata=dict(metadata)))
        return parent_text, docs

    def _save_cache(cache_path: Path, cache_ctx: dict[str, object], parent_text: str, docs: list[Document]) -> None:
        payload = dict(cache_ctx)
        payload["parent_text"] = parent_text
        payload["docs"] = [{"page_content": d.page_content, "metadata": dict(d.metadata)} for d in docs]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache_path)

    cache_ctx = _cache_context()
    cache_path = _cache_path_for(source_key)
    if cache_path is not None and cache_path.is_file():
        cached = _load_cache(cache_path, cache_ctx)
        if cached is not None:
            cached_parent, cached_docs = cached
            if parents_dir is not None and cached_parent:
                write_parent_text(parents_dir, source_key, cached_parent)
            for doc in cached_docs:
                doc.metadata["source"] = source_key
            logger.info("Resume cache hit for source: %s", source_key)
            return cached_docs

    def _report_parse(detail: str) -> None:
        if progress_state is not None:
            progress_state.report("parse", detail=detail)

    def _report_chunking() -> None:
        if progress_state is not None:
            progress_state.report("chunk", detail=source_key)

    suf = raw_path.suffix.lower()
    if progress_state is not None:
        if image_ingest_enabled and (suf in _IMAGE_SUFFIXES or suf in _PARSE_SOURCE_SUFFIXES):
            progress_state.report("parse", detail=source_key)
        else:
            progress_state.report("load", detail=source_key)

    def _build_image_block(
        payload: bytes,
        image_name: str,
        content_type: str | None = None,
    ) -> str:
        if assets_dir is None or kb_name is None or vision_settings is None:
            return ""
        safe_name = image_name.replace("\\", "/").strip("/")
        source_for_asset = f"{source_key}/embedded/{safe_name}"
        png_bytes = convert_image_to_png(
            payload,
            mime=content_type,
            filename_hint=safe_name,
        )
        asset_rel = save_asset_binary(
            assets_dir=assets_dir,
            source_key=source_for_asset,
            payload=png_bytes,
            mime="image/png",
            filename_hint=safe_name,
            convert_to_png=False,
        )
        logger.info("Vision caption for embedded image: %s", safe_name)
        sys.stderr.write(f"ragret-vision: caption {source_key} :: {safe_name}\n")
        sys.stderr.flush()
        _report_parse(f"{source_key} :: {safe_name}")
        caption = caption_image_png(png_bytes, vision_settings)
        asset_url = build_asset_url(
            kb_name=kb_name,
            asset_rel_path=asset_rel,
            public_host=public_host,
        )
        return format_image_enrichment(caption, asset_url)

    if raw_path.suffix.lower() in _IMAGE_SUFFIXES:
        if not image_ingest_enabled or assets_dir is None or kb_name is None:
            return []
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(raw_path.suffix.lower())
        if mime is None:
            return []
        if vision_settings is None:
            raise RuntimeError("Missing vision settings while image ingest enabled")
        _report_parse(source_key)
        png_bytes = convert_image_to_png(
            raw_path.read_bytes(),
            mime=mime,
            filename_hint=raw_path.name,
        )
        asset_rel = save_asset_binary(
            assets_dir=assets_dir,
            source_key=source_key,
            payload=png_bytes,
            mime="image/png",
            filename_hint=raw_path.name,
            convert_to_png=False,
        )
        caption = caption_image_png(png_bytes, vision_settings)
        asset_url = build_asset_url(
            kb_name=kb_name,
            asset_rel_path=asset_rel,
            public_host=public_host,
        )
        enriched = format_image_enrichment(caption, asset_url)
    else:
        image_handler = _build_image_block if image_ingest_enabled else None
        try:
            enriched = preprocess_file(raw_path, image_handler=image_handler)
        except zipfile.BadZipFile:
            logger.warning("Skip corrupt office archive (not a zip file): %s", source_key)
            return []
        except UnicodeDecodeError:
            logger.warning("Skip source with undecodable text encoding: %s", source_key)
            return []
        except UnreadableXlsxError:
            logger.warning("Skip unreadable xlsx: %s", source_key)
            return []
        except DocxPackageNotFoundError:
            logger.warning("Skip invalid office package: %s", source_key)
            return []
    if _is_empty_text(enriched):
        # 图片内容没被转成文本：视为错误，而不是静默跳过
        if raw_path.suffix.lower() in _IMAGE_SUFFIXES and image_ingest_enabled:
            raise RuntimeError(f"Image source produced empty text after caption: {source_key}")
        logger.warning("Skip empty source after preprocess: %s", source_key)
        return []

    if parents_dir is not None:
        write_parent_text(parents_dir, source_key, enriched)

    _report_chunking()
    try:
        parts = chunk_parent_text(
            enriched,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=source_key,
        )
    except ValueError as exc:
        if str(exc) == "No chunks after split.":
            logger.warning("Skip source with no chunks after split: %s", source_key)
            return []
        raise
    for doc in parts:
        doc.metadata["source"] = source_key
    if cache_path is not None:
        try:
            _save_cache(cache_path, cache_ctx, enriched, parts)
        except OSError as e:
            logger.warning("Failed to write resume cache for %s: %s", source_key, e)
    return parts
