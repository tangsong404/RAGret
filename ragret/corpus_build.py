from __future__ import annotations

import logging
import sys
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

from ragret.citation_urls import build_asset_url
from ragret.chunk_parent import chunk_parent_text
from ragret.loader import iter_raw_corpus_files, relative_source_key
from ragret.image_convert import convert_image_to_png
from ragret.media_assets import save_asset_binary
from ragret.parent_store import write_parent_text
from ragret.preprocess import is_preprocess_supported, preprocess_file
from ragret.vision_caption import caption_image_png, format_image_enrichment
from ragret.vision_config import VisionSettings

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})


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
) -> list[Document]:
    work_dir = work_dir.resolve()
    sources = iter_index_source_files(work_dir, image_ingest_enabled=image_ingest_enabled)
    if not sources:
        raise ValueError(f"No indexable files under: {work_dir}")
    if image_ingest_enabled and vision_settings is None:
        raise RuntimeError("Missing vision settings while image ingest enabled")

    texts: list[Document] = []
    for raw in sources:
        rel = relative_source_key(work_dir, str(raw))
        texts.extend(
            build_chunks_for_source(
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
            )
        )
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
) -> list[Document]:
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
        enriched = preprocess_file(raw_path, image_handler=image_handler)
    if parents_dir is not None:
        write_parent_text(parents_dir, source_key, enriched)
    parts = chunk_parent_text(
        enriched,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source=source_key,
    )
    for doc in parts:
        doc.metadata["source"] = source_key
    return parts
