from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_RAW_CORPUS_GLOBS = (
    "**/*.pdf",
    "**/*.txt",
    "**/*.md",
    "**/*.markdown",
    "**/*.docx",
    "**/*.xlsx",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.webp",
    "**/*.gif",
)
_SKIP_CORPUS_DIR_NAMES = frozenset({"kb_parents", "kb_assets"})


def is_ignored_corpus_filename(name: str) -> bool:
    """Skip Office lock/temp files that are not real documents."""
    base = Path(name).name
    if base.startswith("~$"):
        return True
    if base.startswith(".~lock."):
        return True
    return False


def _path_under_skipped_dir(rel_posix: str) -> bool:
    parts = [p for p in rel_posix.split("/") if p]
    return any(part in _SKIP_CORPUS_DIR_NAMES for part in parts)


def iter_raw_corpus_files(work_dir: Path) -> list[Path]:
    if not work_dir.exists():
        raise FileNotFoundError(work_dir)
    if work_dir.is_file():
        return [work_dir.resolve()]
    root = work_dir.resolve()
    out: list[Path] = []
    seen: set[Path] = set()
    for glob_pat in _RAW_CORPUS_GLOBS:
        for f in sorted(root.glob(glob_pat)):
            if not f.is_file():
                continue
            rel = f.relative_to(root).as_posix()
            if _path_under_skipped_dir(rel):
                continue
            if is_ignored_corpus_filename(f.name):
                continue
            resolved = f.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)
    return out


def corpus_fingerprint_map(work_dir: Path) -> dict[str, str]:
    root = work_dir.resolve()
    m: dict[str, str] = {}
    for f in iter_raw_corpus_files(work_dir):
        key = f.relative_to(root).as_posix()
        m[key] = _file_sha256(f)
    return m


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
