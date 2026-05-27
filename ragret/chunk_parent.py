from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _char_to_line(text: str, char_index: int) -> int:
    if char_index <= 0:
        return 1
    return text.count("\n", 0, char_index) + 1


def chunk_parent_text(
    text: str,
    *,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    source: str = "",
) -> list[Document]:
    if not str(text or "").strip():
        raise ValueError("No chunks after split.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    base = Document(page_content=text, metadata={"source": source} if source else {})
    parts = splitter.split_documents([base])
    if not parts:
        raise ValueError("No chunks after split.")

    cursor = 0
    out: list[Document] = []
    for doc in parts:
        piece = doc.page_content
        start = text.find(piece, cursor)
        if start < 0:
            start = text.find(piece)
        if start < 0:
            start = cursor
        end = start + len(piece)
        cursor = max(cursor, end - chunk_overlap) if chunk_overlap else end
        meta = dict(doc.metadata)
        meta["line_start"] = _char_to_line(text, start)
        meta["line_end"] = _char_to_line(text, max(start, end - 1))
        meta["char_start"] = start
        meta["char_end"] = end
        out.append(Document(page_content=piece, metadata=meta))
    return out
