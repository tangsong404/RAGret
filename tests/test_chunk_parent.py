from __future__ import annotations

from ragret.chunk_parent import chunk_parent_text


def test_chunk_parent_text_records_line_numbers() -> None:
    lines = [f"line-{i}" for i in range(1, 41)]
    text = "\n".join(lines)
    chunks = chunk_parent_text(text, chunk_size=80, chunk_overlap=0)
    assert len(chunks) >= 2
    first = chunks[0]
    assert first.metadata["line_start"] == 1
    assert first.metadata["line_end"] >= 1
    assert "line-1" in first.page_content
    for doc in chunks:
        assert doc.metadata["line_start"] <= doc.metadata["line_end"]
