from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ImageHandler = Callable[..., str]

_EMBED_ATTR = qn("r:embed")
_LEGACY_EMBED_ATTR = qn("r:id")
_VML_IMAGEDATA_TAG = "{urn:schemas-microsoft-com:vml}imagedata"


def _run_image_rids(run) -> list[str]:
    rids: list[str] = []
    element = run._element
    for blip in element.iter(qn("a:blip")):
        rid = blip.get(_EMBED_ATTR)
        if rid:
            rids.append(rid)
    for imagedata in element.iter(_VML_IMAGEDATA_TAG):
        rid = imagedata.get(_LEGACY_EMBED_ATTR)
        if rid:
            rids.append(rid)
    return rids


def _resolve_image_part(document, r_id: str):
    part = document.part.related_parts.get(r_id)
    if part is not None:
        return part
    rel = document.part.rels.get(r_id)
    if rel is None:
        return None
    return getattr(rel, "target_part", None)


def _iter_body_paragraphs(document) -> Iterator[Paragraph]:
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            for row in table.rows:
                for cell in row.cells:
                    yield from cell.paragraphs


def _render_paragraph(paragraph: Paragraph, document, image_handler: ImageHandler | None) -> str:
    chunks: list[str] = []
    for run in paragraph.runs:
        text = run.text or ""
        if text:
            chunks.append(text)
        if image_handler is None:
            continue
        for rid in _run_image_rids(run):
            target = _resolve_image_part(document, rid)
            if target is None:
                continue
            ctype = str(getattr(target, "content_type", "") or "")
            if not ctype.startswith("image/"):
                continue
            blob = getattr(target, "blob", b"") or b""
            if not blob:
                continue
            name = str(getattr(target, "partname", "") or "docx-image")
            block = image_handler(blob, name, ctype)
            if block.strip():
                chunks.append("\n\n")
                chunks.append(block.strip())
                chunks.append("\n\n")
    return "".join(chunks).strip()


def preprocess_docx(path: Path, *, image_handler: ImageHandler | None = None) -> str:
    try:
        import docx
    except ImportError as e:
        raise RuntimeError("python-docx is required for .docx preprocessing") from e

    document = docx.Document(str(path))
    sections: list[str] = []
    for paragraph in _iter_body_paragraphs(document):
        block = _render_paragraph(paragraph, document, image_handler)
        if block:
            sections.append(block)
    return "\n\n".join(sections).strip()
