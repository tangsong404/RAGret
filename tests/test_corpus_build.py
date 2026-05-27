from __future__ import annotations

from pathlib import Path

import pytest

from ragret.corpus_build import build_chunks_from_workdir, resolve_build_workers
from ragret.vision_config import VisionSettings


def test_image_ingest_creates_sidecar_chunks_and_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ragret.corpus_build.caption_image_png",
        lambda _png, _settings: "产品界面截图",
    )

    work = tmp_path / "work"
    work.mkdir()
    (work / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\nrest")
    parents = tmp_path / "parents"
    assets = tmp_path / "assets"
    docs = build_chunks_from_workdir(
        work,
        kb_name="demo",
        parents_dir=parents,
        assets_dir=assets,
        image_ingest_enabled=True,
        vision_settings=VisionSettings(
            provider="openai",
            base_url="https://x/v1",
            model="vision",
            api_key="k",
        ),
        public_host="https://ragret.example.com",
        chunk_size=400,
        chunk_overlap=0,
    )
    assert docs
    assert "产品界面截图" in docs[0].page_content
    assert "图片: https://ragret.example.com/api/kb/demo/assets/" in docs[0].page_content
    assert "Image from" not in docs[0].page_content
    assert (parents / "photo.png.txt").is_file()
    assert any(p.is_file() for p in assets.rglob("*.png"))


def test_build_chunks_resume_cache_skips_second_caption_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}

    def _fake_caption(_png: bytes, _settings: VisionSettings) -> str:
        calls["n"] += 1
        return "缓存命中测试"

    monkeypatch.setattr("ragret.corpus_build.caption_image_png", _fake_caption)

    work = tmp_path / "work"
    work.mkdir()
    (work / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\nresume-cache")
    parents = tmp_path / "parents"
    assets = tmp_path / "assets"
    resume_cache = tmp_path / "resume-cache"
    vision = VisionSettings(
        provider="openai",
        base_url="https://x/v1",
        model="vision",
        api_key="k",
    )

    docs_first = build_chunks_from_workdir(
        work,
        kb_name="demo",
        parents_dir=parents,
        assets_dir=assets,
        image_ingest_enabled=True,
        vision_settings=vision,
        public_host="https://ragret.example.com",
        resume_cache_dir=resume_cache,
    )
    docs_second = build_chunks_from_workdir(
        work,
        kb_name="demo",
        parents_dir=parents,
        assets_dir=assets,
        image_ingest_enabled=True,
        vision_settings=vision,
        public_host="https://ragret.example.com",
        resume_cache_dir=resume_cache,
    )

    assert docs_first
    assert docs_second
    assert calls["n"] == 1


def test_parallel_build_workers_processes_multiple_sources(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    for i in range(3):
        (work / f"doc{i}.md").write_text(
            f"document {i} with enough text for chunks\n" * 5,
            encoding="utf-8",
        )
    docs = build_chunks_from_workdir(work, build_workers=4)
    assert len(docs) >= 3
    assert resolve_build_workers(4) == 4


def test_build_chunks_reports_chunk_progress(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.md").write_text("alpha text for chunks\n" * 4, encoding="utf-8")
    (work / "b.md").write_text("beta text for chunks\n" * 4, encoding="utf-8")
    ticks: list[tuple[str, int, str | None]] = []

    def on_progress(phase: str, pct: int, detail: str | None = None) -> None:
        ticks.append((phase, pct, detail))

    build_chunks_from_workdir(work, progress=on_progress, build_workers=1)
    assert ticks
    assert ticks[0][0] == "load"
    phases = {p for p, _, _ in ticks}
    assert "load" in phases
    assert ticks[-1][1] >= ticks[0][1]
    assert "2/2" in str(ticks[-1][2] or "")


def test_corrupt_xlsx_is_skipped(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "bad.xlsx").write_bytes(b"not-a-zip-file")
    (work / "ok.md").write_text("hello world with enough text for chunks", encoding="utf-8")
    docs = build_chunks_from_workdir(work)
    assert docs
    assert all("bad.xlsx" not in str(d.metadata.get("source", "")) for d in docs)


def test_binary_text_file_is_skipped(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "bad.txt").write_bytes(bytes(range(256)))
    (work / "ok.md").write_text("hello world with enough text for chunks", encoding="utf-8")
    docs = build_chunks_from_workdir(work)
    assert docs
    assert all("bad.txt" not in str(d.metadata.get("source", "")) for d in docs)


def test_unreadable_xlsx_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ragret.preprocess.xlsx._preprocess_xlsx_openpyxl",
        lambda _path, _openpyxl: (_ for _ in ()).throw(TypeError("expected Fill")),
    )
    monkeypatch.setattr(
        "ragret.preprocess.xlsx._preprocess_xlsx_xml_fallback",
        lambda _path: (_ for _ in ()).throw(OSError("broken")),
    )
    work = tmp_path / "work"
    work.mkdir()
    (work / "bad.xlsx").write_bytes(b"PK\x03\x04")
    (work / "ok.md").write_text("hello world with enough text for chunks", encoding="utf-8")
    docs = build_chunks_from_workdir(work)
    assert docs
    assert all("bad.xlsx" not in str(d.metadata.get("source", "")) for d in docs)
