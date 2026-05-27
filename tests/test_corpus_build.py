from __future__ import annotations

from pathlib import Path

import pytest

from ragret.corpus_build import build_chunks_from_workdir
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
