from __future__ import annotations

from ragret.citation_urls import build_asset_url


def test_build_asset_url_uses_public_host() -> None:
    url = build_asset_url(
        kb_name="mykb",
        asset_rel_path="docs/img/a.png",
        public_host="https://ragret.example.com",
    )
    assert url == "https://ragret.example.com/api/kb/mykb/assets/docs/img/a.png"
