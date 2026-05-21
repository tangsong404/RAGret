from __future__ import annotations

import pytest

from ragret.embedder import BuildCancelledError, embed_batch


class MockEmbedModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]


def test_embed_batch_empty() -> None:
    assert embed_batch(MockEmbedModel(), []) == []


def test_embed_batch_progress_callback() -> None:
    calls: list[int] = []
    embed_batch(
        MockEmbedModel(),
        ["a", "bb", "ccc"],
        on_batch=lambda done, total: calls.append(done),
    )
    assert calls[-1] == 3


def test_embed_batch_cancellation() -> None:
    with pytest.raises(BuildCancelledError):
        embed_batch(MockEmbedModel(), ["a", "bb", "ccc"], cancel_check=lambda: True)
