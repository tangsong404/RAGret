from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: int
    source: str
    chunk_index: int
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SearchResult:
    content: str
    source: str
    chunk_index: int
    vector_score: float
    relevance_score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class IndexMeta:
    embedding_model: str
    embed_dim: int
    chunk_size: int
    chunk_overlap: int
    indexed_at: int
    schema_version: str
    source_fingerprints: dict[str, str]
