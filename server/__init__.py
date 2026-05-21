"""HTTP API process: auth, knowledge-base ACL, static UI. Depends on ``ragret`` for RAG + registry."""
from __future__ import annotations

from server.main import create_app

__all__ = ["create_app"]
