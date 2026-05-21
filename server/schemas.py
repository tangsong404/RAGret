from __future__ import annotations

from pydantic import BaseModel, Field


# --- Auth ---
class AuthRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9._-]{3,64}$")
    password: str = Field(min_length=8)


class AuthLoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    has_avatar: bool = False


class AuthResponse(BaseModel):
    ok: bool = True
    token: str
    user: UserOut


class MeResponse(BaseModel):
    ok: bool = True
    user: UserOut | None = None
    superuser: bool = False


# --- Search ---
class SearchResultOut(BaseModel):
    content: str
    source: str
    chunk_index: int
    vector_score: float
    relevance_score: float


class SearchResponse(BaseModel):
    ok: bool = True
    index: str
    query: str
    results: list[SearchResultOut]


# --- Knowledge Base ---
class KBPermissionOut(BaseModel):
    can_read: bool
    can_write: bool
    can_delete: bool
    is_owner: bool


class KBOwnerOut(BaseModel):
    id: int
    username: str
    has_avatar: bool


class KBOut(BaseModel):
    name: str
    description: str
    sqlite_exists: bool
    is_public: bool
    icon: str
    source_type: str
    owner: KBOwnerOut
    permission: KBPermissionOut


class KBListResponse(BaseModel):
    ok: bool = True
    indexes: list[KBOut]


class BuildJobRequest(BaseModel):
    name: str
    description: str
    readme_md: str = ""
    upload_id: str | None = None
    source_type: str = "tar"
    is_public: bool = False
    icon: str = "book"
    webhook_provider: str = ""
    webhook_secret: str = ""
    repo_url: str = ""
    ref: str = ""


class BuildJobResponse(BaseModel):
    ok: bool = True
    job_id: str
    webhook_url: str | None = None
    folder_push_url: str | None = None


class JobOut(BaseModel):
    job_id: str
    status: str
    phase: str
    percent: int
    detail: str
    error: str | None = None
    result: dict | None = None
    op: str
    kb_name: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


# --- Generic ---
class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
