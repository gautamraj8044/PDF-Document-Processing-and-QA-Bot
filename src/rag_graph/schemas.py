from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str


class HealthResponse(BaseModel):
    status: str
    file_name: str
    document_count: int


class UploadResponse(BaseModel):
    status: str
    file_name: str
    document_count: int


class SourceSnippet(BaseModel):
    source: str
    page: int | None = None
    preview: str


class QueryResponse(BaseModel):
    answer: str
    file_name: str
    document_count: int
    sources: list[SourceSnippet]

# Add these to existing schemas.py

class SignupRequest(BaseModel):
    email: str = Field(min_length=5)
    password: str = Field(min_length=8)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str


class UserResponse(BaseModel):
    id: str
    email: str

