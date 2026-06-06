from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


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

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = Field(default="default")  # add this line
