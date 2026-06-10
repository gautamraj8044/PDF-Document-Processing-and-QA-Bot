from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag_graph.config import (
    DEFAULT_DATABASE_URL,
    DEFAULT_QDRANT_URL,
    resolve_database_url,
    resolve_qdrant_url,
)
from rag_graph.knowledge_base import documents_from_pdf_bytes
from rag_graph.server import RagRuntime
from rag_graph.user_store import UserStore


def test_resolve_qdrant_url_defaults_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert resolve_qdrant_url() == DEFAULT_QDRANT_URL


def test_resolve_database_url_prefers_explicit_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    assert resolve_database_url() == DEFAULT_DATABASE_URL

    monkeypatch.setenv("DATABASE_URL", "sqlite:///./custom.db")
    assert resolve_database_url() == "sqlite:///./custom.db"

    assert resolve_database_url("sqlite:///./explicit.db") == "sqlite:///./explicit.db"


def test_resolve_database_url_prefers_database_url_over_postgres_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./preferred.db")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/ragdb")

    assert resolve_database_url() == "sqlite:///./preferred.db"


def test_user_store_round_trip() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(dir=workspace_root) as temp_dir:
        database_url = "sqlite:///" + (Path(temp_dir) / "users.db").as_posix()
        store = UserStore(database_url)
        store.setup()
        try:
            user = store.create_user(" Test@Example.com ", "hashed-password")

            assert user.email == "test@example.com"
            assert store.get_by_email("test@example.com") == user
            assert store.get_by_id(user.id) == user

            with pytest.raises(ValueError, match="already exists"):
                store.create_user("test@example.com", "other")
        finally:
            store.close()


def test_documents_from_pdf_bytes_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        documents_from_pdf_bytes(b"", "example.pdf")


def test_session_owner_check() -> None:
    assert RagRuntime.session_belongs_to_user("user-123:abc", "user-123")
    assert not RagRuntime.session_belongs_to_user("user-123:abc", "user-456")
