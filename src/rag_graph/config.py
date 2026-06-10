from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

DEFAULT_CHAT_MODEL = "gemini-3-flash-preview"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_TOP_K = 3
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION_NAME = "rag_graph_documents"
DEFAULT_DATABASE_URL = "sqlite:///./data/rag.db"


def _normalize_qdrant_url(raw_url: str) -> str:
    """Normalize a Qdrant cloud URL to the REST API endpoint."""

    url = raw_url.strip()
    parsed = urlparse(url)
    if not parsed.scheme:
        return url.rstrip("/")

    if parsed.hostname:
        port = parsed.port or 6333
        netloc = parsed.hostname
        if parsed.username:
            auth = parsed.username
            if parsed.password:
                auth = f"{auth}:{parsed.password}"
            netloc = f"{auth}@{netloc}"
        netloc = f"{netloc}:{port}"
    else:
        netloc = parsed.netloc

    return urlunparse((parsed.scheme, netloc, "", "", "", "")).rstrip("/")


def resolve_google_api_key(explicit_key: str | None = None) -> str | None:
    """Return a Gemini/Google API key from an explicit value or environment."""

    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return api_key.strip() if api_key and api_key.strip() else None


def resolve_qdrant_url(explicit_url: str | None = None) -> str:
    """Return the Qdrant service URL."""

    if explicit_url and explicit_url.strip():
        return _normalize_qdrant_url(explicit_url)

    url = os.getenv("QDRANT_URL")
    if url and url.strip():
        return _normalize_qdrant_url(url)

    return _normalize_qdrant_url(DEFAULT_QDRANT_URL)


def resolve_qdrant_api_key(explicit_api_key: str | None = None) -> str | None:
    """Return an optional Qdrant API key."""

    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip()

    api_key = os.getenv("QDRANT_API_KEY")
    return api_key.strip() if api_key and api_key.strip() else None


def resolve_qdrant_collection_name(explicit_collection_name: str | None = None) -> str:
    """Return the collection name used for uploaded document chunks."""

    if explicit_collection_name and explicit_collection_name.strip():
        return explicit_collection_name.strip()

    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    if collection_name and collection_name.strip():
        return collection_name.strip()

    return DEFAULT_QDRANT_COLLECTION_NAME


def resolve_database_url(explicit_url: str | None = None) -> str:
    """Return the database URL used for authentication data.

    `DATABASE_URL` is the primary setting. `POSTGRES_URL` remains a legacy
    fallback so older environments keep working.
    """

    if explicit_url and explicit_url.strip():
        return explicit_url.strip()

    for env_name in ("DATABASE_URL", "POSTGRES_URL"):
        url = os.getenv(env_name)
        if url and url.strip():
            return url.strip()

    return DEFAULT_DATABASE_URL


def resolve_postgres_url(explicit_url: str | None = None) -> str:
    """Backward-compatible alias for :func:`resolve_database_url`."""

    return resolve_database_url(explicit_url)
