from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

DEFAULT_CHAT_MODEL = "gemini-1.5-pro"
DEFAULT_EMBEDDING_MODEL = "models/embedding-001"
DEFAULT_TOP_K = 3
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000
DEFAULT_QDRANT_URL = "https://2468b96d-f0ae--a440-74b23e10aa08.us-east-1-1.aws.cloud.qdrant.io:6333"
DEFAULT_QDRANT_COLLECTION_NAME = "basic_rag_documents"


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


def resolve_qdrant_url(explicit_url: str | None = None) -> str | None:
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


DEFAULT_POSTGRES_URL = "postgresql://user:password@localhost:5432/ragdb"

def resolve_postgres_url(explicit_url: str | None = None) -> str | None:
    if explicit_url and explicit_url.strip():
        return explicit_url.strip()
    url = os.getenv("POSTGRES_URL")
    return url.strip() if url and url.strip() else None
