from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from uuid import uuid4, uuid5, NAMESPACE_DNS
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import (
    DEFAULT_EMBEDDING_MODEL,
    resolve_google_api_key,
    resolve_qdrant_api_key,
    resolve_qdrant_collection_name,
    resolve_qdrant_url,
)

DEFAULT_QDRANT_DISTANCE = "Cosine"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60


def documents_from_pdf_bytes(pdf_bytes: bytes, source: str) -> list[Document]:
    """Extract page text from an uploaded PDF."""

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:  # pragma: no cover - depends on file validity
        raise ValueError("Unable to read the uploaded file as a PDF.") from exc

    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": source, "page": page_number},
            )
        )

    if not documents:
        raise ValueError(
            "No text could be extracted from the PDF. If this is a scanned PDF, OCR is required."
        )

    return documents


@dataclass(slots=True)
class QdrantDocumentStore:
    url: str
    collection_name: str
    api_key: str | None
    embeddings: GoogleGenerativeAIEmbeddings
    upload_id: str

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        *,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        google_api_key: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> QdrantDocumentStore:
        """Create a Qdrant-backed document store from already loaded documents."""

        if not documents:
            raise ValueError("At least one document is required to build a vector store.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        splits = splitter.split_documents(list(documents))
        if not splits:
            raise ValueError("No text could be extracted from the supplied documents.")

        api_key = resolve_google_api_key(google_api_key)
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY."
            )

        url = resolve_qdrant_url(qdrant_url)
        if not url:
            raise ValueError(
                "QDRANT_URL is missing. Set it in the environment or pass it explicitly."
            )

        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model or DEFAULT_EMBEDDING_MODEL,
            google_api_key=api_key,
        )
        store = cls(
            url=url,
            collection_name=resolve_qdrant_collection_name(collection_name),
            api_key=resolve_qdrant_api_key(qdrant_api_key),
            embeddings=embeddings,
            upload_id=uuid4().hex,
        )
        store._replace_collection(splits)
        return store

    def similarity_search(self, query: str, k: int) -> list[Document]:
        """Return the most similar chunks for a query."""

        if not query.strip():
            return []

        query_vector = self.embeddings.embed_query(query)
        response = self._request_json(
            "POST",
            f"/collections/{quote(self.collection_name, safe='')}/points/search",
            body={
                "vector": query_vector,
                "limit": k,
                "with_payload": True,
                "with_vector": False,
                "filter": {
                    "must": [
                        {
                            "key": "upload_id",
                            "match": {"value": self.upload_id},
                        }
                    ]
                },
            },
        )

        results = response.get("result", [])
        if not isinstance(results, list):
            raise RuntimeError("Qdrant returned an unexpected search response.")

        documents: list[Document] = []
        for item in results:
            if not isinstance(item, dict):
                continue

            payload = item.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}

            text = str(payload.get("text", ""))
            metadata = {
                key: value
                for key, value in payload.items()
                if key != "text" and value is not None
            }
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def _replace_collection(self, documents: Sequence[Document]) -> None:
        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        if not vectors:
            raise ValueError("No embeddings could be generated for the supplied documents.")

        self._ensure_collection(vector_size=len(vectors[0]))

        points = []
        total_chunks = len(documents)
        for index, (document, vector) in enumerate(zip(documents, vectors), start=1):
            payload = {
                **{
                    key: value
                    for key, value in document.metadata.items()
                    if value is not None
                },
                "text": document.page_content,
                "chunk_index": index,
                "chunk_total": total_chunks,
                "upload_id": self.upload_id,
            }
            points.append(
                {
                    # "id": f"{self.upload_id}:{index}",
                    "id": str(uuid5(NAMESPACE_DNS, f"{self.upload_id}:{index}")),
                    "payload": payload,
                    "vector": vector,
                }
            )

        self._request_json(
            "PUT",
            f"/collections/{quote(self.collection_name, safe='')}/points",
            query={"wait": "true"},
            body={"points": points},
        )

    def _create_collection(self, *, vector_size: int) -> None:
        self._request_json(
            "PUT",
            f"/collections/{quote(self.collection_name, safe='')}",
            query={"timeout": DEFAULT_REQUEST_TIMEOUT_SECONDS},
            body={
                "vectors": {
                    "size": vector_size,
                    "distance": DEFAULT_QDRANT_DISTANCE,
                }
            },
        )
    def _ensure_payload_index(self) -> None:
        self._request_json(
            "PUT",
            f"/collections/{quote(self.collection_name, safe='')}/index",
            body={
                "field_name": "upload_id",
                "field_schema": "keyword",
            },
        )

    def _ensure_collection(self, *, vector_size: int) -> None:
        existing = self._request_json(
            "GET",
            f"/collections/{quote(self.collection_name, safe='')}",
            ignore_not_found=True,
        )
        if not existing:
            self._create_collection(vector_size=vector_size)
        else:
            existing_size = _extract_vector_size(existing)
            if existing_size is not None and existing_size != vector_size:
                raise RuntimeError(
                    "The Qdrant collection already exists with a different vector size. "
                    "Use a different QDRANT_COLLECTION_NAME or recreate the collection with matching embeddings."
                )

        self._ensure_payload_index()  # always ensure index exists



    # def _ensure_collection(self, *, vector_size: int) -> None:
    #     existing = self._request_json(
    #         "GET",
    #         f"/collections/{quote(self.collection_name, safe='')}",
    #         ignore_not_found=True,
    #     )
    #     if not existing:
    #         self._create_collection(vector_size=vector_size)
    #         return

    #     existing_size = _extract_vector_size(existing)
    #     if existing_size is None:
    #         return

    #     if existing_size != vector_size:
    #         raise RuntimeError(
    #             "The Qdrant collection already exists with a different vector size. "
    #             "Use a different QDRANT_COLLECTION_NAME or recreate the collection with matching embeddings."
    #         )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, object] | None = None,
        body: object | None = None,
        ignore_not_found: bool = False,
    ) -> dict[str, object]:
        url = self._build_url(path, query=query)
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key

        request = Request(url, data=payload, method=method, headers=headers)
        try:
            with urlopen(request, timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8").strip()
        except HTTPError as exc:
            if ignore_not_found and exc.code == 404:
                return {}

            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Qdrant request failed ({exc.code} {exc.reason}) for {path}: {error_body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Unable to reach Qdrant at {self.url}: {exc.reason}") from exc

        if not raw:
            return {}

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - unexpected response
            raise RuntimeError(f"Qdrant returned invalid JSON for {path}: {raw}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"Qdrant returned an unexpected response for {path}.")

        return data

    def _build_url(self, path: str, query: dict[str, object] | None = None) -> str:
        base = f"{self.url.rstrip('/')}/{path.lstrip('/')}"
        if not query:
            return base

        filtered_query = {
            key: value
            for key, value in query.items()
            if value is not None and value != ""
        }
        if not filtered_query:
            return base

        return f"{base}?{urlencode(filtered_query)}"


def _extract_vector_size(response: dict[str, object]) -> int | None:
    """Extract the configured dense vector size from a collection details response."""

    result = response.get("result")
    if not isinstance(result, dict):
        return None

    config = result.get("config")
    if not isinstance(config, dict):
        return None

    params = config.get("params")
    if not isinstance(params, dict):
        return None

    vectors = params.get("vectors")
    if not isinstance(vectors, dict):
        return None

    size = vectors.get("size")
    return size if isinstance(size, int) else None


def build_vector_store_from_documents(
    documents: Sequence[Document],
    *,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    google_api_key: str | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> QdrantDocumentStore:
    """Create a Qdrant-backed document store from already loaded documents."""

    return QdrantDocumentStore.from_documents(
        documents,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        embedding_model=embedding_model,
        google_api_key=google_api_key,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
