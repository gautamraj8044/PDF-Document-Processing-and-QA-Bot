from __future__ import annotations

import argparse
import os
from threading import RLock

from langchain_core.messages import AIMessage, HumanMessage
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from langgraph.checkpoint.postgres import PostgresSaver
from .config import resolve_postgres_url

from .config import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    resolve_google_api_key,
    resolve_qdrant_api_key,
    resolve_qdrant_collection_name,
    resolve_qdrant_url,
)
from .graph import build_rag_graph
from .knowledge_base import documents_from_pdf_bytes
from .schemas import HealthResponse, QueryRequest, QueryResponse, SourceSnippet, UploadResponse


class RagRuntime:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        embedding_model: str | None = None,
        top_k: int | None = None,
        google_api_key: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        qdrant_collection_name: str | None = None,
    ) -> None:
        self._lock = RLock()
        self._model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_CHAT_MODEL)
        self._embedding_model = embedding_model or os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        )
        self._top_k = top_k if top_k is not None else int(os.getenv("BASIC_RAG_TOP_K", str(DEFAULT_TOP_K)))
        self._google_api_key = resolve_google_api_key(google_api_key)
        self._qdrant_url = resolve_qdrant_url(qdrant_url)
        self._qdrant_api_key = resolve_qdrant_api_key(qdrant_api_key)
        self._qdrant_collection_name = resolve_qdrant_collection_name(qdrant_collection_name)
        self._postgres_url = resolve_postgres_url()
        self._checkpointer: PostgresSaver | None = None
        self._init_checkpointer()
        if not self._qdrant_url:
            raise ValueError(
                "QDRANT_URL is missing. Set it in .env or pass it to RagRuntime."
            )
        self._graph = None
        self._uploaded_file = ""
        self._document_count = 0
        self._graph = self._build_general_graph()


    @property
    def active_source(self) -> str:
        with self._lock:
            return self._uploaded_file

    @property
    def document_count(self) -> int:
        with self._lock:
            return self._document_count

    def upload_pdf(self, filename: str, pdf_bytes: bytes) -> dict[str, object]:
        documents = documents_from_pdf_bytes(pdf_bytes, filename)
        graph = build_rag_graph(
            documents=documents,
            model_name=self._model_name,
            embedding_model=self._embedding_model,
            top_k=self._top_k,
            google_api_key=self._google_api_key,
            qdrant_url=self._qdrant_url,
            qdrant_api_key=self._qdrant_api_key,
            qdrant_collection_name=self._qdrant_collection_name,
            checkpointer=self._checkpointer,
            has_pdf=True,  
        )

        with self._lock:
            self._graph = graph
            self._uploaded_file = filename
            self._document_count = len(documents)

        return {
            "file_name": filename,
            "document_count": len(documents),
        }

    def query(self, question: str, session_id: str = "default") -> dict[str, object]:
        with self._lock:
            graph = self._graph
            uploaded_file = self._uploaded_file
            document_count = self._document_count

        if graph is None:
            raise RuntimeError("Upload a PDF first.")

        config = {"configurable": {"thread_id": session_id}}
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        # Last message in history is the AI response
        answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break

        return {
            "answer": answer,
            "retrieved_docs": result.get("retrieved_docs", []),
            "file_name": uploaded_file,
            "document_count": document_count,
        }
    def _init_checkpointer(self) -> None:
        if not self._postgres_url:
            return
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg

            conn = psycopg.connect(self._postgres_url, autocommit=True)
            self._checkpointer = PostgresSaver(conn)
            self._checkpointer.setup()
        except Exception as exc:
            print(f"Warning: Could not connect to Postgres for checkpointing: {exc}")
            self._checkpointer = None

    def _build_general_graph(self):
        """Stateless general-knowledge graph used before any PDF is uploaded."""
        from .graph import build_rag_graph
        return build_rag_graph(
            google_api_key=self._google_api_key,
            model_name=self._model_name,
            checkpointer=self._checkpointer,
            has_pdf=False,
        )


def create_app(runtime: RagRuntime) -> FastAPI:
    app = FastAPI(title="Gemini RAG Graph")
    app.state.rag_runtime = runtime

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            file_name=runtime.active_source,
            document_count=runtime.document_count,
        )

    @app.post("/upload", response_model=UploadResponse)
    async def upload(file: UploadFile = File(...)) -> UploadResponse:
        print("filename called")
        filename = file.filename or "uploaded.pdf"
        print(filename)
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")

        pdf_bytes = await file.read()
        try:
            result = runtime.upload_pdf(filename, pdf_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - surfaced as HTTP error
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return UploadResponse(status="ok", **result)


    @app.get("/history/{session_id}")
    def get_history(session_id: str) -> dict:
        """Return full conversation history for a session."""
        if runtime._checkpointer is None:
            raise HTTPException(status_code=503, detail="Checkpointer not configured.")

        config = {"configurable": {"thread_id": session_id}}
        state = runtime._graph.get_state(config)

        if not state or not state.values:
            raise HTTPException(status_code=404, detail="Session not found.")

        messages = []
        for msg in state.values.get("messages", []):
            messages.append({
                "role": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content,
            })

        return {"session_id": session_id, "message_count": len(messages), "messages": messages}

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        try:
            result = runtime.query(question, session_id=request.session_id)  # pass session_id
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - surfaced as HTTP error
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return QueryResponse(
            answer=result["answer"],
            file_name=result["file_name"],
            document_count=result["document_count"],
            sources=_summarize_documents(result["retrieved_docs"]),
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Basic RAG PDF API server.")
    parser.add_argument(
        "--host",
        default=os.getenv("BASIC_RAG_API_HOST", DEFAULT_API_HOST),
        help="Host interface to bind the API server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("BASIC_RAG_API_PORT", str(DEFAULT_API_PORT))),
        help="Port to bind the API server to.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_CHAT_MODEL),
        help="Chat model to use.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="Embedding model to use.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("BASIC_RAG_TOP_K", str(DEFAULT_TOP_K))),
        help="Number of chunks to retrieve for each question.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=resolve_qdrant_url(),
        help="Qdrant service URL for the vector store.",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=resolve_qdrant_api_key(),
        help="Optional Qdrant API key for cloud deployments.",
    )
    parser.add_argument(
        "--qdrant-collection-name",
        default=resolve_qdrant_collection_name(),
        help="Collection name used to store uploaded PDF chunks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not resolve_google_api_key():
        parser.error("GEMINI_API_KEY or GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
    if not args.qdrant_url:
        parser.error(
            "QDRANT_URL is not set. Copy .env.example to .env and add your Qdrant service URL."
        )

    runtime = RagRuntime(
        model_name=args.model,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_collection_name=args.qdrant_collection_name,
    )
    app = create_app(runtime)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _summarize_documents(documents: list) -> list[SourceSnippet]:
    sources: list[SourceSnippet] = []
    for doc in documents:
        preview = " ".join(doc.page_content.split())
        if len(preview) > 240:
            preview = f"{preview[:237]}..."

        sources.append(
            SourceSnippet(
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                preview=preview,
            )
        )
    return sources


if __name__ == "__main__":
    raise SystemExit(main())
