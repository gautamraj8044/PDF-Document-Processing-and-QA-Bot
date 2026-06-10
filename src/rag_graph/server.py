from __future__ import annotations

import argparse
import logging
import os
from threading import RLock
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from .auth import create_access_token, get_current_user, hash_password, verify_password
from .config import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_DATABASE_URL,
    resolve_database_url,
    resolve_google_api_key,
    resolve_qdrant_api_key,
    resolve_qdrant_collection_name,
    resolve_qdrant_url,
)
from .graph import build_rag_graph
from .knowledge_base import build_vector_store_from_documents, documents_from_pdf_bytes
from .schemas import HealthResponse, LoginResponse, QueryRequest, QueryResponse, SignupRequest, SourceSnippet, UploadResponse, UserResponse
from .user_store import UserStore


logger = logging.getLogger(__name__)


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
        database_url: str | None = None,
    ) -> None:
        self._lock = RLock()
        self._model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_CHAT_MODEL)
        self._embedding_model = embedding_model or os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        )
        self._top_k = top_k if top_k is not None else int(os.getenv("RAG_GRAPH_TOP_K", str(DEFAULT_TOP_K)))
        self._google_api_key = resolve_google_api_key(google_api_key)
        self._qdrant_url = resolve_qdrant_url(qdrant_url)
        self._qdrant_api_key = resolve_qdrant_api_key(qdrant_api_key)
        self._qdrant_collection_name = resolve_qdrant_collection_name(qdrant_collection_name)
        self._database_url = resolve_database_url(database_url)
        self._checkpointer = self._init_checkpointer()
        self._user_store = self._init_user_store()
        self._graph = self._build_general_graph()
        self._document_store = None
        self._uploaded_file: str | None = None
        self._document_count = 0

    @property
    def active_source(self) -> str | None:
        with self._lock:
            return self._uploaded_file

    @property
    def document_count(self) -> int:
        with self._lock:
            return self._document_count

    def upload_pdf(self, filename: str, pdf_bytes: bytes) -> dict[str, object]:
        documents = documents_from_pdf_bytes(pdf_bytes, filename)
        vector_store = build_vector_store_from_documents(
            documents,
            qdrant_url=self._qdrant_url,
            qdrant_api_key=self._qdrant_api_key,
            collection_name=self._qdrant_collection_name,
            embedding_model=self._embedding_model,
            google_api_key=self._google_api_key,
        )
        graph = build_rag_graph(
            vector_store=vector_store,
            model_name=self._model_name,
            embedding_model=self._embedding_model,
            top_k=self._top_k,
            google_api_key=self._google_api_key,
            checkpointer=self._checkpointer,
            has_pdf=True,
        )

        with self._lock:
            self._graph = graph
            self._document_store = vector_store
            self._uploaded_file = filename
            self._document_count = len(documents)

        return {
            "file_name": filename,
            "document_count": len(documents),
        }

    def query(self, question: str, session_id: str = "default") -> dict[str, object]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        with self._lock:
            graph = self._graph
            uploaded_file = self._uploaded_file
            document_count = self._document_count

        if graph is None:
            raise RuntimeError("The runtime is not ready yet.")

        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": session_id}},
        )

        answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                answer = str(msg.content)
                break

        return {
            "answer": answer,
            "retrieved_docs": result.get("retrieved_docs", []),
            "file_name": uploaded_file,
            "document_count": document_count,
        }

    def create_session_id(self, user_id: str) -> str:
        import uuid

        return f"{user_id}:{uuid.uuid4()}"

    @staticmethod
    def session_belongs_to_user(session_id: str, user_id: str) -> bool:
        owner, _, _ = session_id.partition(":")
        return owner == user_id

    def _init_checkpointer(self):
        if not self._database_url.startswith("postgresql"):
            return MemorySaver()

        try:
            import psycopg

            conn = psycopg.connect(self._database_url, autocommit=True)
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            logger.info("Postgres checkpointer ready.")
            return checkpointer
        except Exception as exc:
            logger.warning("Falling back to in-memory history: %s", exc)
            return MemorySaver()

    def _build_general_graph(self):
        return build_rag_graph(
            google_api_key=self._google_api_key,
            model_name=self._model_name,
            checkpointer=self._checkpointer,
            has_pdf=False,
        )

    def _init_user_store(self) -> UserStore:
        try:
            store = UserStore(self._database_url)
            store.setup()
            return store
        except Exception as exc:
            if self._database_url != DEFAULT_DATABASE_URL:
                logger.warning(
                    "User store init failed for %s, falling back to SQLite: %s",
                    self._database_url,
                    exc,
                )
                try:
                    store = UserStore(DEFAULT_DATABASE_URL)
                    store.setup()
                    self._database_url = DEFAULT_DATABASE_URL
                    return store
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Unable to initialize the user store: {fallback_exc}"
                    ) from fallback_exc

            raise RuntimeError(f"Unable to initialize the user store: {exc}") from exc


def create_app(runtime: RagRuntime) -> FastAPI:
    app = FastAPI(
        title="Basic RAG",
        description=(
            "A LangGraph-based RAG API with Gemini, Qdrant, JWT authentication, "
            "and optional Postgres-backed persistence."
        ),
        version="0.1.0",
    )
    app.state.rag_runtime = runtime

    @app.post(
        "/auth/signup",
        response_model=UserResponse,
        tags=["Auth"],
        summary="Create a new account",
        responses={400: {"description": "Email already registered"}},
    )
    def signup(request: SignupRequest) -> UserResponse:
        user = runtime._user_store.create_user(
            email=request.email,
            hashed_password=hash_password(request.password),
        )
        return UserResponse(id=user.id, email=user.email)

    @app.post(
        "/auth/login",
        response_model=LoginResponse,
        tags=["Auth"],
        summary="Login with email and password",
        description="Uses an OAuth2 password form. The username field should contain the email address.",
        responses={401: {"description": "Invalid credentials"}},
    )
    def login(form: OAuth2PasswordRequestForm = Depends()) -> LoginResponse:
        user = runtime._user_store.get_by_email(form.username)
        if not user or not verify_password(form.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        token = create_access_token(user_id=user.id, email=user.email)
        return LoginResponse(access_token=token, email=user.email)

    @app.get(
        "/auth/me",
        response_model=UserResponse,
        tags=["Auth"],
        summary="Get the current user",
    )
    def me(current_user: Annotated[dict, Depends(get_current_user)]) -> UserResponse:
        return UserResponse(id=current_user["sub"], email=current_user["email"])

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Documents"],
        summary="Check server and document status",
    )
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            file_name=runtime.active_source,
            document_count=runtime.document_count,
        )

    @app.post(
        "/upload",
        response_model=UploadResponse,
        tags=["Documents"],
        summary="Upload a PDF for querying",
        responses={
            400: {"description": "Invalid or non-PDF file"},
            503: {"description": "Vector backend unavailable"},
        },
    )
    async def upload(
        file: UploadFile = File(...),
        _current_user: dict = Depends(get_current_user),
    ) -> UploadResponse:
        filename = file.filename or "uploaded.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")

        pdf_bytes = await file.read()
        try:
            result = runtime.upload_pdf(filename, pdf_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return UploadResponse(status="ok", **result)

    @app.get(
        "/session",
        tags=["Conversation"],
        summary="Create a new conversation session",
        description="Returns a session ID scoped to the authenticated user. Use it in /query to maintain conversation history.",
    )
    def new_session(current_user: Annotated[dict, Depends(get_current_user)]) -> dict:
        return {"session_id": runtime.create_session_id(current_user["sub"])}

    @app.post(
        "/query",
        response_model=QueryResponse,
        tags=["Conversation"],
        summary="Ask a question against the active document or chat normally",
        description=(
            "Accepts a question and optional session_id. If no session_id is provided, one is auto-created and returned. "
            "If a PDF has been uploaded, the router decides whether the answer should use the document or general knowledge."
        ),
        responses={
            400: {"description": "Invalid question"},
            403: {"description": "Session does not belong to user"},
            503: {"description": "Backend unavailable"},
        },
    )
    def query(
        request: QueryRequest,
        current_user: Annotated[dict, Depends(get_current_user)],
    ) -> QueryResponse:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        session_id = request.session_id or runtime.create_session_id(current_user["sub"])
        if not runtime.session_belongs_to_user(session_id, current_user["sub"]):
            raise HTTPException(status_code=403, detail="Session does not belong to you.")

        try:
            result = runtime.query(question, session_id=session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return QueryResponse(
            answer=result["answer"],
            session_id=session_id,
            file_name=result["file_name"],
            document_count=result["document_count"],
            sources=_summarize_documents(result["retrieved_docs"]),
        )

    @app.get(
        "/history/{session_id}",
        tags=["Conversation"],
        summary="View conversation history for a session",
        responses={
            403: {"description": "Session does not belong to user"},
            404: {"description": "Session not found"},
        },
    )
    def get_history(
        session_id: str,
        current_user: Annotated[dict, Depends(get_current_user)],
    ) -> dict:
        if not runtime.session_belongs_to_user(session_id, current_user["sub"]):
            raise HTTPException(status_code=403, detail="Session does not belong to you.")

        state = runtime._graph.get_state({"configurable": {"thread_id": session_id}})
        if not state or not state.values:
            raise HTTPException(status_code=404, detail="Session not found.")

        messages = []
        for msg in state.values.get("messages", []):
            messages.append(
                {
                    "role": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content,
                }
            )

        return {
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages,
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Basic RAG PDF API server.")
    parser.add_argument(
        "--host",
        default=os.getenv("RAG_GRAPH_API_HOST", DEFAULT_API_HOST),
        help="Host interface to bind the API server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RAG_GRAPH_API_PORT", str(DEFAULT_API_PORT))),
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
        default=int(os.getenv("RAG_GRAPH_TOP_K", str(DEFAULT_TOP_K))),
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
    parser.add_argument(
        "--database-url",
        default=resolve_database_url(),
        help="Database URL used for authentication data and, when PostgreSQL is selected, chat history.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not resolve_google_api_key():
        parser.error(
            "GEMINI_API_KEY or GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    runtime = RagRuntime(
        model_name=args.model,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_collection_name=args.qdrant_collection_name,
        database_url=args.database_url,
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
