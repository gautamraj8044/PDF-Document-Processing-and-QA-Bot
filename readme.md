# Gemini RAG Graph

A LangGraph-powered RAG API with Gemini embeddings, Qdrant vector storage, PostgreSQL checkpointing, and JWT authentication.

## Stack

- **LLM** — Google Gemini (configurable model)
- **Orchestration** — LangGraph state machine with routing
- **Vector store** — Qdrant (cloud or local)
- **Conversation memory** — PostgreSQL via `langgraph-checkpoint-postgres`
- **Auth** — JWT with bcrypt, user store in PostgreSQL
- **API** — FastAPI with auto-generated Swagger docs

## Quick start

```bash
pip install -e .
copy .env.example .env
```

Edit `.env` with your Gemini API key and Qdrant credentials.

### Docker (includes PostgreSQL)

```bash
docker compose up --build
```

## Usage

### 1. Start the server

```bash
rag-graph-api
```

Open http://localhost:8000/docs

### 2. Auth flow

| Step | Endpoint | Body |
|---|---|---|
| Sign up | `POST /auth/signup` | `{"email": "...", "password": "..."}` |
| Log in | `POST /auth/login` | Form: `username` (email), `password` |
| Get profile | `GET /auth/me` | Bearer token |

After login, paste the `access_token` in Swagger's **Authorize** button.

### 3. Query flow

```
GET /session              → get a session_id
POST /upload (PDF file)   → upload a document
POST /query               → ask a question with session_id
GET /history/{session_id} → view conversation history
```

Query request:

```json
{
  "question": "What does this PDF say?",
  "session_id": "<uuid>:<uuid>"
}
```

## Environment

Key variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Chat model |
| `GEMINI_EMBEDDING_MODEL` | `models/embedding-001` | Embedding model |
| `QDRANT_URL` | (built-in) | Qdrant cluster URL |
| `QDRANT_COLLECTION_NAME` | `rag_graph_documents` | Collection for document chunks |
| `POSTGRES_URL` | — | PostgreSQL connection string |
| `JWT_SECRET_KEY` | `change-me-in-production` | Token signing secret |
| `RAG_GRAPH_TOP_K` | `3` | Retrieved document chunks per query |

## Graph

```
START → router ──→ retrieve ──→ rag_generate ──→ END
               └─→ general_generate ──────────→ END
```

The **router** decides whether the question relates to the uploaded PDF (RAG path) or is general conversation. History is persisted via PostgreSQL checkpointing.
