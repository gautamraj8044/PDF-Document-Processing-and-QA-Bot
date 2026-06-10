# Basic RAG

A practical retrieval-augmented generation system built with FastAPI, LangGraph, Gemini, and Qdrant.

## What It Does

- Upload a PDF, extract its text, split it into chunks, and store embeddings in Qdrant.
- Route each question either to document-grounded RAG or to general chat.
- Authenticate users with JWT.
- Track conversation sessions and history.
- Use local-first defaults so the project runs without hidden infrastructure.

## Architecture

- FastAPI for the HTTP API
- LangGraph for the routing and answer flow
- Gemini for chat and embeddings
- Qdrant for vector search
- SQLAlchemy for user storage

## Setup

1. Install dependencies:

   ```bash
   pip install -e .
   ```

2. Create your environment file:

   ```bash
   copy .env.example .env
   ```

3. Add your `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

4. Start Qdrant and the API:

   ```bash
   docker compose up --build
   ```

   Docker starts a local Qdrant instance plus the API. If you want to run the API without Docker, start Qdrant separately and then run:

   ```bash
   rag-graph-api
   ```

5. Open the API docs:

   ```text
   http://localhost:8000/docs
   ```

`DATABASE_URL` is the primary setting for auth storage and history. `POSTGRES_URL` is kept as a legacy fallback, but `DATABASE_URL` wins if both are present.

If the configured database is unreachable at startup, the server falls back to local SQLite for auth so the API can still boot in a development environment. Conversation history uses an in-memory checkpoint in that case.

## API Flow

1. `POST /auth/signup`
2. `POST /auth/login`
3. `POST /upload`
4. `GET /session`
5. `POST /query`
6. `GET /history/{session_id}`

`/query` also works before a PDF is uploaded. In that case the router falls back to general chat.

## Configuration Notes

- `QDRANT_URL` defaults to `http://localhost:6333`.
- `DATABASE_URL` should point to a reachable database if you want persistent auth data.
- `POSTGRES_URL` is supported for older environments, but it should not be the primary setting in new setups.
- `JWT_SECRET_KEY` should be changed before any real deployment.

## Example Query

```json
{"question": "What does the PDF say about the main goal?"}
```

PowerShell example:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/query `
  -ContentType "application/json" `
  -Body (@{ question = "What does the PDF say about the main goal?" } | ConvertTo-Json)
```

## Resume Bullets

- Built a LangGraph-based RAG API that routes between document-aware answers and general knowledge responses.
- Implemented PDF ingestion, chunking, Gemini embeddings, Qdrant retrieval, JWT auth, and session history.
- Designed local-first defaults with optional PostgreSQL persistence for easy development and deployment.
