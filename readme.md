# Gemini RAG Graph

A retrieval-augmented generation API built with LangGraph, Gemini, Qdrant, and PostgreSQL.

## What it does

1. Uploads a PDF and extracts its text.
2. Splits the text into chunks.
3. Embeds the chunks with Gemini embeddings.
4. Stores the chunks in Qdrant.
5. Routes queries between RAG (PDF-specific) and general conversation.
6. Persists conversation history via PostgreSQL checkpointing.
7. Authenticates users with JWT bearer tokens.

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 16 (for conversation checkpointing)
- A Qdrant instance (cloud or local)

### Install

```bash
pip install -e .
```

### Configure

```bash
copy .env.example .env
```

Edit `.env` with your keys:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `QDRANT_URL` | Qdrant cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `POSTGRES_URL` | PostgreSQL connection string |

### Docker (optional)

```bash
docker compose up --build
```

This starts both the API and a PostgreSQL instance with automatic checkpointing.

## API mode

Run the server:

```bash
rag-graph-api
```

Open the docs:

```
http://localhost:8000/docs
```

### Auth flow

1. **Sign up** — `POST /auth/signup` with `{"email": "...", "password": "..."}`
2. **Log in** — `POST /auth/login` with form fields `username` (your email) and `password`
3. Click **Authorize** in Swagger and paste the returned token
4. All other endpoints now require a valid Bearer token

### Usage

Create a new session:

```
GET /session
```

Upload a PDF:

```
POST /upload  (multipart/form-data, file field)
```

Query with a session:

```json
{"question": "What does this PDF say?", "session_id": "<id from /session>"}
```

Query history is persisted per session in PostgreSQL.

## Graph shape

```
START -> router -> retrieve -> rag_generate -> END
               -> general_generate --------> END
```

The router node decides whether to use RAG (PDF context) or general knowledge based on the user's message.
