# Gemini RAG Graph

This project is a minimal retrieval-augmented generation example built with LangGraph and Gemini.

## What it does

1. Uploads a PDF and extracts its text.
2. Splits the text into chunks.
3. Embeds the chunks with Gemini embeddings.
4. Stores the chunks in Qdrant.
5. Retrieves the most relevant chunks for a question.
6. Generates an answer with a LangGraph workflow.

In API mode, you upload a PDF first and then query that uploaded file. The indexed chunks live in Qdrant instead of process memory, so the vector data survives server restarts.

## Setup

Install dependencies:

```bash
pip install -e .
```

Set your API key:

```bash
copy .env.example .env
```

Edit `.env` and add your `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

This repo is configured to use the Qdrant Cloud cluster at:

```text
https://2468b96d-f0ae-4d97-a440-74b23e10aa08.us-east-1-1.aws.cloud.qdrant.io:6333
```

Set `QDRANT_API_KEY` in `.env` for that cluster. If you point to a different deployment, keep the same `:6333` REST API port unless your Qdrant setup uses something else.

## API mode

If you want a simple upload-and-query API, run the server:

```bash
rag-graph-api
```

Then open the built-in docs page in your browser:

```bash
http://localhost:8000/docs
```

Use `POST /upload` to choose a PDF file. The upload body is `multipart/form-data` with a `file` field. Then use `POST /query` with JSON like:

```json
{"question":"What does this PDF say?"}
```

If you prefer PowerShell for querying:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/query `
  -ContentType "application/json" `
  -Body (@{ question = "What does this PDF say?" } | ConvertTo-Json)
```

Uploading a new PDF replaces the active document set for queries.

## Graph shape

The LangGraph workflow is intentionally small:

`START -> retrieve -> generate -> END`

That makes it easy to extend later with query rewriting, grading, memory, or tool use.
