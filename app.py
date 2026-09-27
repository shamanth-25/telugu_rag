"""
app.py
------
FastAPI service for the Telugu RAG system.

Exposes a single POST /ask endpoint that:
  1. Receives a Telugu (or any language) question
  2. Searches ChromaDB for the top-3 most relevant chunks
  3. Returns the retrieved passages + an optional LLM-generated answer

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm import LLMConfig, generate_answer
from retriever import TeluguRetriever

# LLM config initialised once at module load (backend = ollama by default)
_llm_config = LLMConfig()

# ---------------------------------------------------------------------------
# Application lifespan — load heavy resources once at startup
# ---------------------------------------------------------------------------

# Global retriever instance (initialised in lifespan, reused per request)
_retriever: TeluguRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager.
    The retriever (embedding model + ChromaDB connection) is created once
    when the server starts and shared across all requests.
    """
    global _retriever
    print("[App] Initialising TeluguRetriever …")
    _retriever = TeluguRetriever()
    print("[App] Retriever ready. Server is accepting requests.")
    yield
    # Cleanup (if needed) goes here
    print("[App] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Telugu RAG API",
    description=(
        "Retrieval-Augmented Generation service for Telugu language questions. "
        "Searches a ChromaDB vector store built from Telugu Wikipedia articles "
        "and returns the most relevant passages."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Allow all origins for local development; tighten this for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Schema for the /ask request body."""
    question: str = Field(
        ...,
        min_length=2,
        description="The question to ask in Telugu or English.",
        examples=["రుద్రమ దేవి ఎవరు?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top document chunks to retrieve (1–10).",
    )
    generate: bool = Field(
        default=True,
        description="If true, pass retrieved context to Ollama for a generated Telugu answer.",
    )


class SourceItem(BaseModel):
    """A single retrieved passage with its provenance."""
    file: str = Field(description="Source filename in the data/ directory.")
    title: str = Field(description="Human-readable article title.")
    content: str = Field(description="Retrieved text passage.")
    score: float | None = Field(
        default=None,
        description="Similarity distance (lower = more relevant).",
    )


class AskResponse(BaseModel):
    """Schema for the /ask response body."""
    question: str = Field(description="The original question as received.")
    sources: list[SourceItem] = Field(description="Top-k retrieved passages.")
    answer: str | None = Field(
        default=None,
        description="LLM-generated answer (None when generate=False or no LLM configured).",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
async def root() -> dict[str, str]:
    """Returns a simple health-check message."""
    return {"status": "ok", "service": "Telugu RAG API", "version": "2.0.0"}


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about Telugu history/culture",
    response_description="Retrieved passages and optional generated answer",
)
async def ask(request: AskRequest) -> AskResponse:
    """
    Main RAG endpoint.

    - Searches the ChromaDB vector store for the top-k chunks most
      relevant to the question.
    - Optionally generates a natural-language answer using the configured
      LLM backend (set `generate=true` in the request body).
    """
    if _retriever is None:
        # Should never happen after lifespan initialises the retriever,
        # but guard defensively.
        raise HTTPException(
            status_code=503,
            detail="Retriever not initialised. The server may still be starting up.",
        )

    # Step 1: Retrieve top-k relevant chunks from ChromaDB
    chunks = _retriever.search(request.question, k=request.top_k)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail=(
                "No relevant passages found. "
                "Make sure the database has been populated by running ingest.py."
            ),
        )

    # Step 2: Build the sources list for the response
    sources = [
        SourceItem(
            file=chunk.source,
            title=chunk.title,
            content=chunk.content,
            score=chunk.score,
        )
        for chunk in chunks
    ]

    # Step 3 (optional): Generate a Telugu answer via Ollama
    answer: str | None = None
    if request.generate:
        context = _retriever.as_context_string(request.question, k=request.top_k)
        answer = generate_answer(request.question, context, _llm_config)

    return AskResponse(
        question=request.question,
        sources=sources,
        answer=answer,
    )
