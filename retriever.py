"""
retriever.py
------------
Encapsulates all ChromaDB retrieval logic in a single reusable class.

This module is intentionally kept independent of FastAPI and LLM layers
so that the retrieval pipeline can be swapped, extended, or unit-tested
without touching higher-level code.

Usage (programmatic):
    from retriever import TeluguRetriever

    retriever = TeluguRetriever()
    results = retriever.search("రుద్రమ దేవి ఎవరు?", k=3)
    for r in results:
        print(r["content"], r["source"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Same embedding model used during ingestion — must match exactly
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Path to the persistent ChromaDB directory
CHROMA_DIR = "chroma_db"

# Default number of results to retrieve
DEFAULT_K = 3


# ---------------------------------------------------------------------------
# Data model for a single retrieved result
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with its metadata."""
    content: str          # The raw text of the chunk
    source: str           # Filename the chunk came from
    title: str            # Article title extracted from metadata
    score: float | None   # Similarity distance (lower = more similar), or None


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class TeluguRetriever:
    """
    Loads the multilingual-e5-small embedding model once and provides
    a simple .search() interface over the persisted ChromaDB collection.

    The class is designed to be instantiated once (e.g., at FastAPI startup)
    and reused across many requests.
    """

    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        """
        Initialises the embedding function and connects to the ChromaDB store.

        Args:
            chroma_dir: Path to the persisted ChromaDB directory.
            model_name: HuggingFace model name for embeddings.
        """
        print(f"[Retriever] Loading embedding model: {model_name}")

        # Load the multilingual embedding model.
        # Setting model_kwargs device to 'cpu' ensures compatibility on
        # machines without a CUDA GPU; change to 'cuda' if available.
        self._embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Connect to the existing ChromaDB collection (read-only intent)
        self._db = Chroma(
            persist_directory=chroma_dir,
            embedding_function=self._embedding,
        )

        print(f"[Retriever] Connected to ChromaDB at: {chroma_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, question: str, k: int = DEFAULT_K) -> list[RetrievedChunk]:
        """
        Performs a similarity search against the ChromaDB collection
        and returns the top-k matching chunks with metadata.

        Args:
            question: The natural-language question in Telugu (or any language).
            k: Number of top results to retrieve.

        Returns:
            A list of RetrievedChunk objects sorted by relevance.
        """
        # similarity_search_with_score returns (Document, distance) pairs.
        # Lower distance = more similar for cosine distance.
        raw_results: list[tuple[Any, float]] = (
            self._db.similarity_search_with_score(question, k=k)
        )

        results: list[RetrievedChunk] = []
        for doc, score in raw_results:
            metadata = doc.metadata or {}
            results.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source=metadata.get("source", "unknown"),
                    title=metadata.get("article_title", "unknown"),
                    score=round(score, 4),
                )
            )

        return results

    def as_context_string(self, question: str, k: int = DEFAULT_K) -> str:
        """
        Convenience method: retrieves chunks and returns them as a single
        concatenated string suitable for passing to an LLM as context.

        Args:
            question: The user question.
            k: Number of chunks to retrieve.

        Returns:
            Multi-paragraph string of retrieved passages.
        """
        chunks = self.search(question, k=k)
        passages = [chunk.content for chunk in chunks]
        return "\n\n---\n\n".join(passages)
