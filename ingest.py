"""
ingest.py
---------
Ingestion pipeline for the Telugu RAG system.

Reads all .txt files from the data/ directory, splits them into overlapping
chunks, attaches source metadata, generates multilingual-e5-small embeddings,
and stores everything in ChromaDB — rebuilding the collection from scratch
each run so the index always reflects the current data/ contents.

Run:
    python ingest.py
"""

import os
import re
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Chunk size in characters. 500 chars works well for multilingual-e5-small
# which has a 512-token context window.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_title_from_text(text: str, filename: str) -> str:
    """
    Tries to extract the article title from the first line of the file.
    The fetch_wikipedia.py script writes 'శీర్షిక: <title>' on line 1.
    Falls back to the filename stem if the header is absent.

    Args:
        text:     Full text content of the file.
        filename: Basename of the file (used as fallback).

    Returns:
        Article title string.
    """
    first_line = text.splitlines()[0] if text.strip() else ""

    # Look for the Telugu title header written by fetch_wikipedia.py
    match = re.match(r"శీర్షిక:\s*(.+)", first_line)
    if match:
        return match.group(1).strip()

    # Fallback: derive a readable title from the filename
    stem = os.path.splitext(filename)[0]
    return stem.replace("_", " ")


def load_documents_from_data_dir(data_dir: str) -> list[Document]:
    """
    Loads every .txt file from data_dir using LangChain's TextLoader
    (UTF-8 encoding) and enriches each document's metadata with:
      - source: relative path to the file
      - article_title: human-readable title extracted from the file

    Args:
        data_dir: Path to the directory containing .txt files.

    Returns:
        List of LangChain Document objects.
    """
    all_docs: list[Document] = []

    txt_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".txt")
    )

    if not txt_files:
        print(f"[WARNING] No .txt files found in '{data_dir}'. Run fetch_wikipedia.py first.")
        return all_docs

    print(f"[Ingest] Found {len(txt_files)} text file(s) in '{data_dir}'")

    for filename in txt_files:
        filepath = os.path.join(data_dir, filename)

        # Load the file — TextLoader handles UTF-8 Telugu text correctly
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()

        # Enrich metadata for every page in this file
        for doc in docs:
            title = extract_title_from_text(doc.page_content, filename)
            doc.metadata["source"] = filename          # store just the filename
            doc.metadata["article_title"] = title      # human-readable title

        all_docs.extend(docs)
        print(f"  Loaded: {filename}  (title: '{extract_title_from_text(docs[0].page_content, filename)}')")

    return all_docs


def split_into_chunks(documents: list[Document]) -> list[Document]:
    """
    Splits documents into overlapping text chunks using separators
    appropriate for Telugu text (paragraph breaks, newlines, spaces).

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunk Documents with inherited metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        # Separators tried in order; later ones are fallbacks
        separators=["\n\n", "\n", "।", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Keep separators in the chunks so context isn't lost
        keep_separator=True,
    )

    chunks = splitter.split_documents(documents)
    return chunks


def rebuild_chroma(chroma_dir: str) -> None:
    """
    Deletes the existing ChromaDB directory so the collection is
    rebuilt from scratch on every ingest run. This guarantees the
    index is consistent with the current data/ contents.

    Args:
        chroma_dir: Path to the ChromaDB directory.
    """
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
        print(f"[Ingest] Removed old ChromaDB at '{chroma_dir}'")


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def run_ingestion() -> None:
    """
    Full ingestion pipeline:
      1. Load all .txt files from data/
      2. Split into overlapping chunks
      3. Wipe old ChromaDB collection
      4. Generate embeddings with multilingual-e5-small
      5. Store chunks + metadata in ChromaDB
    """
    print("=" * 60)
    print("Telugu RAG — Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load documents
    documents = load_documents_from_data_dir(DATA_DIR)
    if not documents:
        print("[Ingest] Nothing to ingest. Exiting.")
        return
    print(f"[Ingest] Total documents loaded: {len(documents)}")

    # Step 2: Split into chunks
    chunks = split_into_chunks(documents)
    print(f"[Ingest] Total chunks created: {len(chunks)}")

    # Step 3: Wipe the old vector DB to ensure a clean rebuild
    rebuild_chroma(CHROMA_DIR)

    # Step 4: Load the embedding model
    print(f"[Ingest] Loading embedding model: {EMBEDDING_MODEL}")
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Step 5: Create ChromaDB collection from chunks + embeddings
    print("[Ingest] Generating embeddings and storing in ChromaDB …")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DIR,
    )

    print()
    print("=" * 60)
    print(f"[Ingest] Done. {len(chunks)} chunks stored in '{CHROMA_DIR}'")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
