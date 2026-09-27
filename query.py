"""
query.py
--------
Command-line interface for querying the Telugu RAG vector store.

Updated to use the shared TeluguRetriever class from retriever.py,
so retrieval logic is not duplicated.

Run:
    python query.py
"""

from retriever import TeluguRetriever


def main() -> None:
    """Interactive CLI query loop."""
    print("=" * 60)
    print("Telugu RAG — Query CLI")
    print("Type your question in Telugu or English. Ctrl+C to exit.")
    print("=" * 60)

    # Initialise retriever (loads embedding model + connects to ChromaDB)
    retriever = TeluguRetriever()

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue

        # Retrieve top-3 relevant chunks
        results = retriever.search(question, k=3)

        if not results:
            print("No results found. Make sure ingest.py has been run.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Top {len(results)} result(s) for: {question}")
        print("=" * 60)

        for idx, chunk in enumerate(results, start=1):
            print(f"\n[{idx}] Source: {chunk.source}  |  Title: {chunk.title}  |  Score: {chunk.score}")
            print("-" * 40)
            print(chunk.content)

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
