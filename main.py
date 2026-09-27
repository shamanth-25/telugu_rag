"""
main.py
-------
Run this single file to use the Telugu RAG system.

It will:
  1. Load the ChromaDB vector store
  2. Accept your Telugu question
  3. Retrieve the top-3 relevant chunks
  4. Send them to Qwen 2.5 1.5B via Ollama
  5. Print a Telugu answer

Run:
    python main.py
"""

from retriever import TeluguRetriever
from llm import generate_answer, LLMConfig

# ── Config ────────────────────────────────────────────────────────────────
config = LLMConfig()

# ── Banner ────────────────────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════╗")
print("║       తెలుగు RAG సిస్టమ్                ║")
print("║       Telugu RAG System                  ║")
print(f"║       Model: {config.ollama_model:<27s}║")
print("╚══════════════════════════════════════════╝")
print()

# ── Load retriever once ───────────────────────────────────────────────────
print("Loading ... (first time may take a few seconds)\n")
retriever = TeluguRetriever()
print("Ready! Type your question below.\n")

# ── Interactive loop ──────────────────────────────────────────────────────
while True:
    try:
        question = input("మీ ప్రశ్న (Your question): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nమళ్ళీ కలుద్దాం!  (Goodbye!)")
        break

    if not question:
        continue

    # Step 1: Retrieve relevant chunks
    print("\n🔍 Searching ...\n")
    chunks = retriever.search(question, k=3)

    if not chunks:
        print("❌ No relevant information found. Try a different question.\n")
        continue

    # Step 2: Show sources
    print("📄 Sources found:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   [{i}] {chunk.title}  (score: {chunk.score})")
    print()

    # Step 3: Build context and generate answer
    context = "\n\n".join(c.content for c in chunks)
    print("🤖 Generating answer ...\n")
    answer = generate_answer(question, context, config)

    # Step 4: Print answer
    print("─" * 50)
    print("సమాధానం (Answer):")
    print()
    print(answer)
    print("─" * 50)
    print()
