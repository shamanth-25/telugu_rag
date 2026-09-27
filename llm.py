"""
llm.py
------
LLM integration layer for the Telugu RAG system.

This module is intentionally decoupled from the retrieval layer so that
you can plug in ANY local or remote model (Ollama, Qwen, Llama, etc.)
without touching retriever.py or app.py.

Current state: the generate_answer() function is a placeholder that
returns the raw retrieved context. Replace the body of generate_answer()
(and optionally the helper loaders below) when you are ready to wire in
a real language model.

Design for future LLM backends
--------------------------------
To add Ollama:
    1. pip install ollama (or langchain-ollama)
    2. Implement _generate_with_ollama() below.
    3. Set ACTIVE_BACKEND = "ollama" in LLMConfig.

To add any OpenAI-compatible endpoint (LM Studio, vLLM, etc.):
    1. pip install openai
    2. Implement _generate_with_openai_compat() below.
    3. Set ACTIVE_BACKEND = "openai_compat" in LLMConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """
    Central configuration for the LLM backend.
    Swap fields here when connecting a real model.
    """
    # Which backend to use: "placeholder" | "ollama" | "openai_compat"
    active_backend: str = "ollama"            # ← Ollama is live

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:0.5b"        # fits in RAM on 8GB systems

    # OpenAI-compatible endpoint settings (LM Studio, vLLM, etc.)
    openai_compat_base_url: str = "http://localhost:1234/v1"
    openai_compat_model: str = "local-model"
    openai_compat_api_key: str = "not-needed"

    # Generation parameters (shared across backends)
    temperature: float = 0.3        # lower = more factual, less hallucination
    max_tokens: int = 150           # keep answers concise; lower = faster on CPU
    request_timeout: int = 300      # seconds before giving up (CPU inference is slow)

    # System prompt template injected before the context.
    # Written bilingually so the model understands the task regardless
    # of whether it leans toward Telugu or English.
    system_prompt: str = (
        "మీరు ఒక తెలుగు సహాయకుడు. క్రింద ఇచ్చిన సందర్భాన్ని మాత్రమే ఉపయోగించి "
        "తెలుగులో సమాధానం ఇవ్వండి. సందర్భంలో సమాచారం లేకపోతే అది చెప్పండి.\n\n"
        "You are a Telugu assistant. Answer ONLY using the provided context. "
        "Reply in Telugu. If the context does not contain the answer, say so clearly.\n"
    )


# ---------------------------------------------------------------------------
# Public API — this is the only function the rest of the app calls
# ---------------------------------------------------------------------------

def generate_answer(question: str, context: str, config: LLMConfig | None = None) -> str:
    """
    Generates a natural-language answer for the given question using
    the retrieved context passages.

    This function acts as the single integration point between the
    retrieval layer and whatever LLM backend is configured.

    Args:
        question: The user's question (Telugu or English).
        context:  Concatenated retrieved passages from ChromaDB.
        config:   LLMConfig instance; uses defaults if None.

    Returns:
        A string answer produced by the LLM (or the placeholder).

    --- HOW TO EXTEND ---
    Replace the body of this function (or the selected backend call)
    with your actual LLM call. The retrieval layer never needs to change.
    """
    if config is None:
        config = LLMConfig()

    backend = config.active_backend

    if backend == "ollama":
        return _generate_with_ollama(question, context, config)

    if backend == "openai_compat":
        return _generate_with_openai_compat(question, context, config)

    # Default: placeholder — returns a structured preview of the context
    return _generate_placeholder(question, context)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _generate_placeholder(question: str, context: str) -> str:
    """
    Placeholder backend: returns the retrieved context as-is.
    Useful for testing the retrieval pipeline before connecting an LLM.
    """
    return (
        f"[Placeholder — no LLM connected]\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}"
    )


def _generate_with_ollama(question: str, context: str, config: LLMConfig) -> str:
    """
    Calls a locally running Ollama instance.

    Install: pip install langchain-ollama
    Start:   ollama serve && ollama pull qwen2.5:7b
    """
    # Lazy import so the app doesn't crash if langchain-ollama isn't installed
    try:
        from langchain_ollama import OllamaLLM  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is not installed. "
            "Run: pip install langchain-ollama"
        ) from exc

    prompt = _build_prompt(question, context, config.system_prompt)

    llm = OllamaLLM(
        base_url=config.ollama_base_url,
        model=config.ollama_model,
        temperature=config.temperature,
        num_predict=config.max_tokens,      # Ollama's token limit param
        request_timeout=config.request_timeout,
    )
    return llm.invoke(prompt)


def _generate_with_openai_compat(question: str, context: str, config: LLMConfig) -> str:
    """
    Calls any OpenAI-compatible endpoint (LM Studio, vLLM, etc.).

    Install: pip install openai
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai package is not installed. Run: pip install openai"
        ) from exc

    prompt = _build_prompt(question, context, config.system_prompt)

    client = OpenAI(
        base_url=config.openai_compat_base_url,
        api_key=config.openai_compat_api_key,
    )
    response = client.chat.completions.create(
        model=config.openai_compat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _build_prompt(question: str, context: str, system_prompt: str) -> str:
    """
    Assembles the full prompt string combining system instructions,
    retrieved context, and the user question.

    Args:
        question:      User question.
        context:       Retrieved passages joined as a string.
        system_prompt: Instructions prepended to every request.

    Returns:
        Formatted prompt string.
    """
    return (
        f"{system_prompt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
