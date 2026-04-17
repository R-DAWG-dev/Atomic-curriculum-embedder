# Atomic-curriculum-embedder
# Atomic Curriculum Embedder

Convert any textbook or curriculum into **tiny Q&A pairs**. An AI can retrieve answers instantly with **near‑zero token cost** – no LLM generation needed.

## Why?
- Normal RAG: retrieves 500+ tokens + LLM generates 100+ tokens → ~600 tokens per query.
- This method: exact cache hit = 0 tokens; vector search = ~30 tokens (embedding only).

## How it works
1. You provide atomic Q&A pairs (question + short answer <20 tokens).
2. The system embeds each question using `all-MiniLM-L6-v2`.
3. At query time, it either hits an exact‑match cache or does a fast vector search.
4. The short answer is returned directly – **no LLM call**.

## Installation
```bash
pip install sentence-transformers chromadb
