# Atomic Curriculum Embedder

Convert textbooks or FAQs into atomic Q&A pairs. Retrieve answers instantly with near‑zero token cost – no LLM generation.

## Why
- Standard RAG: ~600 tokens/query (retrieval + generation)
- This embedder: 0 tokens (cache) or ~30 tokens (vector search)

## When to use
- Factual Q&A (math, science, definitions, code snippets)
- FAQs, support docs
- Any domain with short, unambiguous answers

## How it works
1. You provide `(question, short_answer)` pairs (answer ≤20 tokens).
2. System embeds each question using `all-MiniLM-L6-v2`.
3. At query time: exact cache hit → return answer; else vector search → return nearest match.
4. No LLM call – answer is looked up directly.

## Installation
```bash
pip install sentence-transformers chromadb
