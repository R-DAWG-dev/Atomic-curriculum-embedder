# Project Summary — Atomic Curriculum Embedder + Gemini Chatbot

## Purpose
This repository implements a lightweight pediatric curriculum Q&A system that minimises LLM token usage. It converts textbooks or FAQs into atomic (short) Q&A pairs, embeds them locally with `sentence-transformers`, and retrieves answers via vector similarity search — no LLM generation required per query.

A companion Gemini chatbot (`gemini_chatbot.ipynb`) wraps this system as a tool, allowing a conversational interface over the curriculum.

---

## Repository Structure

```
Atomic-curriculum-embedder/
├── Atomic_embedder.py       # Core embedder class (AtomicCurriculumEmbedder)
├── gemini_chatbot.ipynb     # Gemini 2.0 Flash chatbot with tool calling
├── project_summary.md       # This file
├── Requirements.txt         # sentence-transformers, chromadb
├── README.md
└── LICENSE
```

---

## Embedder API

The `AtomicCurriculumEmbedder` is served as a Flask API (via ngrok). The chatbot notebook POSTs to it.

**Endpoint:** `POST https://feel-dork-entree.ngrok-free.dev/ask`

**Request body:**
```json
{ "question": "What is the normal respiratory rate for a newborn?" }
```

**Response body:**
```json
{ "answer": "Normal respiratory rate for a newborn is 30-60 breaths per minute." }
```

The URL is defined in one place in the notebook (`EMBEDDER_URL` in Cell 4) — update it there if ngrok restarts.

---

## Chatbot Notebook (`gemini_chatbot.ipynb`)

### Known Issue Fixed
Cell-2 originally used `getpass()` to collect the Gemini API key. **`getpass` hangs in non-interactive Jupyter/Colab kernels** because there is no real TTY stdin file descriptor. Fixed with a 3-strategy fallback:

1. **Google Colab Secrets** (preferred): Add key named `GEMINI_API_KEY` via the Colab sidebar (key icon). Loaded with `from google.colab import userdata`.
2. **Environment variable**: Set `GEMINI_API_KEY` before launching Jupyter. Loaded with `os.environ.get("GEMINI_API_KEY")`.
3. **Direct assignment**: Edit Cell-2 and paste the key into `GEMINI_API_KEY = ""`.

### Cell Structure

| Cell | Purpose |
|------|---------|
| 1 | Install `google-generativeai` and `requests` |
| 2 | API key setup (3-strategy fallback — no `getpass`) |
| 3 | Configure Gemini SDK |
| 4 | Define `ask_embedder()` + `pediatric_tool` schema |
| 5 | Create `gemini-2.0-flash` model with tool |
| 6 | `chat_turn()` — handles Gemini function-calling protocol |
| 7 | Interactive conversation loop |

### Tool Calling Flow

```
User message
  → Gemini 2.0 Flash (tool schema registered)
      → emits FunctionCall(pediatric_fact_lookup, {query: "..."})
          → Python calls ask_embedder(query)
              → POST /ask → {"answer": "..."}
          → sends FunctionResponse back to Gemini
              → Gemini synthesises final text answer
  → printed to user
```

---

## Dependencies

**Embedder server** (runs separately, not in the notebook):
- `sentence-transformers>=2.2.0`
- `chromadb>=0.4.0`

**Chatbot notebook** (installed in Cell 1):
- `google-generativeai` (Gemini SDK)
- `requests`

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Cell-2 hangs indefinitely | `getpass` with no TTY | Use Colab Secrets or env var instead |
| `ConnectionError` or `Timeout` in chat | ngrok tunnel restarted | Update `EMBEDDER_URL` in Cell 4 |
| `No answer field in response` | API returned unexpected shape | Check Flask server is running and healthy |
| Gemini never calls the tool | Question not clearly pediatric | Rephrase or adjust `system_instruction` in Cell 5 |
| `Part.from_function_response` import error | SDK version mismatch | Upgrade: `pip install -U google-generativeai` |
