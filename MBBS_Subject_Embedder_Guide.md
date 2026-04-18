# MBBS Subject Atomic Q&A Embedder – Complete Build Guide

> **Reference implementation:** This guide is based on **P.A.E.D.S** (Pediatric Atomic Embedding & Database System), a working system built for the MBBS Pediatrics curriculum. Every step below has been validated against that system. The repository at `R-DAWG-dev/Atomic-curriculum-embedder` is the canonical reference.

---

## Objective
Build a **token‑efficient retrieval system** that answers factual questions from any MBBS subject (e.g., Medicine, Surgery, OB/GYN, Pediatrics, etc.) using the official competency‑based curriculum and a standard textbook. The system will:
- Convert every competency into a natural‑language question.
- Generate short answers (≤20 words) from the textbook.
- Embed all Q&A pairs using a lightweight vector database.
- Expose the embedder as a live API (via ngrok).
- Connect to a free chatbot (Gemini or Groq) that automatically calls the embedder for factual recall.

**Result:** Zero‑cost, instantaneous, textbook‑grounded answers – no LLM hallucinations, no per‑query token fees.

---

## Prerequisites (for the AI to execute)
- Access to the subject's **competency list** (e.g., from NMC or university syllabus). Usually a PDF or structured text.
- Access to a **standard textbook** for that subject (e.g., Bailey & Love for Surgery, Robbins for Pathology, etc.). The AI must be able to query it.
- A free Google account (for Colab, Gemini API) or Groq account.
- Basic familiarity with Python, Google Colab, and Git (optional but helpful).

---

## Step‑by‑step guide

### Step 1: Extract competencies and generate questions
**Input:** PDF or text file with competencies (e.g., "Subject X, Topic 1: …").
**Action:**
1. Parse the competency list. Each competency is a short statement like *"Describe the etiopathogenesis of acute appendicitis."*
2. For each competency, write a **natural‑language question** that a student would ask. Example:
   - Competency: "Define shock and classify it."
   - Question: "What is shock and how is it classified?"
3. Output a Python list of dictionaries:
   ```python
   all_questions = [
       {"competency_id": "PE1.1", "question": "What is the normal heart rate for a newborn?"},
       ...
   ]
   ```

Note: Keep the competency ID for traceability. Aim for one question per competency (unless a competency contains multiple distinct facts, then split).

---

### Step 2: Generate short answers from the textbook

**Input:** The `all_questions` list and the subject textbook (digital text).
**Action:**
For each question, find the most direct, factual answer in the textbook. Compress the answer to ≤20 words – use abbreviations, drop examples, keep only the core fact.
Output a new list:

```python
all_answers = [
    {"competency_id": "PE1.1", "question": "...", "answer": "30-60 breaths/min"},
    ...
]
```

Rules:
- If the textbook does not contain the answer, write `"Not found in textbook"`.
- Preserve the exact order of questions.
- Do not add explanations or extra text.

---

### Step 3: Build the embedder code

Create a Python file named `atomic_embedder.py` with the following class. This code is universal – it works for any subject.

```python
import hashlib
from typing import List, Dict, Tuple
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

class AtomicCurriculumEmbedder:
    def __init__(self, persist_directory: str = "./curriculum_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name='all-MiniLM-L6-v2'
        )
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="atomic_curriculum",
            embedding_function=self.ef
        )
        self.cache: Dict[str, str] = {}

    def embed_curriculum(self, qa_pairs: List[Dict[str, str]]) -> int:
        for pair in qa_pairs:
            q = pair["question"].strip()
            a = pair["answer"].strip()
            doc_id = hashlib.md5(q.encode()).hexdigest()
            self.collection.add(
                documents=[q],
                metadatas=[{"answer": a, "tokens": len(a.split())}],
                ids=[doc_id]
            )
            self.cache[q.lower()] = a
        return len(qa_pairs)

    def ask(self, user_question: str, similarity_threshold: float = 0.7) -> Tuple[str, str]:
        normalized = user_question.lower().strip()
        if normalized in self.cache:
            return self.cache[normalized], "exact cache"
        results = self.collection.query(query_texts=[user_question], n_results=1)
        if not results['ids'][0]:
            return "No relevant answer found.", "no match"
        distance = results['distances'][0][0]
        if distance < similarity_threshold:
            answer = results['metadatas'][0][0]['answer']
            return answer, "vector DB"
        else:
            return "No confident match. Please rephrase.", "low confidence"

    def clear_cache(self):
        self.cache.clear()

    def delete_collection(self):
        self.client.delete_collection("atomic_curriculum")
        self.cache.clear()
```

Place this file in a GitHub repository (or keep locally). Also include `requirements.txt`:

```
sentence-transformers>=2.2.0
chromadb>=0.4.0
```

---

### Step 4: Embed the Q&A pairs in Google Colab

1. Open Google Colab.
2. Clone the repository (or upload the files):
   ```python
   !git clone <your-repo-url>
   %cd <repo-name>
   !pip install -r requirements.txt
   ```
3. Import the embedder and create an instance:
   ```python
   from atomic_embedder import AtomicCurriculumEmbedder
   embedder = AtomicCurriculumEmbedder(persist_directory="./subject_db")
   ```
4. Paste the `all_answers` list (from Step 2) into a cell and run:
   ```python
   embedder.embed_curriculum(all_answers)
   print(f"Embedded {len(all_answers)} Q&A pairs.")
   ```
5. Test with:
   ```python
   answer, src = embedder.ask("What is the normal heart rate for a newborn?")
   print(answer)
   ```

---

### Step 5: Expose the embedder as a live API (ngrok)

In the same Colab notebook, add a cell that installs FastAPI, uvicorn, pyngrok, and nest_asyncio. Then run the server.

Full API server cell:

```python
!pip install fastapi uvicorn pyngrok nest-asyncio

import threading
import nest_asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok

# Kill existing ngrok processes
ngrok.kill()

# Add your ngrok authtoken (get from https://dashboard.ngrok.com)
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer, source = embedder.ask(query.question)
    return {"answer": answer, "source": source}

nest_asyncio.apply()
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_server, daemon=True).start()

public_url = ngrok.connect(8000).public_url
print(f"\n✅ Public API endpoint: {public_url}/ask")
print("Keep this Colab running. Do not close the tab.\n")
```

Note: The user must sign up for a free ngrok account and paste their authtoken.

---

### Step 6: Build the free chatbot client

> **Important:** Use the `google-genai` package (not the deprecated `google-generativeai`). The P.A.E.D.S reference system was migrated to this SDK. All code below reflects the current working version.

#### Option A: Google Gemini (free tier, 1500 requests/day)

**Cell 1 — Install**
```python
!pip install -q -U google-genai requests==2.32.4
```

**Cell 2 — API key (mobile-safe, key never saved in code)**
```python
import os

GEMINI_API_KEY = None

# Strategy 1: Google Colab Secrets (desktop — key icon in left sidebar)
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        print("API key loaded from Colab Secrets.")
except Exception:
    pass

# Strategy 2: Environment variable
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        print("API key loaded from environment variable.")

# Strategy 3: Runtime prompt (key typed at runtime, not stored in code)
if not GEMINI_API_KEY:
    GEMINI_API_KEY = input("Paste your Gemini API key and press Enter: ").strip()
    if GEMINI_API_KEY:
        print("API key accepted.")

if not GEMINI_API_KEY:
    raise ValueError("No API key provided.")
```

**Cell 3 — Configure client**
```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)
print("Gemini client configured.")
```

**Cell 4 — Tool definition + embedder helper**
```python
import requests

EMBEDDER_URL = "https://your-ngrok-url.ngrok-free.dev/ask"  # from Step 5

def ask_embedder(query: str) -> str:
    try:
        resp = requests.post(EMBEDDER_URL, json={"question": query}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("answer", "No answer field in response.")
    except requests.exceptions.Timeout:
        return "Error: The embedder API timed out."
    except requests.exceptions.RequestException as e:
        return f"Error calling embedder API: {e}"

subject_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="subject_fact_lookup",   # rename per subject e.g. surgery_fact_lookup
            description="Get a short, factual answer to a medical question from the curriculum.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The medical question to look up."
                    )
                },
                required=["query"]
            )
        )
    ]
)

print("Tool and helper function defined.")
```

**Cell 5 — Chat configuration**
```python
CHAT_CONFIG = types.GenerateContentConfig(
    tools=[subject_tool],
    system_instruction=(
        "You are a helpful medical education assistant for [SUBJECT NAME]. "
        "You have access to a validated curriculum knowledge base via the "
        "subject_fact_lookup tool. Always use this tool when the user asks a factual "
        "question. Synthesise the retrieved fact into a clear, conversational answer. "
        "If the knowledge base returns no relevant answer, say so honestly."
    )
)
print("Chat configuration ready.")
```

**Cell 6 — Chat turn handler**
```python
def chat_turn(chat, user_message: str) -> str:
    response = chat.send_message(user_message)

    while True:
        if not response.function_calls:
            return response.text or "(No response)"

        fn_response_parts = []
        for fn_call in response.function_calls:
            name = fn_call.name
            args = dict(fn_call.args)
            result = ask_embedder(args.get("query", ""))
            fn_response_parts.append(
                types.Part.from_function_response(
                    name=name,
                    response={"result": result}
                )
            )

        response = chat.send_message(fn_response_parts)

print("Chat handler ready.")
```

**Cell 7 — Conversation loop**
```python
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=CHAT_CONFIG
)

print("Chatbot ready (type 'quit' to exit)\n")

while True:
    try:
        user_input = input("You: ").strip()
    except EOFError:
        break

    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    print("Assistant: ", end="", flush=True)
    try:
        print(chat_turn(chat, user_input))
    except Exception as e:
        print(f"[Error: {e}]")
    print()
```

#### Option B: Groq (14,400 requests/day free tier)

Replace the client with Groq's API. Install `groq` instead of `google-genai`. The tool definition schema and conversation loop structure are nearly identical; only the client initialisation and response parsing differ.

---

### Step 7: Test the complete system

- Keep the embedder Colab running (Step 5).
- Run the chatbot Colab (Step 6).
- Ask a question from the subject (e.g., *"What are the signs of hypovolemic shock?"*).
- The chatbot will automatically call the embedder and return the short answer embedded in a natural response.

---

## File and folder structure (recommended)

```
subject_embedder/
├── atomic_embedder.py
├── requirements.txt
├── README.md
├── .gitignore          # add curriculum_db/ and *.ipynb_checkpoints
└── LICENSE
```

Keep the `all_answers` list and the client notebook separately (or in a `/notebooks` subfolder).

---

## Troubleshooting common issues

| Error | Likely fix |
|-------|-----------|
| `NameError: name 'embedder' is not defined` | Run the cell that creates `embedder` before the API server cell |
| `PyngrokNGrokError: authentication failed` | Sign up at ngrok.com, get authtoken, add `ngrok.set_auth_token(...)` |
| `ConnectionError` when client calls API | Embedder Colab closed or ngrok URL changed – restart and update `EMBEDDER_URL` |
| `ValueError: API key not valid` | Regenerate Gemini/Groq key and re‑enter |
| `Embedder returns "No confident match"` | Rephrase using keywords from the original competency |
| `429 TooManyRequests` | Free tier rate limit hit – wait 60 seconds and retry |
| `DeprecationWarning: google.generativeai` | Switch to `google-genai` package (see Step 6 Option A) |
| `requests` version conflict in Colab | Pin with `requests==2.32.4` in Cell 1 |

---

## Customization for any MBBS subject

| Parameter | P.A.E.D.S value | Change to |
|-----------|----------------|-----------|
| Tool name | `pediatric_fact_lookup` | `surgery_fact_lookup`, `medicine_fact_lookup`, etc. |
| System instruction | Pediatric medicine | Your subject name and scope |
| `persist_directory` | `./curriculum_db` | `./surgery_db`, `./medicine_db`, etc. |
| Textbook | Nelson's Textbook of Pediatrics | Bailey & Love, Robbins, etc. |
| Competency source | NMC Pediatrics syllabus | NMC syllabus for your subject |
| Embedding model | `all-MiniLM-L6-v2` | Upgrade to `all-mpnet-base-v2` for higher accuracy if needed |
| Answer length | ≤20 words | Up to 50 words for more complex facts |

---

## Deliverables for the AI

After following this guide, the AI should produce:

1. A GitHub repository with `atomic_embedder.py`, `requirements.txt`, `.gitignore`, `README.md`.
2. A Colab notebook for embedding and API server (with ngrok).
3. A Colab notebook for the Gemini chatbot client (7 cells as shown in Step 6).
4. The `all_answers` list (questions + short answers) as a Python file or JSON.

The final system will answer any factual question from the subject instantly and for free.

---

*End of guide. Provide this document to any capable AI along with the subject competency list and textbook, and it will replicate the full P.A.E.D.S architecture for your chosen MBBS subject.*
