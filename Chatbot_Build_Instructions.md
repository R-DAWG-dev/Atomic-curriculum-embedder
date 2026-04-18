# Step‑by‑Step Instructions for Building a Chatbot with P.A.E.D.S. API

## Overview
You will build a conversational chatbot that answers pediatric questions by querying a live API endpoint. The API returns short, textbook‑grounded answers. Your chatbot will use function calling (tools) to decide when to call the API and will present answers naturally.

**Prerequisites:**
- Access to the P.A.E.D.S. API endpoint: `https://feel-dork-entree.ngrok-free.dev/ask`
- A free Gemini API key (or any LLM with function calling). This guide uses Gemini.
- Python environment (Google Colab recommended).

---

## Step 1: Set Up Your Environment

### 1.1 Open Google Colab
- Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### 1.2 Install Required Libraries
Run this cell:
```python
!pip install -q -U google-generativeai requests
```

### 1.3 Store Your Gemini API Key Securely

- In the left sidebar, click the key icon (Secrets).
- Add a new secret named `GOOGLE_API_KEY` and paste your key.
- Toggle **Notebook access** ON.
- In a code cell, retrieve the key:

```python
from google.colab import userdata
GEMINI_API_KEY = userdata.get('GOOGLE_API_KEY')
```

---

## Step 2: Define the API Call Function

Create a function that sends a question to the P.A.E.D.S. API and returns the answer.

```python
import requests

PEDS_API_URL = "https://feel-dork-entree.ngrok-free.dev/ask"

def ask_paeds(question: str) -> str:
    try:
        response = requests.post(PEDS_API_URL, json={"question": question}, timeout=10)
        if response.status_code == 200:
            return response.json().get("answer", "No answer field in response.")
        else:
            return f"API error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Connection error: {e}"
```

Test it:

```python
print(ask_paeds("What is the normal respiratory rate for a newborn?"))
# Expected output: "30-60 breaths/min"
```

---

## Step 3: Set Up Gemini with Tool Calling

### 3.1 Configure Gemini

```python
import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)
```

### 3.2 Define the Tool (Function Declaration)

```python
pediatric_tool = {
    "name": "pediatric_fact_lookup",
    "description": "Get a short, factual answer to a pediatric question from a validated curriculum.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The pediatric question to look up (e.g., 'What is the normal respiratory rate for a newborn?')"
            }
        },
        "required": ["question"]
    }
}
```

### 3.3 Create the Chat Model with Automatic Function Calling

```python
model = genai.GenerativeModel('gemini-2.0-flash', tools=[pediatric_tool])
chat = model.start_chat(enable_automatic_function_calling=True)
```

---

## Step 4: Build the Chat Loop

The loop will:

- Accept user input.
- Send it to Gemini.
- If Gemini decides to call `pediatric_fact_lookup`, the library automatically invokes the function (you must provide a mapping).

> **Note:** `enable_automatic_function_calling=True` requires you to register a function handler. We'll use a simpler manual approach below to have full control.

**Simpler manual approach (recommended):**
We will not use automatic calling; instead, we will manually check the response and call the API ourselves.

```python
print("P.A.E.D.S. Chatbot (type 'quit' to exit)")
messages = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
    messages.append({"role": "user", "content": user_input})

    # Send to Gemini with tools
    response = model.generate_content(
        contents=messages,
        tools=[pediatric_tool],
        tool_config={"function_calling": "auto"}
    )

    # Check if Gemini wants to call the tool
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call.name == "pediatric_fact_lookup":
            question = function_call.args["question"]
            print(f"[P.A.E.D.S. lookup: {question}]")
            answer = ask_paeds(question)
            # Append assistant's function call and the result as a function response
            messages.append({"role": "assistant", "content": [function_call]})
            messages.append({
                "role": "function",
                "name": "pediatric_fact_lookup",
                "content": answer
            })
            # Get final response from Gemini
            final_response = model.generate_content(messages)
            print(f"Bot: {final_response.text}")
            messages.append({"role": "assistant", "content": final_response.text})
        else:
            # Unknown function – just echo the text
            print(f"Bot: {response.text}")
            messages.append({"role": "assistant", "content": response.text})
    else:
        # No function call – just print the text
        print(f"Bot: {response.text}")
        messages.append({"role": "assistant", "content": response.text})
```

---

## Step 5: Run and Test

Run the entire notebook. You should see the prompt `You:`.

Example interaction:

```
You: What is the normal heart rate for a newborn?
[P.A.E.D.S. lookup: What is the normal heart rate for a newborn?]
Bot: The normal heart rate for a newborn is 120–160 beats per minute.
```

---

## Step 6: Optional Enhancements

- **Add conversation memory** – the `messages` list already stores history.
- **Add error handling** – if the API fails, Gemini can say "I'm having trouble fetching the answer."
- **Use a different LLM** – replace Gemini with OpenAI GPT‑4 or Claude by changing the API calls and tool definition.

---

## Step 7: Deploy as a Persistent Chatbot (Optional)

- Convert the notebook to a Python script and run on a server.
- Wrap the loop in a web framework (e.g., Gradio, Streamlit) for a UI.

---

## Troubleshooting Tips

| Problem | Likely Fix |
|---|---|
| `ConnectionError` to API | The ngrok URL may have changed. Update `PEDS_API_URL`. |
| `KeyError: 'answer'` | API returned unexpected JSON – check the endpoint is correct. |
| Gemini not calling the tool | Ensure the tool definition matches exactly. Try `tool_config={"function_calling": "any"}`. |
| `AttributeError: 'GenerateContentResponse' object has no attribute 'candidates'` | The response structure may differ – use `response.text` if no function call. |

---

You have now built a fully functional pediatric chatbot that uses your own curriculum as the knowledge base. No LLM training, no fine‑tuning – just a tool call to a fast embedder API.
