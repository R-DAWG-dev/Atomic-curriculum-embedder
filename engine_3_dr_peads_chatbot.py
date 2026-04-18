"""
Engine 3 — Dr. P.E.A.D.S. Universal LLM Adapter
Wraps any supported LLM (Gemini, OpenAI, Claude) with the Dr. P.E.A.D.S.
persona and wires up automatic tool-calling to the P.A.E.D.S. curriculum API.

Usage (Colab / script):
    from engine_3_dr_peads_chatbot import build_bot, run_chat

    bot = build_bot("gemini")    # or "openai" / "claude"
    run_chat(bot)
"""

from __future__ import annotations

import json
import os

from engine_2_api_link import ask_paeds, get_api_key

# ---------------------------------------------------------------------------
# Shared persona & tool definition
# ---------------------------------------------------------------------------

DR_PEADS_SYSTEM = (
    "You are Dr. P.E.A.D.S. (Paediatric Education and Diagnostic Support), "
    "a warm, authoritative paediatric AI assistant. "
    "You answer clinical and academic paediatric questions with precision and "
    "compassion, drawing exclusively on validated curriculum data retrieved via "
    "your knowledge tool. "
    "When a question is outside paediatrics, politely redirect the user. "
    "Never guess — if the curriculum returns no confident answer, say so clearly "
    "and suggest the user consult a specialist."
)

_TOOL_NAME = "paeds_fact_lookup"
_TOOL_DESC = (
    "Retrieve a short, validated answer from the P.E.A.D.S. paediatric "
    "curriculum database. Always call this tool before answering any "
    "paediatric clinical or factual question."
)
_TOOL_PARAMS = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "The exact paediatric question to look up.",
        }
    },
    "required": ["question"],
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _DrPeadsBase:
    def __init__(self):
        self._paeds_key = get_api_key()

    def _lookup(self, question: str) -> str:
        print(f"  [P.E.A.D.S. lookup] {question}")
        return ask_paeds(question, api_key=self._paeds_key)

    def chat(self, user_message: str) -> str:
        raise NotImplementedError

    def reset(self):
        """Clear conversation history to start a fresh session."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Gemini adapter
# ---------------------------------------------------------------------------

class DrPeadsGemini(_DrPeadsBase):
    """Dr. P.E.A.D.S. powered by Google Gemini."""

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        super().__init__()
        import google.generativeai as genai

        key = api_key or os.environ.get("GOOGLE_API_KEY") or _colab_secret("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found.\n"
                "Set the GOOGLE_API_KEY environment variable or add it to Colab Secrets."
            )
        genai.configure(api_key=key)
        self._genai = genai
        self._model_name = model
        self._model = genai.GenerativeModel(
            model_name=model,
            system_instruction=DR_PEADS_SYSTEM,
            tools=[{"name": _TOOL_NAME, "description": _TOOL_DESC, "parameters": _TOOL_PARAMS}],
        )
        self._history: list = []

    def chat(self, user_message: str) -> str:
        self._history.append({"role": "user", "parts": [user_message]})
        response = self._model.generate_content(self._history)
        part = response.candidates[0].content.parts[0]

        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            if fc.name == _TOOL_NAME:
                fact = self._lookup(fc.args["question"])
                # Feed tool result back into the conversation
                self._history.append({"role": "model", "parts": [part]})
                self._history.append({
                    "role": "function",
                    "parts": [{
                        "function_response": {
                            "name": _TOOL_NAME,
                            "response": {"answer": fact},
                        }
                    }],
                })
                final = self._model.generate_content(self._history)
                reply = final.text
            else:
                reply = getattr(response, "text", str(response))
        else:
            reply = getattr(response, "text", str(response))

        self._history.append({"role": "model", "parts": [reply]})
        return reply

    def reset(self):
        self._history = []


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------

class DrPeadsOpenAI(_DrPeadsBase):
    """Dr. P.E.A.D.S. powered by OpenAI GPT."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        super().__init__()
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY") or _colab_secret("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found.\n"
                "Set the OPENAI_API_KEY environment variable or add it to Colab Secrets."
            )
        self._client = OpenAI(api_key=key)
        self._model = model
        self._tools = [{"type": "function", "function": {
            "name": _TOOL_NAME,
            "description": _TOOL_DESC,
            "parameters": _TOOL_PARAMS,
        }}]
        self._messages: list = [{"role": "system", "content": DR_PEADS_SYSTEM}]

    def chat(self, user_message: str) -> str:
        self._messages.append({"role": "user", "content": user_message})
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            tools=self._tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            self._messages.append(msg)
            for tc in msg.tool_calls:
                if tc.function.name == _TOOL_NAME:
                    q = json.loads(tc.function.arguments).get("question", "")
                    fact = self._lookup(q)
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": fact,
                    })
            final = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
            )
            reply = final.choices[0].message.content
        else:
            reply = msg.content

        self._messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self._messages = [{"role": "system", "content": DR_PEADS_SYSTEM}]


# ---------------------------------------------------------------------------
# Claude (Anthropic) adapter
# ---------------------------------------------------------------------------

class DrPeadsClaude(_DrPeadsBase):
    """Dr. P.E.A.D.S. powered by Anthropic Claude."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-6"):
        super().__init__()
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or _colab_secret("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key not found.\n"
                "Set the ANTHROPIC_API_KEY environment variable or add it to Colab Secrets."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self._tool_def = {
            "name": _TOOL_NAME,
            "description": _TOOL_DESC,
            "input_schema": _TOOL_PARAMS,
        }
        self._messages: list = []

    def chat(self, user_message: str) -> str:
        import anthropic

        self._messages.append({"role": "user", "content": user_message})
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=DR_PEADS_SYSTEM,
            tools=[self._tool_def],
            messages=self._messages,
        )

        if response.stop_reason == "tool_use":
            self._messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == _TOOL_NAME:
                    fact = self._lookup(block.input.get("question", ""))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": fact,
                    })
            self._messages.append({"role": "user", "content": tool_results})
            final = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=DR_PEADS_SYSTEM,
                tools=[self._tool_def],
                messages=self._messages,
            )
            reply = final.content[0].text
        else:
            reply = response.content[0].text

        self._messages.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self._messages = []


# ---------------------------------------------------------------------------
# Factory & chat loop
# ---------------------------------------------------------------------------

PROVIDERS = {
    "gemini": DrPeadsGemini,
    "openai": DrPeadsOpenAI,
    "claude": DrPeadsClaude,
}


def build_bot(provider: str = "gemini", **kwargs) -> _DrPeadsBase:
    """
    Instantiate Dr. P.E.A.D.S. for the chosen LLM provider.

    Args:
        provider: "gemini" | "openai" | "claude"
        **kwargs: forwarded to the adapter (e.g. model="gpt-4o", api_key="...")

    Returns:
        A ready-to-use Dr. P.E.A.D.S. chatbot instance.

    Example:
        bot = build_bot("gemini")
        print(bot.chat("What is the normal heart rate for a toddler?"))
    """
    key = provider.lower()
    if key not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}"
        )
    print(f"[OK]   Dr. P.E.A.D.S. initialised with {provider.capitalize()}.")
    return PROVIDERS[key](**kwargs)


def run_chat(bot: _DrPeadsBase, provider_label: str = ""):
    """
    Start an interactive terminal/Colab chat session with Dr. P.E.A.D.S.
    Type 'quit' or 'exit' to end. Type 'reset' to clear history.
    """
    label = f" ({provider_label})" if provider_label else ""
    print(f"\n{'='*55}")
    print(f"  Dr. P.E.A.D.S.{label}")
    print(f"  Paediatric Education and Diagnostic Support")
    print(f"{'='*55}")
    print("  Type 'quit' to exit | 'reset' to clear history\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\nDr. P.E.A.D.S.: Goodbye! Keep learning and stay curious.")
            break
        if user_input.lower() == "reset":
            bot.reset()
            print("[INFO] Conversation history cleared.\n")
            continue

        reply = bot.chat(user_input)
        print(f"\nDr. P.E.A.D.S.: {reply}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Change "gemini" to "openai" or "claude" to switch providers.
    bot = build_bot("gemini")
    run_chat(bot, provider_label="Gemini")
