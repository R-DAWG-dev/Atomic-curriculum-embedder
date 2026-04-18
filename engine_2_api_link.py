"""
Engine 2 — P.A.E.D.S. API Link & Key Manager
Handles API key retrieval from multiple sources, connection health checks,
and the single query interface used by Engine 3.
"""

import os
import requests

PAEDS_API_URL = "https://feel-dork-entree.ngrok-free.dev/ask"
_API_KEY_ENV_VAR = "PAEDS_API_KEY"


# ---------------------------------------------------------------------------
# Key retrieval
# ---------------------------------------------------------------------------

def get_api_key() -> str | None:
    """
    Retrieve the P.A.E.D.S. API key from the first available source:
      1. Environment variable  PAEDS_API_KEY
      2. Google Colab secret   PAEDS_API_KEY
      3. Manual prompt (interactive fallback)
    Returns None if no key is found and the prompt is declined.
    """
    # 1. Already in environment
    key = os.environ.get(_API_KEY_ENV_VAR)
    if key:
        print(f"[OK]   API key loaded from environment ({len(key)} chars).")
        return key

    # 2. Google Colab Secrets panel
    try:
        from google.colab import userdata
        key = userdata.get(_API_KEY_ENV_VAR)
        if key:
            os.environ[_API_KEY_ENV_VAR] = key          # cache for the session
            print(f"[OK]   API key loaded from Colab Secrets ({len(key)} chars).")
            return key
    except Exception:
        pass

    # 3. Interactive prompt (notebooks / terminals)
    try:
        import getpass
        key = getpass.getpass(
            f"Enter your P.A.E.D.S. API key (or press Enter to skip): "
        ).strip()
        if key:
            os.environ[_API_KEY_ENV_VAR] = key
            print("[OK]   API key saved to session environment.")
            return key
    except Exception:
        pass

    print("[WARN] No API key found. Requests will be sent without authentication.")
    return None


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def health_check(api_key: str = None) -> bool:
    """
    Confirm the P.A.E.D.S. API endpoint is reachable.
    Returns True if the service responds, False otherwise.
    """
    headers = _auth_headers(api_key)

    # Try a lightweight health endpoint first
    health_url = PAEDS_API_URL.replace("/ask", "/health")
    try:
        r = requests.get(health_url, headers=headers, timeout=5)
        if r.status_code < 500:
            print(f"[OK]   P.A.E.D.S. API reachable (GET /health → {r.status_code}).")
            return True
    except requests.exceptions.ConnectionError:
        pass
    except Exception:
        pass

    # Fallback: send a benign test question
    try:
        r = requests.post(
            PAEDS_API_URL,
            json={"question": "ping"},
            headers=headers,
            timeout=8,
        )
        if r.status_code < 500:
            print(f"[OK]   P.A.E.D.S. API reachable (POST /ask → {r.status_code}).")
            return True
        print(f"[WARN] API responded with status {r.status_code}.")
        return True   # reachable, even if the answer is odd
    except requests.exceptions.ConnectionError:
        print(
            "[FAIL] Cannot reach P.A.E.D.S. API.\n"
            "       Check that the ngrok tunnel is running and update PAEDS_API_URL."
        )
    except requests.exceptions.Timeout:
        print("[FAIL] API request timed out.")
    except Exception as exc:
        print(f"[ERROR] Health check: {exc}")

    return False


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def ask_paeds(question: str, api_key: str = None) -> str:
    """
    Send a question to the P.A.E.D.S. API and return the answer string.
    Safe to call from Engine 3 — never raises, always returns a string.
    """
    headers = _auth_headers(api_key)
    try:
        response = requests.post(
            PAEDS_API_URL,
            json={"question": question},
            headers=headers,
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("answer") or data.get("result") or str(data)
        return f"API error {response.status_code}: {response.text[:200]}"
    except requests.exceptions.ConnectionError:
        return (
            "Connection error: P.A.E.D.S. API is unreachable. "
            "Check the ngrok URL or network."
        )
    except requests.exceptions.Timeout:
        return "The P.A.E.D.S. API did not respond in time. Please try again."
    except Exception as exc:
        return f"Unexpected error querying P.A.E.D.S. API: {exc}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_headers(api_key: str | None) -> dict:
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    key = get_api_key()
    if health_check(api_key=key):
        test_answer = ask_paeds(
            "What is the normal respiratory rate for a newborn?",
            api_key=key,
        )
        print(f"\nTest query result: {test_answer}")
    else:
        print("\nSkipping test query — API not reachable.")
