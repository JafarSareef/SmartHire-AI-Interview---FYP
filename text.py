import os
import requests
import time
from typing import List, Dict, Optional

# Default model and key can be overridden by env vars
# Use the provided API key as the fallback default when GEMINI_API_KEY is not set.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash-lite")
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAkieYUkc8EFftcHsaLvS_6tXEjlar9vT8")


def call_gemini(messages: List[Dict[str, str]], temperature: float = 0.7, top_p: float = 0.9,
                model: Optional[str] = None, api_key: Optional[str] = None) -> Optional[str]:
    """Calls the Gemini API using the modern generateContent method."""
    key = api_key or DEFAULT_API_KEY
    model_to_use = model or DEFAULT_MODEL
    if not key:
        return "ERROR: GEMINI_API_KEY is not set."

    # Normalize model name to just the ID
    model_id = model_to_use.split('/')[-1]

    # Combine system and user prompts for models that don't support systemInstruction
    system_prompt = ""
    user_prompts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_prompt = content
        else:
            user_prompts.append(content)
    
    combined_prompt = f"{system_prompt}\n\n{''.join(user_prompts)}"

    body = {
        "contents": [
            {"role": "user", "parts": [{"text": combined_prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": 8192
        }
    }

    # The new models use v1 or v1beta endpoints with generateContent
    base_urls = [
        f"https://generativelanguage.googleapis.com/v1/models/{model_id}:generateContent",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent",
    ]

    headers = {"Content-Type": "application/json"}
    params = {"key": key}

    last_error = "No response from API."
    for url in base_urls:
        try:
            resp = requests.post(url, json=body, headers=headers, params=params, timeout=20)
            
            if resp.status_code == 200:
                try:
                    j = resp.json()
                    # Parse the new response structure
                    return j["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError) as e:
                    last_error = f"Could not parse content from response: {resp.text[:500]} (exception: {e})"
                    # This was a successful request, so don't retry, return the error
                    return f"ERROR: {last_error}"
            else:
                last_error = f"status={resp.status_code} body={resp.text[:500]}"
                if resp.status_code == 404:
                    # If model not found at this URL, try the next one
                    continue
                else:
                    # For other errors (400, 401, 500), fail fast as retrying won't help
                    return f"ERROR: {last_error}"

        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {e}"
            time.sleep(1)  # Wait a second before trying the next URL
            continue
    
    # This is returned if all URLs and retries fail
    return f"ERROR: Failed after trying all endpoints. Last error: {last_error}"
