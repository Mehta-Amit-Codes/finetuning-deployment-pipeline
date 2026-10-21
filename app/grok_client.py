"""Thin Grok (xAI) client. Grok's API is OpenAI-compatible, so the standard
`openai` package works pointed at xAI's base_url."""
import os
import time
from functools import lru_cache

from openai import OpenAI

from app.config import GROK_BASE_URL, GROK_MODEL


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    key = os.environ.get("GROK_API_KEY")
    if not key:
        raise RuntimeError("GROK_API_KEY is not set (copy .env.example to .env and fill it in).")
    return OpenAI(api_key=key, base_url=GROK_BASE_URL, timeout=120)


def grok_available() -> bool:
    return bool(os.environ.get("GROK_API_KEY"))


def chat(messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1500,
         model: str | None = None, retries: int = 3) -> str:
    client = get_client()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model or GROK_MODEL, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # network / rate limit / transient server errors
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Grok call failed after {retries} attempts: {last_err}")
