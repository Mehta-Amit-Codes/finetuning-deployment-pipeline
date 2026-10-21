"""Minimal OpenAI-compatible server standing in for vLLM in tests."""
import json
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI

app = FastAPI()


def _answer(email: str) -> dict:
    e = email.lower()
    category = ("billing" if any(w in e for w in ("charge", "refund", "invoice"))
                else "technical" if any(w in e for w in ("crash", "error", "bug"))
                else "shipping" if any(w in e for w in ("package", "delivery", "shipping"))
                else "account" if any(w in e for w in ("password", "login", "account"))
                else "other")
    urgency = "high" if any(w in e for w in ("asap", "twice", "!!")) else "low"
    return {"category": category, "urgency": urgency, "summary": "Customer reports an issue."}


@app.post("/v1/chat/completions")
def chat(body: dict):
    msgs = body["messages"]
    email = msgs[-1]["content"]
    model = body["model"]
    # Simulate a weak base model: wraps JSON in prose half the time and mislabels urgency.
    text = json.dumps(_answer(email))
    if "triage-lora" not in model:
        text = "Sure! Here is the triage: " + text.replace('"urgency": "high"', '"urgency": "low"')
    return {"id": "x", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 40 + 30 * (len(msgs) > 2), "completion_tokens": 20, "total_tokens": 80}}


def start_server():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}/v1", server
