"""
Central config for the fine-tuning + deployment pipeline.

Task: support-ticket triage. A messy customer email goes in; a strict JSON
object comes out: {"category", "urgency", "summary"}. It's a good fine-tuning
target because (a) base models often get the format or label set subtly
wrong, (b) category/urgency can be scored deterministically, and (c) the
summary can be scored by an LLM judge.

Grok's role here: Grok cannot be fine-tuned through this pipeline (the model
being tuned must be open-weights). Grok is used for (1) synthetic dataset
generation and (2) LLM-as-judge scoring in the A/B test and live monitoring.
"""
import json
import os
from pathlib import Path

try:  # training boxes may not have python-dotenv installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "outputs"))
EVAL_DIR = Path(os.environ.get("EVAL_DIR", ROOT / "eval_results"))
DB_PATH = os.environ.get("GATEWAY_DB", str(ROOT / "gateway.db"))

# --- Task definition ---------------------------------------------------------
CATEGORIES = ["billing", "technical", "account", "shipping", "other"]
URGENCIES = ["low", "medium", "high"]

SYSTEM_PROMPT = (
    "You are a support-ticket triage assistant. Read the customer email and respond "
    "with ONLY a JSON object with exactly these keys: "
    '"category" (one of: billing, technical, account, shipping, other), '
    '"urgency" (one of: low, medium, high), and '
    '"summary" (one sentence, at most 20 words). No other text.'
)

# Used for the "base + few-shot prompting" baseline, which answers the honest
# question: does fine-tuning beat just prompting the base model better?
FEWSHOT = [
    {
        "email": "Hi, I was charged twice for my order #4471 this month and my bank shows both. Please refund one ASAP.",
        "label": {"category": "billing", "urgency": "high",
                  "summary": "Customer was double-charged for order 4471 and wants a refund."},
    },
    {
        "email": "how do i change the email on my account? no rush",
        "label": {"category": "account", "urgency": "low",
                  "summary": "Customer asks how to change the email address on their account."},
    },
    {
        "email": "The app crashes every time I open the camera feature since the last update. Works fine otherwise.",
        "label": {"category": "technical", "urgency": "medium",
                  "summary": "App crashes when opening the camera feature after the latest update."},
    },
]

# --- Models / endpoints (OpenAI-compatible, e.g. vLLM) --------------------------
BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
BASE_ENDPOINT = os.environ.get("BASE_ENDPOINT", "http://localhost:8000/v1")
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", BASE_MODEL_ID)
# With vLLM's --enable-lora, base and fine-tuned share one server and one GPU:
# same endpoint, different model name.
FT_ENDPOINT = os.environ.get("FT_ENDPOINT", BASE_ENDPOINT)
FT_MODEL_NAME = os.environ.get("FT_MODEL_NAME", "triage-lora")

# --- Grok (dataset generation + judge) ------------------------------------------
GROK_BASE_URL = os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1")
GROK_MODEL = os.environ.get("GROK_MODEL", "grok-4")


def build_messages(email: str, fewshot: bool = False) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if fewshot:
        for ex in FEWSHOT:
            msgs.append({"role": "user", "content": ex["email"]})
            msgs.append({"role": "assistant", "content": json.dumps(ex["label"])})
    msgs.append({"role": "user", "content": email})
    return msgs


def parse_triage(text: str) -> dict | None:
    """Strict parse: returns the triage dict, or None if the output isn't valid
    JSON with legal enum values. 'None' is what we count as a format failure."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("category") not in CATEGORIES or obj.get("urgency") not in URGENCIES:
        return None
    if not isinstance(obj.get("summary"), str):
        return None
    return {k: obj[k] for k in ("category", "urgency", "summary")}
