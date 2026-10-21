"""
LLM-as-judge for triage outputs, used in both the offline A/B test and the
live monitoring that drives rollback.

Design choices worth knowing:
  - Invalid / unparseable output scores 0.0 (not "skipped"): a model that
    can't produce the format is worse, and skipping would flatter it.
  - The judge sees the parsed object re-serialised, not the raw text, so
    stray prose around the JSON can't sway the score.
  - The judge is Grok, a different model family from the open models being
    compared, which avoids a model grading its own style.
"""
import json

from app.config import parse_triage
from app.grok_client import chat

JUDGE_SYSTEM = (
    "You grade support-ticket triage outputs. Given a customer email and a triage "
    "object (category, urgency, summary), rate overall quality from 1 to 5:\n"
    "5 = category and urgency both clearly right, summary accurate and concise\n"
    "3 = one of category/urgency is debatable or the summary misses a key point\n"
    "1 = category or urgency clearly wrong, or the summary is wrong/fabricated\n"
    'Urgency guide: high = outage / money lost / time-critical; medium = real problem '
    "but workable; low = question or minor request.\n"
    'Respond with ONLY JSON: {"score": <1-5 integer>, "reason": "<one sentence>"}'
)


def judge_triage(email: str, output_text: str) -> float | None:
    """Returns a score in [0, 1], 0.0 for invalid output, or None if the judge
    call itself failed or returned something unparseable (so it's excluded)."""
    parsed = parse_triage(output_text)
    if parsed is None:
        return 0.0
    try:
        raw = chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"Customer email:\n{email}\n\nTriage output:\n{json.dumps(parsed)}"},
            ],
            temperature=0.0, max_tokens=200,
        )
        start = raw.find("{")
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        score = int(obj["score"])
        return (min(5, max(1, score)) - 1) / 4
    except Exception:
        return None
