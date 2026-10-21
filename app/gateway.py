"""
Serving gateway in front of the model server(s) (vLLM exposes an OpenAI-compatible API).

  POST /v1/triage   routes a request to base or fine-tuned per the traffic split
  GET  /stats       live per-arm latency / JSON validity / judge score
  /admin/*          traffic split, rollback controls, fault injection for demos

Rollback (the stretch goal): every response is checked cheaply (did it parse
as valid JSON?). A sampled fraction is also scored by the Grok judge in a
background task, so judging never adds user-facing latency. If, over a
rolling window of the fine-tuned arm's recent traffic, JSON validity or the
mean judge score falls below threshold, rollback_active flips on and ALL
traffic goes to the base model until an operator resets it.

Why base uses few-shot here: the base arm is also the safety net, so it should
be the strongest *prompting* baseline, not the weakest one.
"""
import hashlib
import os
import random
import statistics
import uuid
from functools import lru_cache

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from app import store
from app.config import (BASE_ENDPOINT, BASE_MODEL_NAME, FT_ENDPOINT, FT_MODEL_NAME,
                        build_messages, parse_triage)
from app.metrics import percentile

LIVE_JUDGE_SAMPLE_RATE = float(os.environ.get("LIVE_JUDGE_SAMPLE_RATE", "0.3"))
ROLLBACK_WINDOW = int(os.environ.get("ROLLBACK_WINDOW", "30"))
ROLLBACK_MIN_SAMPLES = int(os.environ.get("ROLLBACK_MIN_SAMPLES", "15"))
ROLLBACK_MIN_JUDGED = int(os.environ.get("ROLLBACK_MIN_JUDGED", "8"))
ROLLBACK_MIN_JSON_VALID = float(os.environ.get("ROLLBACK_MIN_JSON_VALID", "0.90"))
ROLLBACK_MIN_JUDGE = float(os.environ.get("ROLLBACK_MIN_JUDGE", "0.60"))
GATEWAY_BASE_FEWSHOT = os.environ.get("GATEWAY_BASE_FEWSHOT", "1") == "1"
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")

ARMS = {
    "base": {"endpoint": BASE_ENDPOINT, "model": BASE_MODEL_NAME, "fewshot": GATEWAY_BASE_FEWSHOT},
    "finetuned": {"endpoint": FT_ENDPOINT, "model": FT_MODEL_NAME, "fewshot": False},
}

app = FastAPI(title="Fine-tuned model gateway", version="0.1.0")
store.init_db()


@lru_cache(maxsize=8)
def _client(endpoint: str) -> OpenAI:
    return OpenAI(base_url=endpoint, api_key="EMPTY", timeout=30)


def require_admin(x_admin_key: str = Header(default="")):
    if ADMIN_KEY and x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")


class TriageRequest(BaseModel):
    email: str = Field(..., min_length=1)
    user_id: str | None = None  # optional: gives sticky assignment to one arm per user


def choose_arm(user_id: str | None) -> tuple[str, str]:
    state = store.get_state()
    if state["rollback_active"]:
        return "base", "rollback"
    if user_id:
        roll = (int(hashlib.sha256(user_id.encode()).hexdigest()[:8], 16) % 10000) / 100
    else:
        roll = random.random() * 100
    return ("finetuned" if roll < state["ft_traffic_pct"] else "base"), "split"


def _call_arm(arm: str, email: str) -> dict:
    cfg = ARMS[arm]
    import time
    t0 = time.perf_counter()
    resp = _client(cfg["endpoint"]).chat.completions.create(
        model=cfg["model"], messages=build_messages(email, cfg["fewshot"]),
        temperature=0, max_tokens=150)
    text = resp.choices[0].message.content or ""
    if arm == "finetuned" and store.get_state()["demo_degrade"]:
        # Fault injection for demos: simulate a bad fine-tune regression by truncating
        # output, which breaks the JSON. Lets you watch automatic rollback fire.
        text = text[: max(1, len(text) // 3)]
    return {"text": text, "latency_ms": (time.perf_counter() - t0) * 1000,
            "prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}


@app.post("/v1/triage")
def triage(req: TriageRequest, background: BackgroundTasks):
    arm, routed_by = choose_arm(req.user_id)
    try:
        result = _call_arm(arm, req.email)
    except Exception as e:
        if arm != "finetuned":
            raise HTTPException(status_code=502, detail=f"base model unavailable: {e}")
        # Fine-tuned arm down: degrade gracefully to base instead of failing the user.
        store.log_event("fallback", f"finetuned arm failed ({e}); request served by base")
        arm, routed_by = "base", "fallback"
        try:
            result = _call_arm("base", req.email)
        except Exception as e2:
            raise HTTPException(status_code=502, detail=f"both arms unavailable: {e2}")

    parsed = parse_triage(result["text"])
    request_id = str(uuid.uuid4())
    store.insert_request(
        id=request_id, arm=arm, routed_by=routed_by, user_id=req.user_id, email=req.email,
        output=result["text"], json_valid=1 if parsed else 0, latency_ms=result["latency_ms"],
        prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"])
    background.add_task(_post_process, request_id, req.email, result["text"], arm)

    return {"request_id": request_id, "arm": arm, "model": ARMS[arm]["model"], "routed_by": routed_by,
            "result": parsed, "raw": result["text"], "json_valid": parsed is not None,
            "latency_ms": round(result["latency_ms"], 1)}


def _post_process(request_id: str, email: str, text: str, arm: str) -> None:
    from app.grok_client import grok_available
    if grok_available() and random.random() < LIVE_JUDGE_SAMPLE_RATE:
        from app.judge import judge_triage
        score = judge_triage(email, text)
        if score is not None:
            store.update_judge(request_id, score)
    _check_rollback()


def _check_rollback() -> None:
    state = store.get_state()
    if state["rollback_active"]:
        return
    rows = store.recent_requests("finetuned", ROLLBACK_WINDOW, since=state["rollback_reset_ts"])
    if len(rows) < ROLLBACK_MIN_SAMPLES:
        return
    reasons = []
    valid_rate = sum(r["json_valid"] for r in rows) / len(rows)
    if valid_rate < ROLLBACK_MIN_JSON_VALID:
        reasons.append(f"JSON validity {valid_rate:.0%} < {ROLLBACK_MIN_JSON_VALID:.0%} over last {len(rows)} requests")
    judged = [r["judge_score"] for r in rows if r["judge_score"] is not None]
    if len(judged) >= ROLLBACK_MIN_JUDGED and statistics.mean(judged) < ROLLBACK_MIN_JUDGE:
        reasons.append(f"mean judge score {statistics.mean(judged):.2f} < {ROLLBACK_MIN_JUDGE:.2f} over {len(judged)} judged")
    if reasons:
        store.set_kv("rollback_active", "1")
        store.log_event("rollback", "AUTO-ROLLBACK to base: " + "; ".join(reasons))


def _arm_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"requests": 0}
    lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    judged = [r["judge_score"] for r in rows if r["judge_score"] is not None]
    return {
        "requests": len(rows),
        "avg_latency_ms": round(statistics.mean(lat), 1) if lat else None,
        "p95_latency_ms": round(percentile(lat, 95), 1) if lat else None,
        "json_valid_rate": round(sum(r["json_valid"] for r in rows) / len(rows), 3),
        "avg_judge_score": round(statistics.mean(judged), 3) if judged else None,
        "judged": len(judged),
        "avg_prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in rows), 1),
    }


@app.get("/stats")
def stats():
    return {arm: _arm_stats(store.recent_requests(arm, 1000)) for arm in ARMS}


@app.get("/events")
def events(limit: int = 30):
    return store.recent_events(limit)


@app.get("/admin/state", dependencies=[Depends(require_admin)])
def admin_state():
    return store.get_state()


class TrafficRequest(BaseModel):
    ft_traffic_pct: float = Field(..., ge=0, le=100)


@app.post("/admin/traffic", dependencies=[Depends(require_admin)])
def set_traffic(req: TrafficRequest):
    store.set_kv("ft_traffic_pct", req.ft_traffic_pct)
    store.log_event("traffic", f"fine-tuned traffic set to {req.ft_traffic_pct:.0f}%")
    return store.get_state()


@app.post("/admin/rollback/trigger", dependencies=[Depends(require_admin)])
def trigger_rollback():
    store.set_kv("rollback_active", "1")
    store.log_event("rollback", "MANUAL rollback to base")
    return store.get_state()


@app.post("/admin/rollback/reset", dependencies=[Depends(require_admin)])
def reset_rollback():
    import time
    store.set_kv("rollback_active", "0")
    store.set_kv("rollback_reset_ts", time.time())  # old bad requests must not instantly re-trigger
    store.log_event("rollback", "rollback cleared; fine-tuned model eligible for traffic again")
    return store.get_state()


class DegradeRequest(BaseModel):
    enabled: bool


@app.post("/admin/demo-degrade", dependencies=[Depends(require_admin)])
def demo_degrade(req: DegradeRequest):
    store.set_kv("demo_degrade", "1" if req.enabled else "0")
    store.log_event("demo", f"fault injection {'ENABLED' if req.enabled else 'disabled'} on fine-tuned arm")
    return store.get_state()


@app.get("/health")
def health():
    return {"status": "ok"}
