import json

import pytest
from fastapi.testclient import TestClient

from app import ab_test, gateway, store
from tests.fake_vllm import start_server

BILLING = "I was charged twice for my order, refund me asap"


@pytest.fixture(scope="module")
def endpoint():
    url, server = start_server()
    yield url
    server.should_exit = True


@pytest.fixture()
def client(endpoint, monkeypatch):
    for arm in gateway.ARMS.values():
        monkeypatch.setitem(arm, "endpoint", endpoint)
    monkeypatch.setitem(gateway.ARMS["finetuned"], "model", "triage-lora")
    monkeypatch.setitem(gateway.ARMS["base"], "model", "base-model")
    monkeypatch.setattr(gateway, "ROLLBACK_MIN_SAMPLES", 6)
    store.set_kv("rollback_active", "0")
    store.set_kv("demo_degrade", "0")
    store.set_kv("ft_traffic_pct", "100")
    return TestClient(gateway.app)


def test_routes_by_traffic_split(client):
    store.set_kv("ft_traffic_pct", "100")
    assert client.post("/v1/triage", json={"email": BILLING}).json()["arm"] == "finetuned"
    store.set_kv("ft_traffic_pct", "0")
    assert client.post("/v1/triage", json={"email": BILLING}).json()["arm"] == "base"


def test_sticky_assignment_per_user(client):
    store.set_kv("ft_traffic_pct", "50")
    arms = {client.post("/v1/triage", json={"email": BILLING, "user_id": "user-42"}).json()["arm"] for _ in range(8)}
    assert len(arms) == 1


def test_finetuned_returns_valid_json(client):
    out = client.post("/v1/triage", json={"email": BILLING}).json()
    assert out["json_valid"] and out["result"]["category"] == "billing"


def test_auto_rollback_fires_when_model_degrades(client):
    client.post("/admin/demo-degrade", json={"enabled": True})
    for _ in range(8):
        client.post("/v1/triage", json={"email": BILLING})
    assert client.get("/admin/state").json()["rollback_active"] is True
    after = client.post("/v1/triage", json={"email": BILLING}).json()
    assert after["arm"] == "base" and after["routed_by"] == "rollback"
    assert any(e["kind"] == "rollback" for e in client.get("/events").json())


def test_reset_does_not_instantly_retrigger(client):
    client.post("/admin/demo-degrade", json={"enabled": True})
    for _ in range(8):
        client.post("/v1/triage", json={"email": BILLING})
    client.post("/admin/demo-degrade", json={"enabled": False})
    client.post("/admin/rollback/reset")
    out = client.post("/v1/triage", json={"email": BILLING}).json()
    assert out["arm"] == "finetuned" and out["json_valid"]
    assert client.get("/admin/state").json()["rollback_active"] is False


def test_falls_back_to_base_when_finetuned_arm_down(client, monkeypatch):
    monkeypatch.setitem(gateway.ARMS["finetuned"], "endpoint", "http://127.0.0.1:1/v1")
    out = client.post("/v1/triage", json={"email": BILLING}).json()
    assert out["arm"] == "base" and out["routed_by"] == "fallback"


def test_admin_key_enforced(client, monkeypatch):
    monkeypatch.setattr(gateway, "ADMIN_KEY", "secret")
    assert client.get("/admin/state").status_code == 401
    assert client.get("/admin/state", headers={"X-Admin-Key": "secret"}).status_code == 200


def test_ab_harness_end_to_end(endpoint, tmp_path):
    test_file = tmp_path / "test.jsonl"
    rows = [{"email": BILLING, "label": {"category": "billing", "urgency": "high", "summary": "s"}},
            {"email": "app crashes with an error on login!!", "label": {"category": "technical", "urgency": "high", "summary": "s"}}] * 10
    test_file.write_text("\n".join(json.dumps(r) for r in rows))
    arms = [{"name": "base", "endpoint": endpoint, "model": "base-model", "fewshot": False},
            {"name": "base_fewshot", "endpoint": endpoint, "model": "base-model", "fewshot": True},
            {"name": "finetuned", "endpoint": endpoint, "model": "triage-lora", "fewshot": False}]
    s = ab_test.evaluate(arms, use_judge=False, test_path=test_file, out_dir=tmp_path / "out")
    assert s["arms"]["finetuned"]["exact_match"]["mean"] == 1.0
    assert s["arms"]["base"]["json_valid"]["mean"] == 1.0  # JSON present, just wrapped in prose
    assert s["arms"]["base"]["exact_match"]["mean"] < 1.0   # fake base mislabels urgency
    lift = s["comparisons"]["finetuned_vs_base"]["exact_match"]
    assert lift["mean_diff"] > 0 and lift["ci_low"] > 0
    assert s["arms"]["base_fewshot"]["prompt_tokens_mean"] > s["arms"]["finetuned"]["prompt_tokens_mean"]
    assert (tmp_path / "out" / "ab_summary.json").exists()
