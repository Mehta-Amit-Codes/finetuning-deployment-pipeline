"""
Streamlit UI for the Fine-Tuning + Deployment Pipeline.

Tabs:
  1. Dataset        generate with Grok (or upload your CSV), inspect the split
  2. Training       loss curves + run config from the GPU box's train_log.json
  3. A/B Results    run/see the base vs few-shot vs fine-tuned comparison with CIs
  4. Gateway        live traffic split, playground, per-arm stats, rollback control
  5. Scaling        scaling policy + capacity planner

Run:  streamlit run streamlit_app.py
"""
import json
import math
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

from app import data_prep
from app.config import DATA_DIR, EVAL_DIR, OUTPUT_DIR, ROOT
from app.grok_client import grok_available

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8004")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "")
ADMIN_HEADERS = {"X-Admin-Key": ADMIN_KEY} if ADMIN_KEY else {}

st.set_page_config(page_title="Fine-Tuning Pipeline", page_icon="🧪", layout="wide")
st.title("🧪 Fine-Tuning + Deployment Pipeline")
st.caption(f"Support-ticket triage · LoRA/QLoRA · vLLM serving · A/B + rollback · Gateway: {GATEWAY_URL}")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def gateway(method: str, path: str, admin: bool = False, **kw):
    try:
        r = requests.request(method, f"{GATEWAY_URL}{path}", timeout=60,
                             headers=ADMIN_HEADERS if admin else {}, **kw)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Gateway request failed ({path}): {e}")
        return None


tab_data, tab_train, tab_ab, tab_gw, tab_scale = st.tabs(
    ["Dataset", "Training", "⚖️ A/B Results", "🚦 Gateway", "Scaling"])

# ---------------------------------------------------------------- Dataset ----
with tab_data:
    st.subheader("Task-specific instruction dataset")
    st.write("Train / val / test split is **stratified by category** and checked for leakage. "
             "Grok writes the synthetic emails; the model being fine-tuned is an open-weights one.")

    manifest = read_json(DATA_DIR / "manifest.json")
    if manifest:
        c = manifest["counts"]
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [("Train", "train"), ("Val", "val"), ("Test (held out)", "test"), ("Total", "total")]):
            col.metric(label, c[key])
        dist = pd.DataFrame({"category": list(manifest["category_distribution"]),
                             "count": list(manifest["category_distribution"].values())}).set_index("category")
        st.bar_chart(dist)
    else:
        st.info("No dataset yet. Generate one below or upload your own CSV.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Generate with Grok**")
        n = st.slider("Examples", 100, 2000, 600, step=100)
        if st.button("Generate dataset", type="primary"):
            if not grok_available():
                st.error("GROK_API_KEY is not set. Add it to .env and restart.")
            else:
                bar = st.progress(0.0, text="Starting…")
                examples = data_prep.generate_dataset(
                    n, progress=lambda f, cnt: bar.progress(min(f, 1.0), text=f"{cnt} unique examples"))
                if len(examples) < n:
                    st.warning(f"Only {len(examples)} valid unique examples came back (asked for {n}).")
                data_prep.split_and_save(examples)
                st.rerun()
    with right:
        st.markdown("**…or upload your own CSV** (columns: `email, category, urgency, summary`)")
        up = st.file_uploader("Labelled CSV", type="csv")
        if up and st.button("Import CSV"):
            examples = data_prep.load_csv(up.getvalue().decode("utf-8"))
            if not examples:
                st.error("No valid rows found. Check column names and label values.")
            else:
                data_prep.split_and_save(examples)
                st.rerun()

    train_file = DATA_DIR / "train.jsonl"
    if train_file.exists():
        st.markdown("**Training examples (preview)**")
        rows = []
        for line in train_file.read_text(encoding="utf-8").splitlines()[:15]:
            msgs = json.loads(line)["messages"]
            label = json.loads(msgs[2]["content"])
            rows.append({"email": msgs[1]["content"][:140], **label})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# --------------------------------------------------------------- Training ----
with tab_train:
    st.subheader("QLoRA training")
    st.write("Training needs a CUDA GPU, so it normally runs remotely (Colab T4, Modal, RunPod spot). "
             "This tab reads the artifacts it produces.")
    st.code("pip install -r requirements-train.txt\n"
            "python -m app.train_lora --epochs 3          # add --resume after a spot pre-emption\n"
            "python -m app.merge_adapter                  # optional: standalone merged model", language="bash")

    uploaded_log = st.file_uploader("Load train_log.json from your GPU box (optional)", type="json")
    log = json.loads(uploaded_log.getvalue()) if uploaded_log else read_json(OUTPUT_DIR / "train_log.json")
    if log:
        train = {e["step"]: e["loss"] for e in log if "loss" in e and "eval_loss" not in e}
        evalv = {e["step"]: e["eval_loss"] for e in log if "eval_loss" in e}
        df = pd.DataFrame({"train_loss": pd.Series(train), "eval_loss": pd.Series(evalv)}).sort_index()
        st.line_chart(df)
        st.caption("Eval loss rising while train loss keeps falling = overfitting; stop earlier or add data.")
    else:
        st.info("No train_log.json found yet.")

    cfg = read_json(OUTPUT_DIR / "run_config.json")
    if cfg:
        cols = st.columns(4)
        cols[0].metric("Trainable params", f"{cfg['trainable_pct']}%")
        cols[1].metric("Train time", f"{cfg['train_seconds']:.0f}s")
        cols[2].metric("LoRA rank", cfg["lora_rank"])
        cols[3].metric("GPU", cfg.get("gpu", "?"))
        with st.expander("Full run config"):
            st.json(cfg)
    pkg = read_json(OUTPUT_DIR / "package_report.json")
    if pkg:
        st.markdown("**Packaging**")
        cols = st.columns(2)
        cols[0].metric("Adapter only", f"{pkg['adapter_size_mb']} MB")
        cols[1].metric("Merged model", f"{pkg['merged_size_mb']} MB")

# -------------------------------------------------------------------- A/B ----
with tab_ab:
    st.subheader("Base vs few-shot base vs fine-tuned")
    st.write("Identical held-out emails through every arm. **few-shot base** is the honest "
             "'could I just prompt it better?' baseline. Needs the model server running.")

    c1, c2, c3 = st.columns([1, 1, 2])
    limit = c1.number_input("Max test examples", 10, 1000, 100, step=10)
    use_judge = c2.checkbox("Score with Grok judge", value=grok_available(), disabled=not grok_available())
    if c3.button("Run A/B test", type="primary"):
        from app import ab_test
        bar = st.progress(0.0, text="Starting…")
        try:
            ab_test.evaluate(limit=int(limit), use_judge=use_judge,
                             progress=lambda f, msg: bar.progress(min(f, 1.0), text=msg))
        except FileNotFoundError:
            st.error("No test set found; build the dataset first.")
        except Exception as e:
            st.error(f"A/B run failed: {e}")

    summary = read_json(EVAL_DIR / "ab_summary.json")
    if summary:
        st.caption(f"{summary['n_examples']} test examples · judge {'on' if summary['judge_used'] else 'off'} · "
                   "bars show 95% bootstrap CIs")
        chart_rows, table_rows = [], []
        for arm, s in summary["arms"].items():
            row = {"arm": arm, "errors": s["errors"], "p95 latency (ms)": round(s["latency_ms"]["p95"] or 0),
                   "prompt tokens": round(s["prompt_tokens_mean"] or 0)}
            for m in ["json_valid", "category_correct", "urgency_correct", "exact_match", "judge"]:
                v = s[m]
                if v["mean"] is None:
                    continue
                row[m] = f"{v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
                chart_rows.append({"arm": arm, "metric": m, "mean": v["mean"], "lo": v["ci_low"], "hi": v["ci_high"]})
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        if chart_rows:
            cdf = pd.DataFrame(chart_rows)
            bars = alt.Chart(cdf).mark_bar().encode(
                x=alt.X("metric:N", title=None), y=alt.Y("mean:Q", scale=alt.Scale(domain=[0, 1])),
                color="arm:N", xOffset="arm:N")
            errs = alt.Chart(cdf).mark_errorbar().encode(
                x="metric:N", y=alt.Y("lo:Q", title="score"), y2="hi:Q", xOffset="arm:N")
            try:
                st.altair_chart(bars + errs, use_container_width=True)
            except Exception:
                st.bar_chart(cdf.pivot(index="metric", columns="arm", values="mean"))

        st.markdown(f"#### Lift of `{summary['candidate']}` (paired, per-example)")
        comp_rows = []
        for name, comp in summary["comparisons"].items():
            for m, d in comp.items():
                if d["mean_diff"] is None:
                    continue
                verdict = ("real lift" if d["ci_low"] > 0 else "worse" if d["ci_high"] < 0
                           else "➖ inconclusive (CI spans 0)")
                comp_rows.append({"comparison": name, "metric": m, "lift": f"{d['mean_diff']:+.3f}",
                                  "95% CI": f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]",
                                  "P(better)": f"{d['p_better']:.2f}", "verdict": verdict})
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
        st.caption("A CI that excludes 0 means the lift holds up at this sample size. "
                   "A CI spanning 0 means: don't claim a win yet — get more test data.")
    else:
        st.info("No A/B results yet. Run the test above once the model server is up.")

# ---------------------------------------------------------------- Gateway ----
with tab_gw:
    st.subheader("Live traffic, playground and rollback")
    state = gateway("GET", "/admin/state", admin=True)
    if state:
        if state["rollback_active"]:
            st.error("ROLLBACK ACTIVE: 100% of traffic is on the base model.")
        else:
            st.success(f"Serving: {state['ft_traffic_pct']:.0f}% fine-tuned / {100 - state['ft_traffic_pct']:.0f}% base")

        c1, c2 = st.columns(2)
        with c1:
            pct = st.slider("Fine-tuned traffic %", 0, 100, int(state["ft_traffic_pct"]))
            if st.button("Apply traffic split"):
                gateway("POST", "/admin/traffic", admin=True, json={"ft_traffic_pct": pct})
                st.rerun()
        with c2:
            b1, b2 = st.columns(2)
            if b1.button("Manual rollback"):
                gateway("POST", "/admin/rollback/trigger", admin=True)
                st.rerun()
            if b2.button("Clear rollback"):
                gateway("POST", "/admin/rollback/reset", admin=True)
                st.rerun()
            degrade = st.toggle("Demo: degrade fine-tuned model", value=state["demo_degrade"],
                                help="Truncates fine-tuned output to simulate a bad model regression. "
                                     "Send ~15 requests and watch automatic rollback fire.")
            if degrade != state["demo_degrade"]:
                gateway("POST", "/admin/demo-degrade", admin=True, json={"enabled": degrade})
                st.rerun()

    st.markdown("**Playground**")
    email = st.text_area("Customer email", height=100,
                         value="I was charged twice for my subscription this month and nobody is answering my emails!!")
    pc1, pc2 = st.columns(2)
    if pc1.button("Send one", type="primary") and email:
        out = gateway("POST", "/v1/triage", json={"email": email})
        if out:
            st.write(f"Served by **{out['arm']}** (`{out['model']}`) · routed by *{out['routed_by']}* · {out['latency_ms']} ms")
            st.json(out["result"] if out["result"] else {"invalid_output": out["raw"]})
    n_burst = pc2.number_input("Burst size", 1, 100, 20)
    if pc2.button("Send burst") and email:
        bar = st.progress(0.0)
        for i in range(int(n_burst)):
            gateway("POST", "/v1/triage", json={"email": email})
            bar.progress((i + 1) / int(n_burst))
        st.rerun()

    st.markdown("**Live per-arm stats**")
    stats = gateway("GET", "/stats")
    if stats:
        st.dataframe(pd.DataFrame(stats).T, use_container_width=True)
    st.markdown("**Event log**")
    events = gateway("GET", "/events")
    if events:
        st.dataframe(pd.DataFrame(events)[["ts", "kind", "message"]].assign(
            ts=lambda d: pd.to_datetime(d["ts"], unit="s")), use_container_width=True)

# ---------------------------------------------------------------- Scaling ----
with tab_scale:
    st.subheader("Scaling policy and capacity planner")
    policy = ROOT / "docs" / "SCALING_POLICY.md"
    if policy.exists():
        st.markdown(policy.read_text(encoding="utf-8"))

    st.markdown("### Capacity planner (rough estimate)")
    st.caption("Decode is the bottleneck for short-output tasks like this one. Replace the defaults with "
               "numbers from your own load test; the A/B tab's latency and token columns are a starting point.")
    p1, p2, p3, p4 = st.columns(4)
    peak_rps = p1.number_input("Peak requests/sec", 0.1, 1000.0, 20.0)
    out_tokens = p2.number_input("Avg output tokens", 1, 1000, 45)
    tps = p3.number_input("Replica throughput (tok/s, batched)", 10, 20000, 1500)
    util = p4.slider("Target utilization", 0.3, 0.9, 0.6)
    replicas = math.ceil(peak_rps * out_tokens / (tps * util))
    st.metric("Replicas needed at peak", replicas)
    st.caption(f"replicas = ceil(peak_rps × output_tokens / (tokens_per_sec × utilization)) "
               f"= ceil({peak_rps} × {out_tokens} / ({tps} × {util}))")
