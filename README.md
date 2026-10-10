# Fine-Tuning + Deployment Pipeline

Reference implementation of the "Fine-Tuning + Deployment Pipeline" blueprint:
task-specific dataset → QLoRA fine-tune → package → serve behind a gateway →
rigorous A/B against the base model → automatic rollback. Includes a Streamlit
UI over the whole thing.

**Task:** support-ticket triage. A messy customer email goes in; strict JSON
comes out: `{"category", "urgency", "summary"}`.

**Where Grok fits:** Grok can't be fine-tuned here (the tuned model must be
open-weights, e.g. Qwen/Llama/Mistral). Grok is used for **(1) synthetic
dataset generation** and **(2) LLM-as-judge** scoring in the A/B test and in
live monitoring. It is a different model family from the one being graded,
which avoids a model marking its own homework.

## Pipeline

```
 Grok writes labelled emails ──► validate, dedup, stratified split ──► train / val / test(held out)
                                                                              │
                                      GPU box (Colab / Modal / RunPod spot)   ▼
                                      QLoRA (4-bit base + LoRA) via PEFT+TRL ─► adapter/  train_log.json
                                                                              │
                                                  merge (optional) ◄──────────┤
                                                                              ▼
                       vLLM serves base + adapter from ONE GPU (--enable-lora)
                                                                              │
              ┌───────────────────────────────────────────────────────────────┤
              ▼                                                               ▼
   A/B harness (offline)                                          Gateway (online)
   base vs base+few-shot vs fine-tuned                            traffic split, fallback,
   same held-out emails, paired bootstrap CIs                     live JSON-validity + Grok judge
                                                                  └─► AUTO-ROLLBACK to base if quality drops
```

## What was verified, and what wasn't

| Part | Status |
|---|---|
| Data parsing/validation/split, stats (bootstrap CIs), gateway routing, sticky users, fallback, admin auth, automatic rollback, reset, A/B harness | **Covered by 16 automated tests** against a fake OpenAI-compatible server (`pytest`) |
| Streamlit UI | Loads headlessly without errors; degrades to error messages if the gateway is down |
| `train_lora.py`, `merge_adapter.py`, vLLM serving | **Not executed here (no GPU).** Written against current PEFT/TRL/vLLM APIs, with compatibility shims for TRL's renamed arguments. Expect to fix small version-specific issues on first run |
| Grok calls (generation, judge) | Wired and tested with Grok disabled; first real run will use your `GROK_API_KEY` |

## Setup

Your laptop only needs the lean requirements. Training deps (torch,
bitsandbytes) live on the GPU box, because bitsandbytes needs CUDA and is
awkward on Windows.

```bash
cp .env.example .env            # add GROK_API_KEY
pip install -r requirements.txt
pytest                          # optional: 16 tests, no GPU or API key needed
```

## Run order

**1. Dataset** (local; Streamlit *Dataset* tab, or CLI)
```bash
python -m app.data_prep --n 600            # Grok-generated
python -m app.data_prep --csv my_data.csv  # or your own labelled CSV
```

**2. Train** (GPU machine: Linux, CUDA)
```bash
pip install -r requirements-train.txt      # after a CUDA build of torch
python -m app.train_lora --epochs 3        # add --resume after a spot pre-emption
python -m app.merge_adapter                # optional standalone merged model
```
Copy `outputs/` back, or load `train_log.json` in the *Training* tab for loss curves.
Use `--base-model Qwen/Qwen2.5-0.5B-Instruct` for a fast dry run.

**3. Serve** (GPU machine)
```bash
docker compose up -d                       # vLLM (base + adapter) + gateway
# or:  ./scripts/serve_vllm.sh  and  uvicorn app.gateway:app --port 8004
```

**4. A/B test**
```bash
python -m app.ab_test --limit 100
```

**5. UI**
```bash
streamlit run streamlit_app.py
```

## How the A/B test stays honest

- **Three arms, not two.** `base_fewshot` answers "could I just prompt the base model better?" Fine-tuning only earns its keep if it beats that.
- **Paired bootstrap.** Every arm answers the *same* emails, so the lift is computed per example, removing "some emails are harder" noise.
- **Verdicts follow the CI.** If the CI for `finetuned - baseline` excludes 0, the lift is real at this sample size; if it spans 0 the UI says *inconclusive* rather than declaring a win.
- **Format failures count.** Invalid JSON scores 0 on accuracy *and* on the judge; skipping it would flatter a model that can't follow the format.
- **Cost proxies.** Prompt-token and p95-latency columns show whether the fine-tuned model (no few-shot examples) is cheaper to serve, rather than assuming it.

Caveat: the data is synthetic from one LLM, so it is likely easier and more uniform than real traffic. Treat the numbers as an upper bound and re-check on even a small set of real labelled emails.

## Rollback (stretch goal)

Every gateway response is checked for valid JSON; a sampled fraction is also
scored by the Grok judge in a background task (no added user latency). Over a
rolling window of the fine-tuned arm's traffic, if JSON validity drops below
90% or mean judge score below 0.60 (all configurable), `rollback_active` flips
on and **all traffic goes to base** until an operator clears it. Clearing resets
the window so old bad requests don't instantly re-trigger it.

Try it: in the *Gateway* tab turn on **💥 Demo: degrade fine-tuned model**
(truncates fine-tuned output to simulate a regression), send a burst of ~20
requests, and watch the event log record the automatic rollback.

If the fine-tuned arm is unreachable, the gateway degrades gracefully by
serving that request from base and logging a `fallback` event.

## Quantization

Serve the merged model with vLLM's in-flight 4-bit mode
(`--quantization bitsandbytes`, see `scripts/serve_vllm.sh`) and add it as an
extra arm via `configs/arms.example.json` (`python -m app.ab_test --arms ...`)
to measure the quality/latency trade-off against fp16 with the same harness.

## Scaling policy

`docs/SCALING_POLICY.md` documents the policy (signals, thresholds, cooldowns,
spot-GPU handling), and the Streamlit *Scaling* tab has a capacity planner. These
are a documented policy, not load-tested numbers.

## Layout

```
app/
  config.py          task schema, prompts, few-shot, endpoints, strict output parser
  grok_client.py     Grok via the OpenAI-compatible API, with retries
  judge.py           Grok LLM-as-judge (invalid output scores 0)
  data_prep.py       Grok generation / CSV import, validation, dedup, stratified split
  train_lora.py      QLoRA with PEFT + TRL (GPU)
  merge_adapter.py   merge LoRA into base, report artifact sizes (GPU)
  metrics.py         bootstrap and paired-bootstrap statistics
  ab_test.py         A/B harness
  store.py           SQLite: request log, routing state, events
  gateway.py         FastAPI: traffic split, fallback, live scoring, auto-rollback, admin
streamlit_app.py     UI: dataset, training curves, A/B results, gateway control, scaling
docs/SCALING_POLICY.md   scaling policy
configs/arms.example.json  A/B arms (adds a quantized arm)
scripts/serve_vllm.sh      serve base + adapter (or merged / 4-bit) with vLLM
tests/                     16 tests + a fake vLLM server
```
