### Scaling policy (documented, not load-tested)

This is the policy I would deploy with. The numbers are starting points to be
replaced by measurements from a real load test on your GPU type; nothing here
was measured on production hardware.

**What scales.** The vLLM server is the only GPU-bound component. The gateway
is a thin CPU service and scales horizontally behind any load balancer. With
`--enable-lora`, base and fine-tuned share each replica, so scaling is per
replica, not per model.

**Signals (in priority order).**

| Signal | Scale out when | Scale in when |
|---|---|---|
| In-flight requests per replica | > 8 for 2 min | < 3 for 10 min |
| p95 latency (from `/stats`) | > 1.5× the A/B-measured p95 for 2 min | within 1.1× for 10 min |
| GPU utilization | > 80% for 5 min | < 35% for 15 min |

**Rules.**
- Minimum 1 replica, maximum bounded by budget (e.g. 4). Never scale to zero for a live gateway: cold-loading weights takes minutes.
- Scale out fast, scale in slowly: a 10-minute cooldown avoids flapping on bursty traffic.
- Spot GPUs are acceptable for the replicas beyond the first. Keep 1 on-demand replica, and rely on the gateway's fallback + checkpoint `--resume` for pre-emptions.
- Quality gates scaling decisions too: if the automatic rollback has fired, do not scale up the fine-tuned path until the rollback is cleared.

**Sizing.** `replicas = ceil(peak_rps × avg_output_tokens / (tokens_per_sec × target_utilization))`. Use the planner in the Streamlit "Scaling" tab; take `tokens_per_sec` from a batched load test, not from single-request latency.

**Why the fine-tuned model is cheaper to serve.** It needs no few-shot examples, so every request carries fewer prompt tokens than the few-shot baseline. The A/B tab reports mean prompt tokens per arm so this claim is checked rather than assumed.
