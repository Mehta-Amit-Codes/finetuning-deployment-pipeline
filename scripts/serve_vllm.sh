#!/usr/bin/env bash
# Serve the base model + LoRA adapter from ONE vLLM server (one GPU, both A/B arms).
# Run on the GPU machine, from the repo root, after training has produced outputs/adapter.
set -euo pipefail

BASE_MODEL="${BASE_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"

# Shape A: base + hot-served adapter (default). Requests with model=triage-lora use the adapter.
vllm serve "$BASE_MODEL" \
  --enable-lora --max-lora-rank 16 \
  --lora-modules triage-lora=./outputs/adapter \
  --port 8000

# Shape B (alternative): standalone merged model, optionally 4-bit quantized in flight.
# Run on a different port and add it as an arm in configs/arms.example.json.
#   python -m app.merge_adapter
#   vllm serve ./outputs/merged --quantization bitsandbytes --port 8010
