"""
Package the fine-tuned model. Two deployment shapes, pick one:

  A) Keep the adapter (default in docker-compose): vLLM loads the base model
     once and hot-serves the small LoRA adapter next to it (--enable-lora).
     Base and fine-tuned share one GPU, which makes A/B testing cheap.

  B) Merge the adapter into the base weights (this script): a single
     standalone model directory with no adapter indirection. Slightly faster
     per token and simpler to ship, but each variant costs a full model copy.

Quantization for cheaper serving: serve the merged model with vLLM's in-flight
4-bit option (`--quantization bitsandbytes`, see scripts/serve_vllm.sh), then add
it as another arm in configs/arms.example.json to measure the quality/latency
trade-off against fp16 with the same A/B harness.

    python -m app.merge_adapter
"""
import argparse
import json
from pathlib import Path

from app.config import BASE_MODEL_ID, OUTPUT_DIR


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL_ID)
    ap.add_argument("--adapter", default=str(OUTPUT_DIR / "adapter"))
    ap.add_argument("--out", default=str(OUTPUT_DIR / "merged"))
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    # Merge in fp16 onto an unquantized base: merging into 4-bit weights loses precision.
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map=device_map)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(str(out))

    report = {
        "adapter_size_mb": dir_size_mb(Path(args.adapter)),
        "merged_size_mb": dir_size_mb(out),
        "note": "Adapter is shipped alone for shape A; merged dir is a full standalone model (shape B).",
    }
    (Path(args.out).parent / "package_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
