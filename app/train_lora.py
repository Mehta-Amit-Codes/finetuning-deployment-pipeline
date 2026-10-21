"""
QLoRA fine-tuning (4-bit NF4 base + LoRA adapters) with Hugging Face PEFT + TRL.

Run on a Linux GPU box (Colab T4, Modal, RunPod, ...), NOT necessarily your
laptop: bitsandbytes needs CUDA and is awkward on Windows.

    python -m app.train_lora --epochs 3

Outputs in OUTPUT_DIR:
  adapter/           LoRA adapter weights (tens of MB) + tokenizer
  train_log.json     trainer log_history (train loss, eval loss per step) -> Streamlit loss curves
  run_config.json    hyperparameters, trainable-param count, wall-clock time
  checkpoints/       periodic checkpoints so a pre-empted spot instance can --resume

Why QLoRA: the frozen base is held in 4-bit so a 1.5B-7B model trains on one
small GPU; only the small LoRA matrices (~1% of params) get gradients.
"""
import argparse
import inspect
import json
import time
from pathlib import Path

from app.config import BASE_MODEL_ID, DATA_DIR, OUTPUT_DIR


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL_ID)
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint if one exists")
    return ap.parse_args()


def main():
    args = _parse_args()

    # Heavy imports are lazy so the rest of the repo works without torch installed.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Run this on a GPU machine (Colab/Modal/RunPod).")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto", torch_dtype=compute_dtype)
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM")

    ds = load_dataset("json", data_files={
        "train": str(data_dir / "train.jsonl"), "validation": str(data_dir / "val.jsonl")})

    ckpt_dir = out / "checkpoints"
    cfg_kwargs = dict(
        output_dir=str(ckpt_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=5,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", report_to="none", seed=args.seed,
    )
    # TRL renamed max_seq_length -> max_length across versions; support both.
    cfg_params = inspect.signature(SFTConfig).parameters
    cfg_kwargs["max_length" if "max_length" in cfg_params else "max_seq_length"] = args.max_len
    sft_config = SFTConfig(**cfg_kwargs)

    trainer_kwargs = dict(
        model=model, args=sft_config, train_dataset=ds["train"],
        eval_dataset=ds["validation"], peft_config=lora)
    # ...and tokenizer -> processing_class.
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())

    resume = args.resume and any(ckpt_dir.glob("checkpoint-*"))
    if args.resume and not resume:
        print("--resume given but no checkpoint found; starting fresh.")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    train_seconds = time.time() - t0

    final_eval = trainer.evaluate()
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))

    (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (out / "run_config.json").write_text(json.dumps({
        "base_model": args.base_model, "lora_rank": args.rank, "lora_alpha": args.alpha,
        "learning_rate": args.lr, "epochs": args.epochs,
        "effective_batch_size": args.batch_size * args.grad_accum, "max_len": args.max_len,
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "train_seconds": round(train_seconds, 1), "final_eval": final_eval,
        "gpu": torch.cuda.get_device_name(0),
    }, indent=2))
    print(f"Done in {train_seconds:.0f}s. Adapter saved to {out / 'adapter'}")


if __name__ == "__main__":
    main()
