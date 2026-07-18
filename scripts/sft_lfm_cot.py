"""LoRA instruction-tune LFM2.5-230M on chain-of-thought <thought><action> targets (Spike B).

Fine-tunes the base torch model (models/lfm2.5-230m-base, already downloaded) on
cot_sft.jsonl so it learns to REASON to the 6-way label instead of guessing —
the fair, apples-to-apples comparison to the fine-tuned encoder that zero-shot
never was. LoRA keeps it CPU-feasible for a 230M model.

Torch is used here for the OFFLINE fine-tune only; the result is merged and
re-exported to ONNX (ort-genai) for torch-free inference — same contract as the
SetFit head. After this: re-run scripts/spike_lfm_classify.py on the re-exported
model vs the encoder.

Pipeline after training:
  python -m onnxruntime_genai.models.builder -i models/lfm2.5-230m-cot-merged \
      -o models/lfm2.5-230m-cot-genai -p int4 -e cpu
  uv run --with onnxruntime-genai python scripts/spike_lfm_classify.py \
      --model-dir models/lfm2.5-230m-cot-genai

Usage (long CPU job — run in your own session):
  uv run --with torch --with transformers --with trl --with peft --with datasets \
      python scripts/sft_lfm_cot.py [--epochs 3] [--batch 8] [--lr 2e-4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "models/lfm2.5-230m-base"
DATA = "research/train_corpus/cot_sft.jsonl"
OUT_ADAPTER = "models/lfm2.5-230m-cot-lora"
OUT_MERGED = "models/lfm2.5-230m-cot-merged"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # low_cpu_mem_usage avoids the transient weight-doubling on load that hit the
    # Windows page-file limit (os error 1455) on the re-run.
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.float32, trust_remote_code=True,
        low_cpu_mem_usage=True)
    # gradient checkpointing is incompatible with KV cache, and with a frozen
    # LoRA base the inputs must be made to require grad or backprop finds nothing.
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ds = load_dataset("json", data_files=args.data, split="train")

    def fmt(ex):
        # Render the {user, assistant} messages through LFM2's chat template so the
        # tuned prompt shape MATCHES what spike_lfm_classify.py sends at inference.
        return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}

    ds = ds.map(fmt, remove_columns=ds.column_names)
    print(f"SFT: {len(ds)} examples | sample:\n{ds[0]['text'][:220]}")

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    )
    cfg = SFTConfig(
        output_dir=OUT_ADAPTER, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=4,
        learning_rate=args.lr, max_length=args.max_seq, logging_steps=20,
        report_to="none", seed=args.seed,
        completion_only_loss=True,  # loss on the <thought><action> completion only
        use_cpu=True, bf16=False, fp16=False,  # CPU training (no GPU here)
        dataloader_num_workers=0,  # avoid Windows multiprocessing teardown noise
        # Memory-constrained CPU box (os error 1455): checkpoint activations, and
        # save periodically so an interrupt doesn't lose the whole run.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=40, save_total_limit=1,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora)
    # Resume from the last checkpoint if a prior run was interrupted (the box is
    # memory-fragile), else start fresh.
    ckpts = list(Path(OUT_ADAPTER).glob("checkpoint-*")) if Path(OUT_ADAPTER).exists() else []
    trainer.train(resume_from_checkpoint=bool(ckpts))

    trainer.save_model(OUT_ADAPTER)
    print(f"saved LoRA adapter -> {OUT_ADAPTER}")

    # Merge LoRA into the base and save full weights for ONNX re-export.
    merged = trainer.model.merge_and_unload()
    Path(OUT_MERGED).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(OUT_MERGED)
    tok.save_pretrained(OUT_MERGED)
    # Carry the chat template alongside for the builder/inference.
    src = Path(args.base) / "chat_template.jinja"
    if src.exists():
        (Path(OUT_MERGED) / "chat_template.jinja").write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"saved merged model -> {OUT_MERGED}")
    print("\nNext: re-export to ONNX then re-run the spike:")
    print(f"  python -m onnxruntime_genai.models.builder -i {OUT_MERGED} "
          f"-o models/lfm2.5-230m-cot-genai -p int4 -e cpu")
    print("  uv run --with onnxruntime-genai python scripts/spike_lfm_classify.py "
          "--model-dir models/lfm2.5-230m-cot-genai")


if __name__ == "__main__":
    main()
