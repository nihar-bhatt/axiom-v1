"""
Supervised fine-tuning (SFT) trainer for Axiom v1 datasets.

Input format:
- A JSONL file where each line is a JSON object with:
  - "messages": list of {"role": "system"|"user"|"assistant", "content": str}
  - optional "meta": arbitrary metadata

This script:
1) Loads JSONL as a Hugging Face Dataset
2) Converts each example into model tokens using the model's chat template
3) Masks labels so ONLY the assistant answer is trained (prefix is -100)
4) Fine-tunes a pretrained model using LoRA (PEFT)
5) Saves the LoRA adapter (and tokenizer) to output_dir

Run (example):
python3 -m src.train.sft_train \
  --data data/processed/aops_sft_2k.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output_dir runs/axiom_aops_2k_lora \
  --max_seq_len 2048 \
  --epochs 1

Notes:
- Training requires a GPU for anything non-trivial.
- On Mac CPU/MPS this will be slow or may fail for large models.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model



def _last_assistant_index(messages: List[Dict[str, str]]) -> Optional[int]:
    """Return index of the last assistant message, or None."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    return None


def _apply_chat_template_ids(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> List[int]:
    """
    Convert structured chat messages to token IDs using the tokenizer's chat template.
    Falls back to a simple format if chat templates aren't available.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    chunks = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        chunks.append(f"{role}:\n{content}")
    if add_generation_prompt:
        chunks.append("ASSISTANT:\n")
    text = "\n\n".join(chunks)
    return tokenizer(text, add_special_tokens=True)["input_ids"]


def _tokenize_and_mask_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
) -> Dict[str, Any]:
    """
    Produces:
      input_ids, attention_mask, labels

    labels are -100 for everything up to (and including) the assistant "start",
    and normal token IDs for the assistant answer tokens.
    """
    messages = example.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    ai = _last_assistant_index(messages)
    if ai is None:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    prefix_msgs = messages[:ai]  # everything before the assistant answer
    full_msgs = messages         # includes assistant answer

    prefix_ids = _apply_chat_template_ids(
        tokenizer,
        prefix_msgs,
        add_generation_prompt=True,
    )
    full_ids = _apply_chat_template_ids(
        tokenizer,
        full_msgs,
        add_generation_prompt=False,
    )

    prefix_len = min(len(prefix_ids), len(full_ids))
    labels = [-100] * prefix_len + full_ids[prefix_len:]
    attn = [1] * len(full_ids)

    if len(full_ids) > max_seq_len:
        shift = len(full_ids) - max_seq_len
        full_ids = full_ids[shift:]
        attn = attn[shift:]
        labels = labels[shift:]

    return {"input_ids": full_ids, "attention_mask": attn, "labels": labels}



@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features = [f for f in features if f.get("input_ids")]
        if not features:
            # This should not happen often; Trainer will crash if batches are empty repeatedly.
            return {
                "input_ids": torch.zeros((1, 1), dtype=torch.long),
                "attention_mask": torch.zeros((1, 1), dtype=torch.long),
                "labels": torch.full((1, 1), -100, dtype=torch.long),
            }

        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attn = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        max_len = max(x.size(0) for x in input_ids)

        def pad_1d(x: torch.Tensor, pad_value: int) -> torch.Tensor:
            if x.size(0) == max_len:
                return x
            pad = torch.full((max_len - x.size(0),), pad_value, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        input_ids = torch.stack([pad_1d(x, self.pad_token_id) for x in input_ids], dim=0)
        attn = torch.stack([pad_1d(x, 0) for x in attn], dim=0)
        labels = torch.stack([pad_1d(x, -100) for x in labels], dim=0)

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}



def guess_lora_targets(model) -> List[str]:
    """
    Try to pick common projection module names across popular decoder-only LMs.
    This avoids hardcoding to exactly one architecture.
    """
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj",  # GPT-like
        "Wqkv", "wo", "wq", "wk", "wv",  # some variants
    ]

    found = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            found.add(leaf)

    llama_set = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if llama_set.issubset(found):
        # include MLP projections too if available
        extras = ["gate_proj", "up_proj", "down_proj"]
        return sorted(list(llama_set.union(set(e for e in extras if e in found))))

    if found:
        return sorted(found)

    raise ValueError(
        "Could not guess LoRA target modules for this model. "
        "Pick a standard instruct/chat causal LM, or add a manual target list."
    )



def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA SFT trainer for Axiom v1 JSONL corpora.")
    ap.add_argument("--data", required=True, help="Path to JSONL created by build_corpus (messages format).")
    ap.add_argument("--model", required=True, help="HF model id (pretrained instruct/chat causal LM).")
    ap.add_argument("--output_dir", required=True, help="Where to save the LoRA adapter + tokenizer.")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1, help="Per-device train batch size.")
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="If >0, limit number of examples (useful for smoke tests).")
    ap.add_argument("--trust_remote_code", action="store_true", help="Enable for some models (Qwen, etc.).")
    ap.add_argument("--no_lora", action="store_true", help="Train full model (NOT recommended).")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        # Many causal LMs have no pad token; use EOS as padding
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=args.data, split="train")
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    split = ds.train_test_split(test_size=args.val_frac, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    use_cuda = torch.cuda.is_available()
    bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    fp16 = bool(use_cuda and not bf16)
    dtype = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if not args.no_lora:
        target_modules = guess_lora_targets(model)
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    def map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        return _tokenize_and_mask_example(ex, tokenizer, args.max_seq_len)

    train_tok = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    eval_tok = eval_ds.map(map_fn, remove_columns=eval_ds.column_names)

    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)

    use_cuda = torch.cuda.is_available()
    bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    fp16 = bool(use_cuda and not bf16)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=20,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        report_to="none",  # keep it simple; you can enable wandb later
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
