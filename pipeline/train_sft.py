"""Threnody SFT: bf16 LoRA on Qwen3.5-9B with assistant-only loss.

Designed for a single Colab A100 (40 or 80 GB). Key memory tricks:
  * gradient checkpointing on the language model
  * the LM head + cross-entropy are computed in chunks, so 16k x 248k
    logits are never materialized at once
  * batch size 1 (no padding waste) with gradient accumulation

Resumes automatically from the newest checkpoint in --out.

    python train_sft.py --data /content/data --out /content/drive/MyDrive/Threnody/sft \
        --max_len 16384 --epochs 2 --lr 1.5e-4
    python train_sft.py ... --max_steps 60      # pilot: measure throughput
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForImageTextToText, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_data import encode, load_jsonl  # noqa: E402

BASE = "Qwen/Qwen3.5-9B"
LORA_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
               "in_proj_qkv", "in_proj_z", "in_proj_qkvz", "in_proj_ba", "in_proj_b", "in_proj_a", "out_proj")


def lora_targets(model):
    """Every Linear in the language model's decoder layers (not vision, not MTP, not lm_head)."""
    names = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "language_model" in n and ".layers." in n \
                and n.rsplit(".", 1)[-1] in LORA_LEAVES:
            names.append(n)
    return names


class Collator:
    def __call__(self, feats):
        f = feats[0]  # batch size 1
        return {"input_ids": torch.tensor([f["input_ids"]]), "labels": torch.tensor([f["labels"]])}


class ChunkedLossTrainer(Trainer):
    chunk = 2048

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        out = model(input_ids=inputs["input_ids"], output_hidden_states=True, logits_to_keep=1, use_cache=False)
        h = out.hidden_states[-1][:, :-1]
        y = labels[:, 1:]
        lm_head = model.get_output_embeddings()
        total = torch.zeros((), device=h.device, dtype=torch.float32)
        n_tok = (y != -100).sum()
        for s in range(0, h.shape[1], self.chunk):
            ys = y[:, s:s + self.chunk]
            if (ys != -100).any():
                total = total + torch.utils.checkpoint.checkpoint(
                    _chunk_ce, h[:, s:s + self.chunk], lm_head.weight, ys, use_reentrant=False)
        denom = num_items_in_batch if num_items_in_batch is not None else n_tok.clamp(min=1)
        loss = total / denom
        return (loss, out) if return_outputs else loss


def _chunk_ce(h, w, y):
    logits = F.linear(h.to(w.dtype), w).float()  # final norm may upcast h to fp32 outside autocast
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum")


class TimeLimit(TrainerCallback):
    """Stop cleanly after max_hours of training; the final adapter is still saved."""

    def __init__(self, max_hours):
        self.deadline = time.time() + max_hours * 3600

    def on_step_end(self, args, state, control, **kw):
        if time.time() > self.deadline:
            control.should_training_stop = True
            control.should_save = True


class StopFile(TrainerCallback):
    """Stop cleanly when the checkpoint gate (ckpt_gate.py) writes the STOP file."""

    def __init__(self, path):
        self.path = path

    def on_step_end(self, args, state, control, **kw):
        if self.path and os.path.exists(self.path):
            print(f"STOP file found at step {state.global_step}: {open(self.path).read().strip()}", flush=True)
            control.should_training_stop = True
            control.should_save = True


class Throughput(TrainerCallback):
    def __init__(self, tok_per_step):
        self.t0, self.tok_per_step = None, tok_per_step

    def on_step_begin(self, args, state, control, **kw):
        if self.t0 is None:
            self.t0, self.s0 = time.time(), state.global_step

    def on_log(self, args, state, control, logs=None, **kw):
        if self.t0 and state.global_step > self.s0:
            dt = time.time() - self.t0
            steps = state.global_step - self.s0
            logs["sec_per_step"] = round(dt / steps, 2)
            logs["tok_per_sec"] = round(steps * self.tok_per_step / dt)
            if torch.cuda.is_available():
                logs["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=16384)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--save_steps", type=int, default=20)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--max_hours", type=float, default=1e9)
    ap.add_argument("--stop_file", default="", help="training stops when this file appears (checkpoint gate)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    cache = os.path.join(args.data, f"tokenized_{args.max_len}.pt")
    if os.path.exists(cache):
        train, val = torch.load(cache)
    else:
        train = [encode(tok, r, args.max_len) for r in load_jsonl(os.path.join(args.data, "train.jsonl"))]
        val = [encode(tok, r, args.max_len) for r in load_jsonl(os.path.join(args.data, "val.jsonl"))]
        train = [x for x in train if any(l != -100 for l in x["labels"])]
        val = [x for x in val if any(l != -100 for l in x["labels"])]
        torch.save((train, val), cache)
    random.Random(0).shuffle(train)
    n_tok = sum(len(x["input_ids"]) for x in train)
    print(f"train={len(train)} val={len(val)} tokens={n_tok/1e6:.1f}M avg={n_tok//len(train)}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0} if torch.cuda.is_available() else None)
    model.config.use_cache = False
    targets = lora_targets(model)
    print(f"LoRA on {len(targets)} linear modules, e.g. {sorted({t.rsplit('.',1)[-1] for t in targets})}", flush=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        seed=42,
    )
    trainer = ChunkedLossTrainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                                 data_collator=Collator(),
                                 callbacks=[Throughput(n_tok / len(train) * args.grad_accum), TimeLimit(args.max_hours),
                                            StopFile(args.stop_file)])
    resume = bool(glob.glob(os.path.join(args.out, "checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume or None)
    final = os.path.join(args.out, "final_adapter")
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    json.dump(trainer.state.log_history, open(os.path.join(args.out, "log_history.json"), "w"), indent=1)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
