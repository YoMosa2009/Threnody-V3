"""Merge a LoRA adapter into Qwen3.5-9B at a chosen strength (task arithmetic).

strength 1.0 = plain merge; 0.5/0.75 interpolate toward the base, which often
keeps the fine-tune's gains while recovering forgotten general skill.

    python merge.py --adapter .../final_adapter --out /content/threnody-merged-0.75 --strength 0.75
    python merge.py --adapter ckA,ckB,ckC --out ...      # average the adapters' LoRA weights first
    python merge.py --adapter ... --out ... --ties_density 0.5   # keep only the largest 50% of each weight delta
    python merge.py --adapter ... --out ... --device cpu         # while the GPU is busy training
"""
import argparse
import json
import os
import shutil
import tempfile

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

BASE = "Qwen/Qwen3.5-9B"


def average_adapters(paths):
    """Average LoRA A/B weights of checkpoints from one run (cheap approximation of averaging their deltas)."""
    out = tempfile.mkdtemp(prefix="avg_adapter_")
    tensors = [load_file(os.path.join(p, "adapter_model.safetensors")) for p in paths]
    avg = {k: sum(t[k].float() for t in tensors) / len(tensors) for k in tensors[0]}
    save_file({k: v.to(tensors[0][k].dtype) for k, v in avg.items()}, os.path.join(out, "adapter_model.safetensors"))
    shutil.copy(os.path.join(paths[0], "adapter_config.json"), out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="adapter dir, or comma-separated dirs to average")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--ties_density", type=float, default=0.0, help="0 = off; else keep top fraction of |delta|")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    paths = a.adapter.split(",")
    adapter = paths[0] if len(paths) == 1 else average_adapters(paths)
    dev = {"": 0} if a.device == "cuda" else {"": "cpu"}
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map=dev)
    base_w = None
    if a.ties_density:
        base_w = {n: p.detach().clone() for n, p in model.named_parameters() if "language_model" in n and p.dim() == 2}
    model = PeftModel.from_pretrained(model, adapter)
    if a.strength != 1.0:
        # Scaling every LoRA B matrix scales the weight delta linearly.
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.data.mul_(a.strength)
    model = model.merge_and_unload()
    if base_w is not None:
        kept = total = 0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in base_w:
                    continue
                d = (p.float() - base_w[n].float())
                if not d.any():
                    continue
                k = max(1, int(d.numel() * (1 - a.ties_density)))
                thr = d.abs().flatten().kthvalue(k).values
                mask = d.abs() > thr
                kept, total = kept + int(mask.sum()), total + d.numel()
                p.copy_((base_w[n].float() + d * mask).to(p.dtype))
        print(f"ties trim: kept {kept / max(total, 1):.1%} of delta entries")
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(BASE).save_pretrained(a.out)
    try:
        AutoProcessor.from_pretrained(BASE).save_pretrained(a.out)
    except Exception as e:
        print("processor not saved:", e)
    # Identity + recommended generation defaults.
    gen = os.path.join(a.out, "generation_config.json")
    cfg = json.load(open(gen)) if os.path.exists(gen) else {}
    cfg.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    json.dump(cfg, open(gen, "w"), indent=2)
    print("merged ->", a.out, "strength", a.strength, "ties", a.ties_density)


if __name__ == "__main__":
    main()
