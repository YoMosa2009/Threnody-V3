"""Merge a LoRA adapter into Qwen3.5-9B at a chosen strength (task arithmetic).

strength 1.0 = plain merge; 0.5/0.75 interpolate toward the base, which often
keeps the fine-tune's gains while recovering forgotten general skill.

    python merge.py --adapter .../final_adapter --out /content/threnody-merged-0.75 --strength 0.75
"""
import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

BASE = "Qwen/Qwen3.5-9B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    a = ap.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, a.adapter)
    if a.strength != 1.0:
        # Scaling every LoRA B matrix scales the weight delta linearly.
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.data.mul_(a.strength)
    model = model.merge_and_unload()
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
    print("merged ->", a.out, "strength", a.strength)


if __name__ == "__main__":
    main()
