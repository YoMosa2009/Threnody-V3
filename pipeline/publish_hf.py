"""Upload the exported V2 GGUFs + a model card to Hugging Face (<user>/Threnody-9B-GGUF, public).
Uses the token already stored by huggingface_hub.login() in the notebook kernel (from Colab Secrets: HF_TOKEN).

    python publish_hf.py --gguf_dir $DRIVE/gguf_v2 --final $DRIVE/eval_v2/final.json
"""
import argparse
import json
import os

from huggingface_hub import HfApi

ROWS = [("AgentLoop (14, in-harness)", lambda r: f"{r['al']['passed']}/14"),
        ("AgentLoop held-out harness (opencode, 14)", lambda r: f"{r['al_opencode']['passed']}/14"),
        ("AgentLoop-Hard (held-out, 30)", lambda r: f"{r['hard']['passed']}/{r['hard']['total']}" if "hard" in r else "-"),
        ("OmniAgent-Bench weighted", lambda r: f"{r['omni']['weighted']:.1f}%" if "omni" in r else "-"),
        ("MMLU-Pro biology+chemistry (400)", lambda r: f"{r['sci']['overall']:.1f}%")]


def card(final, user):
    b, c = final["base"]["nothink"], final["cand"]["nothink"]
    status = ("Passed every gate against the same-build base model." if final["passed"] else
              "**Did not pass every gate against the base model** (see table); released as the best V2 candidate.")
    table = "\n".join(f"| {name} | {f(b)} | {f(c)} |" for name, f in ROWS)
    files = "\n".join(f"- `{f}`" for f in sorted(final.get("files", [])))
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
tags: [gguf, agentic-coding, tool-use, llama.cpp, threnody]
---
# Threnody-9B V2 (GGUF)

Fine-tune of Qwen/Qwen3.5-9B for multi-turn agentic coding across many CLI-agent harnesses, plus biology/chemistry.
Training data comes only from June-2026+ open-weight models (Qwen3.8-27B teacher) and the base model's own verified
successes. Code: https://github.com/YoMosa2009/Threnody

**Status:** {status} Chosen candidate: `{final['chosen']}`.

## Results (Q8_0, same build and chat template for both, temperature 0, non-thinking)
| Benchmark | Qwen3.5-9B base | Threnody-9B V2 |
|---|---|---|
{table}

## Files
{files}

## Run
```
llama-server -m Threnody-9B-Q4_K_M.gguf -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --jinja
```
Optional vision: add `--mmproj mmproj-Threnody-9B-F16.gguf`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf_dir", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--repo_name", default="Threnody-9B-GGUF")
    a = ap.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    repo = f"{user}/{a.repo_name}"
    final = json.load(open(a.final))
    final["files"] = [f for f in os.listdir(a.gguf_dir) if f.endswith(".gguf") and "imatrix" not in f]
    open(os.path.join(a.gguf_dir, "README.md"), "w", encoding="utf-8").write(card(final, user))
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(repo_id=repo, folder_path=a.gguf_dir, allow_patterns=["*.gguf", "README.md", "quant_report.json"],
                      ignore_patterns=["*imatrix*"], commit_message="Threnody-9B V2 GGUF")
    final["hf_repo"] = f"https://huggingface.co/{repo}"
    json.dump(final, open(a.final, "w"), indent=2)
    print("uploaded", final["hf_repo"], flush=True)


if __name__ == "__main__":
    main()
