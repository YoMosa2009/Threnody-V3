"""Merged HF checkpoint -> text-only GGUF quant ladder with an agentic imatrix + KLD report.

    python export_gguf.py --hf /content/threnody-merged --llama /content/llama.cpp \
        --calib /content/data/val.jsonl --out /content/drive/MyDrive/Threnody/gguf --name Threnody-9B
"""
import argparse
import json
import os
import subprocess
import sys

from transformers import AutoTokenizer

QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "IQ3_M"]


def sh(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def calib_text(path, tok, n_chars=4_000_000):
    """Render held-out records with the real chat template so the imatrix sees agent traffic."""
    out, total = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = tok.apply_chat_template(r["messages"], tools=r.get("tools") or None, tokenize=False)
            out.append(t)
            total += len(t)
            if total > n_chars:
                break
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--llama", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="Threnody-9B")
    ap.add_argument("--quants", default=",".join(QUANTS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bin_ = os.path.join(a.llama, "build", "bin")
    bf16 = os.path.join("/content", f"{a.name}-BF16.gguf")
    if not os.path.exists(bf16):
        sh([sys.executable, os.path.join(a.llama, "convert_hf_to_gguf.py"), a.hf, "--outfile", bf16, "--outtype", "bf16"])

    tok = AutoTokenizer.from_pretrained(a.hf)
    calib = "/content/calib.txt"
    open(calib, "w", encoding="utf-8").write(calib_text(a.calib, tok))
    imat = os.path.join(a.out, f"{a.name}.imatrix.gguf")
    if not os.path.exists(imat):
        sh([os.path.join(bin_, "llama-imatrix"), "-m", bf16, "-f", calib, "-o", imat, "-ngl", "99", "-c", "4096",
            "--chunks", "250"])

    report = {}
    for q in a.quants.split(","):
        dst = os.path.join(a.out, f"{a.name}-{q}.gguf")
        if not os.path.exists(dst):
            sh([os.path.join(bin_, "llama-quantize"), "--imatrix", imat, bf16, dst, q])
        report[q] = {"bytes": os.path.getsize(dst)}

    # KL divergence of each quant vs BF16 on the agentic calibration text.
    base_logits = "/content/kld_base.bin"
    if not os.path.exists(base_logits):
        sh([os.path.join(bin_, "llama-perplexity"), "-m", bf16, "-f", calib, "-ngl", "99", "-c", "4096",
            "--chunks", "40", "--kl-divergence-base", base_logits])
    for q in report:
        dst = os.path.join(a.out, f"{a.name}-{q}.gguf")
        res = subprocess.run([os.path.join(bin_, "llama-perplexity"), "-m", dst, "-ngl", "99", "-c", "4096",
                              "--chunks", "40", "--kl-divergence-base", base_logits, "--kl-divergence"],
                             capture_output=True, text=True)
        tail = [l for l in (res.stdout + res.stderr).splitlines() if "KLD" in l or "Same top" in l or "Mean" in l]
        report[q]["kld_lines"] = tail[-8:]
        print(q, tail[-4:], flush=True)
    json.dump(report, open(os.path.join(a.out, "quant_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
