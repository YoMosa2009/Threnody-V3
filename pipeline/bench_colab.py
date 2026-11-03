"""Run OmniAgent-Bench (unmodified) against GGUF models via llama-server on Colab.

Every model is served the same way: llama.cpp CUDA build, all layers on GPU,
32k context, flash attention, q8_0 KV cache, temperature 0. Qwen3.5-family
models are run twice: non-thinking with the bench's default 1024 max tokens,
and thinking with a 16k budget (reasoning returned separately by the server,
so graders only see the final answer).

    python bench_colab.py --llama /content/llama.cpp/build/bin --bench /content/OmniAgent-Bench \
        --models models.json --out /content/drive/MyDrive/Threnody/bench
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request


def wait_ready(port, proc, timeout=600):
    for _ in range(timeout):
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_one(a, m, mode):
    port = 8090
    tag = f"{m['id']}__{mode}"
    cmd = [os.path.join(a.llama, "llama-server"), "-m", m["path"], "-ngl", "99", "-c", "32768", "-np", "1",
           "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "--port", str(port), "--host", "127.0.0.1",
           "--temp", "0", "--seed", "0"]
    if mode == "nothink":
        cmd += ["--chat-template-kwargs", json.dumps({"enable_thinking": False})]
    log = open(os.path.join(a.out, f"{tag}_server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(port, proc):
            print("server failed for", tag)
            return None
        max_tokens = "16384" if mode == "think" else "1024"
        name = f"{m['name']} [{mode}]"
        subprocess.run([sys.executable, "run_bench.py", "--model", name, "--backend", "openai",
                        "--base-url", f"http://127.0.0.1:{port}/v1", "--temperature", "0",
                        "--max-tokens", max_tokens], cwd=a.bench, check=False)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    rep = os.path.join(a.bench, "reports", f"{m['name']} [{mode}]_report.json")
    if os.path.exists(rep):
        shutil.copy(rep, a.out)
        return json.load(open(rep))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary_path = os.path.join(a.out, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for m in json.load(open(a.models)):
        for mode in m.get("modes", ["default"]):
            key = f"{m['id']}__{mode}"
            if key in summary:
                continue
            r = run_one(a, m, mode)
            if r:
                summary[key] = {k: r[k] for k in ("model_name", "tasks_passed", "total_tasks", "accuracy_pct",
                                                  "weighted_score_pct", "category_breakdown")}
                summary[key]["per_task"] = {t["id"]: round(t["points_earned"], 2) for t in r["task_results"]}
                json.dump(summary, open(summary_path, "w"), indent=2)
    for k, v in summary.items():
        print(f"{k:40s} pass {v['tasks_passed']}/{v['total_tasks']}  weighted {v['weighted_score_pct']:.1f}%")


if __name__ == "__main__":
    main()
