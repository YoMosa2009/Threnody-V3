"""Biology/chemistry gate: MMLU-Pro (TIGER-Lab/MMLU-Pro, MIT) biology + chemistry test questions against an
OpenAI-compatible endpoint (llama-server). Deterministic subset with --n; letter accuracy.

    python sci_eval.py --base-url http://127.0.0.1:8090/v1 --out sci.json --n 200 [--think]
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import urllib.request

LETTERS = "ABCDEFGHIJ"
ANS_RE = re.compile(r"answer is \(?([A-J])\)?", re.I)


def ask(url, q, think, max_tokens):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    prompt = (f"{q['question']}\n\n{opts}\n\nFinish with \"The answer is (X)\" where X is the letter of the correct "
              "option." + ("" if think else " Be brief."))
    body = {"model": "m", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        text = json.load(r)["choices"][0]["message"].get("content") or ""
    m = ANS_RE.findall(text) or re.findall(r"\b([A-J])\b\W*$", text.strip())
    return (m[-1].upper() if m else None) == q["answer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = all biology+chemistry questions")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from datasets import load_dataset
    qs = [r for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test") if r["category"] in ("biology", "chemistry")]
    qs.sort(key=lambda r: r["question_id"])
    if a.n:
        qs = random.Random(0).sample(qs, min(a.n, len(qs)))
    max_tokens = 8192 if a.think else 1024
    res = {"biology": [0, 0], "chemistry": [0, 0]}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(ask, a.base_url, q, a.think, max_tokens): q for q in qs}
        for f in cf.as_completed(futs):
            c = futs[f]["category"]
            try:
                res[c][0] += bool(f.result())
            except Exception:
                pass
            res[c][1] += 1
    out = {c: round(100 * k / max(n, 1), 2) for c, (k, n) in res.items()}
    out["overall"] = round(100 * sum(k for k, _ in res.values()) / max(sum(n for _, n in res.values()), 1), 2)
    out["n"] = sum(n for _, n in res.values())
    out["think"] = a.think
    json.dump(out, open(a.out, "w"), indent=2)
    print("sci", out, flush=True)


if __name__ == "__main__":
    main()
