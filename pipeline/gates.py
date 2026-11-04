"""Same-build GGUF + evaluation suites + hard gates for V2.

Every model (base and candidates) goes through the SAME build and serving path:
  convert_hf_to_gguf (bf16) -> [llama-imatrix on agentic calibration text] -> llama-quantize
  llama-server with the pinned chat template (templates/qwen35_chat_template.jinja) and an EXPLICIT
  enable_thinking in both modes.

    python gates.py build --src /content/merged --name cand --quant Q4_K_M --calib val.jsonl --outdir /content/models
    python gates.py eval  --gguf /content/models/cand-Q4_K_M.gguf --suite full --out .../eval_cand
    python gates.py gate  --cand .../eval_cand/results.json --base .../eval_base/results.json --out gate.json

Suites
  fast (checkpoint gate, nothink): AgentLoop[agentloop] 14, AgentLoop[opencode, held-out] 14, sci 200
  full (final gate, nothink+think): OmniAgent-Bench, AgentLoop[agentloop], AgentLoop[opencode]+[codex] (held-out),
        AgentLoop-Hard (hard_eval.jsonl), MMLU-Pro biology+chemistry
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "templates", "qwen35_chat_template.jinja")
HARNESS = os.path.join(HERE, "..", "eval", "agentloop", "harness.py")
LLAMA = os.environ.get("LLAMA", "/content/llama.cpp")
BENCH = os.environ.get("BENCH", "/content/OmniAgent-Bench")
DRIVE = os.environ.get("DRIVE", "/content/drive/MyDrive/Threnody")
HARD = os.path.join(os.environ.get("TEACHER_DIR", os.path.join(DRIVE, "teacher")), "hard_eval.jsonl")
DRY = os.environ.get("GATES_DRY") == "1"  # dry run: 2 tasks per AgentLoop run, 20 sci questions


def pin(model_dir):
    """Embed the pinned chat template in an HF dir so convert_hf_to_gguf writes it into the GGUF."""
    tpl = open(TEMPLATE, encoding="utf-8").read()
    open(os.path.join(model_dir, "chat_template.jinja"), "w", encoding="utf-8").write(tpl)
    p = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(p):
        cfg = json.load(open(p, encoding="utf-8"))
        cfg["chat_template"] = tpl
        json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def candidates(run_dir):
    """Lines '<name> <adapter[,adapter..]> <strength> <ties_density>' for the selection stage."""
    import glob
    keep = run_dir + "_keep"
    st_p = os.path.join(run_dir, "gate_state.json")
    best_n = (json.load(open(st_p)).get("best") if os.path.exists(st_p) else None)
    final = os.path.join(run_dir, "final_adapter")
    best = os.path.join(keep, f"checkpoint-{best_n}") if best_n else final
    out = [f"best_s1.0 {best} 1.0 0", f"best_s0.75 {best} 0.75 0", f"best_ties {best} 1.0 0.5"]
    if best != final and os.path.exists(final):
        out.append(f"final_s1.0 {final} 1.0 0")
    if best_n:
        near = [p for p in glob.glob(os.path.join(keep, "checkpoint-*"))
                if abs(int(p.rsplit("-", 1)[1]) - int(best_n)) <= 40]
        if len(near) >= 2:
            out.append(f"avg {','.join(sorted(near))} 1.0 0")
    if not os.path.exists(best):
        raise SystemExit(1)
    return out


def rank(eval_dir):
    import glob
    base = fast_score(json.load(open(os.path.join(eval_dir, "base_fast", "results.json"))))
    rows = sorted(((fast_score(json.load(open(p))), p.split(os.sep)[-2][5:])
                   for p in glob.glob(os.path.join(eval_dir, "cand_*", "results.json"))), reverse=True)
    return [f"{n} {s:.2f} base {base:.2f}" for s, n in rows]


def sh(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)


# ---------------------------------------------------------------- build
def calib_file(calib_jsonl, path, n_chars=1_500_000):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    out, total = [], 0
    for line in open(calib_jsonl, encoding="utf-8"):
        r = json.loads(line)
        t = tok.apply_chat_template(r["messages"], tools=r.get("tools") or None, tokenize=False)
        out.append(t)
        total += len(t)
        if total > n_chars:
            break
    open(path, "w", encoding="utf-8").write("\n".join(out))
    return path


def build(src, name, quant, outdir, calib=None):
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, f"{name}-{quant}.gguf")
    if os.path.exists(dst) and os.path.getsize(dst) > 1e9:
        return dst
    bf16 = f"/content/tmp_{name}_bf16.gguf"
    sh([sys.executable, os.path.join(LLAMA, "convert_hf_to_gguf.py"), src, "--outfile", bf16, "--outtype", "bf16"],
       stdout=subprocess.DEVNULL)
    cmd = [os.path.join(LLAMA, "build", "bin", "llama-quantize")]
    if calib and quant not in ("Q8_0", "BF16"):
        imat = os.path.join(outdir, f"{name}.imatrix.gguf")
        cf = calib_file(calib, f"/content/calib_{name}.txt")
        sh([os.path.join(LLAMA, "build", "bin", "llama-imatrix"), "-m", bf16, "-f", cf, "-o", imat, "-ngl", "99",
            "-c", "4096", "--chunks", "100"], stdout=subprocess.DEVNULL)
        cmd += ["--imatrix", imat]
    sh(cmd + [bf16, dst, quant], stdout=subprocess.DEVNULL)
    os.remove(bf16)
    if not os.path.exists(dst) or os.path.getsize(dst) < 1e9:
        raise RuntimeError(f"build failed: {dst}")
    return dst


# ---------------------------------------------------------------- serve + eval
class Server:
    def __init__(self, gguf, think, port, log, slots=4):
        cmd = [os.path.join(LLAMA, "build", "bin", "llama-server"), "-m", gguf, "-ngl", "99", "-c", str(32768 * slots),
               "-np", str(slots), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "--chat-template-file", TEMPLATE,
               "--chat-template-kwargs", json.dumps({"enable_thinking": think}), "--port", str(port),
               "--host", "127.0.0.1", "--temp", "0", "--seed", "0"]
        self.port, self.proc = port, subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{port}/v1"

    def ready(self, timeout=600):
        for _ in range(timeout):
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def agentloop(url, out, think, schema, tasks_file=""):
    cmd = [sys.executable, HARNESS, "--base-url", url, "--name", "x", "--out", out, "--schema", schema]
    if think:
        cmd += ["--think"]
    if tasks_file:
        cmd += ["--tasks_file", tasks_file]
    if DRY:
        cmd += ["--max_tasks", "2"]
    subprocess.run(cmd, check=False)
    s = json.load(open(os.path.join(out, "agentloop_summary.json")))
    return {"passed": s["passed"], "total": s["total"], "max_repeat": s["max_repeat"]}


def omni(url, out, think, tag):
    name = f"{tag} [{'think' if think else 'nothink'}]"
    subprocess.run([sys.executable, "run_bench.py", "--model", name, "--backend", "openai", "--base-url", url,
                    "--temperature", "0", "--max-tokens", "16384" if think else "1024"], cwd=BENCH, check=False)
    rep = os.path.join(BENCH, "reports", f"{name}_report.json")
    r = json.load(open(rep))
    shutil.copy(rep, out)
    return {"weighted": r["weighted_score_pct"], "passed": r["tasks_passed"], "total": r["total_tasks"]}


def sci(url, out, think, n):
    p = os.path.join(out, f"sci_{'think' if think else 'nothink'}.json")
    cmd = [sys.executable, os.path.join(HERE, "sci_eval.py"), "--base-url", url, "--out", p, "--n", str(20 if DRY else n)]
    if think:
        cmd += ["--think"]
    subprocess.run(cmd, check=True)
    return json.load(open(p))


def evaluate(gguf, suite, out, tag="model", port=8090):
    os.makedirs(out, exist_ok=True)
    res_path = os.path.join(out, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    modes = ["nothink", "think"] if suite in ("full", "v3") else ["nothink"]  # "mid": nothink, + hard + OmniAgent
    for mode in modes:
        think = mode == "think"
        r = res.get(mode, {})
        srv = Server(gguf, think, port, os.path.join(out, f"server_{mode}.log"))
        try:
            if not srv.ready():
                raise RuntimeError(f"llama-server failed to start ({mode}); see server_{mode}.log")
            d = lambda k: os.path.join(out, f"al_{k}_{mode}")  # noqa: E731
            if "al" not in r:
                r["al"] = agentloop(srv.url, d("agentloop"), think, "agentloop")
            if "al_opencode" not in r:
                r["al_opencode"] = agentloop(srv.url, d("opencode"), think, "opencode")
            if suite in ("mid", "full", "v3") and "hard" not in r and os.path.exists(HARD):
                r["hard"] = agentloop(srv.url, d("hard"), think, "agentloop", HARD)
            if suite in ("mid", "full", "v3") and "omni" not in r:
                r["omni"] = omni(srv.url, out, think, tag)
            if suite == "full":
                if "al_codex" not in r:
                    r["al_codex"] = agentloop(srv.url, d("codex"), think, "codex")
                if "hard" not in r and os.path.exists(HARD):
                    r["hard"] = agentloop(srv.url, d("hard"), think, "agentloop", HARD)
                if "omni" not in r:
                    r["omni"] = omni(srv.url, out, think, tag)
            if "sci" not in r:
                r["sci"] = sci(srv.url, out, think, 200 if suite == "fast" else 400 if suite == "mid" else (0 if not think else 300))
        finally:
            srv.stop()
        res[mode] = r
        json.dump(res, open(res_path, "w"), indent=2)
    return res


def fast_score(r):
    """Single number for checkpoint gating (nothink): AgentLoop + held-out passes + sci accuracy scaled to 14."""
    n = r["nothink"]
    return n["al"]["passed"] + n["al_opencode"]["passed"] + 14 * n["sci"]["overall"] / 100


# ---------------------------------------------------------------- gate

def gate_v3(cand, base):
    checks = []

    def chk(name, c, b, cond):
        checks.append({"gate": name, "cand": c, "target": b, "pass": bool(cond)})

    c_nt = cand.get("nothink", cand)
    b_nt = base.get("nothink", base)

    # 1. OmniAgent-Bench: >= 93.8% and >= 92.7% weighted score
    omni_c = c_nt.get("omni", {}).get("weighted", 0.0)
    omni_b = b_nt.get("omni", {}).get("weighted", 92.71)
    chk("OmniAgent-Bench weighted >= 93.8%", omni_c, 93.8, omni_c >= 93.8)
    chk("OmniAgent-Bench beats Base", omni_c, omni_b, omni_c >= omni_b)

    # 2. AgentLoop In-Harness: >= 13/14 passed (beats Base 12/14); max_repeat <= 2
    al_c = c_nt.get("al", {}).get("passed", 0)
    al_b = b_nt.get("al", {}).get("passed", 12)
    rep_al = c_nt.get("al", {}).get("max_repeat", 99)
    chk("AgentLoop In-Harness >= 13/14", f"{al_c}/14", "13/14", al_c >= 13)
    chk("AgentLoop In-Harness beats Base", f"{al_c}/14", f"{al_b}/14", al_c > al_b or (al_c >= al_b and al_c >= 13))
    chk("AgentLoop In-Harness max_repeat <= 2", rep_al, 2, rep_al <= 2)

    # 3. AgentLoop Held-Out (opencode): >= 10/14 passed (beats Base 9/14); max_repeat <= 2
    op_c = c_nt.get("al_opencode", {}).get("passed", 0)
    op_b = b_nt.get("al_opencode", {}).get("passed", 9)
    rep_op = c_nt.get("al_opencode", {}).get("max_repeat", 99)
    chk("AgentLoop Held-Out (opencode) >= 10/14", f"{op_c}/14", "10/14", op_c >= 10)
    chk("AgentLoop Held-Out beats Base", f"{op_c}/14", f"{op_b}/14", op_c > op_b or (op_c >= op_b and op_c >= 10))
    chk("AgentLoop Held-Out max_repeat <= 2", rep_op, 2, rep_op <= 2)

    # 4. BioChem-Hard: >= 40.0% overall, Chemistry >= 15.0%
    bch_c = c_nt.get("biochem_hard") or c_nt.get("sci", {})
    overall_c = bch_c.get("overall", 0.0)
    chem_c = bch_c.get("chemistry", 0.0)
    chk("BioChem-Hard overall >= 40.0%", f"{overall_c:.1f}%", "40.0%", overall_c >= 40.0)
    chk("BioChem-Hard chemistry >= 15.0%", f"{chem_c:.1f}%", "15.0%", chem_c >= 15.0)

    return {"pass": all(x["pass"] for x in checks), "checks": checks}

def gate(cand, base, suite="full"):
    if suite == "v3":
        return gate_v3(cand, base)
    checks = []

    def chk(name, c, b, strict=False):
        ok = c > b if strict else c >= b
        checks.append({"gate": name, "cand": c, "base": b, "pass": ok})

    worst_rep = 0
    modes = ["nothink", "think"] if suite == "full" else ["nothink"]
    for mode in modes:
        c, b = cand[mode], base[mode]
        chk(f"{mode}: AgentLoop", c["al"]["passed"], b["al"]["passed"])
        held_c = c["al_opencode"]["passed"] + c.get("al_codex", {}).get("passed", 0)
        held_b = b["al_opencode"]["passed"] + b.get("al_codex", {}).get("passed", 0)
        chk(f"{mode}: AgentLoop held-out harnesses", held_c, held_b)
        chk(f"{mode}: MMLU-Pro bio+chem", c["sci"]["overall"], b["sci"]["overall"])
        if suite in ("mid", "full"):
            chk(f"{mode}: OmniAgent weighted", c["omni"]["weighted"], b["omni"]["weighted"])
            if "hard" in c and "hard" in b:
                chk(f"{mode}: AgentLoop-Hard", c["hard"]["passed"], b["hard"]["passed"])
        worst_rep = max([worst_rep] + [v["max_repeat"] for k, v in c.items() if isinstance(v, dict) and "max_repeat" in v])
    if suite in ("mid", "full") and all("hard" in cand[m] and "hard" in base[m] for m in modes):
        chk("AgentLoop-Hard total (must be strictly better)", sum(cand[m]["hard"]["passed"] for m in modes),
            sum(base[m]["hard"]["passed"] for m in modes), strict=True)
    checks.append({"gate": "no action repeated >= 5 times", "cand": worst_rep, "base": 5, "pass": worst_rep < 5})
    return {"pass": all(x["pass"] for x in checks), "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--name", required=True)
    b.add_argument("--quant", default="Q4_K_M")
    b.add_argument("--calib", default="")
    b.add_argument("--outdir", default="/content/models")
    e = sub.add_parser("eval")
    e.add_argument("--gguf", required=True)
    e.add_argument("--suite", choices=["fast", "mid", "full", "v3"], default="full")
    e.add_argument("--out", required=True)
    e.add_argument("--tag", default="model")
    e.add_argument("--port", type=int, default=8090)
    g = sub.add_parser("gate")
    g.add_argument("--cand", required=True)
    g.add_argument("--base", required=True)
    g.add_argument("--suite", choices=["fast", "mid", "full", "v3"], default="full")
    g.add_argument("--out", required=True)
    sub.add_parser("pin").add_argument("--dir", required=True)
    sub.add_parser("candidates").add_argument("--run_dir", required=True)
    sub.add_parser("rank").add_argument("--eval_dir", required=True)
    a = ap.parse_args()
    if a.cmd == "pin":
        pin(a.dir)
    elif a.cmd == "candidates":
        print("\n".join(candidates(a.run_dir)))
    elif a.cmd == "rank":
        print("\n".join(rank(a.eval_dir)))
    elif a.cmd == "build":
        print(build(a.src, a.name, a.quant, a.outdir, a.calib or None))
    elif a.cmd == "eval":
        print(json.dumps(evaluate(a.gguf, a.suite, a.out, a.tag, a.port), indent=1))
    else:
        r = gate(json.load(open(a.cand)), json.load(open(a.base)), a.suite)
        json.dump(r, open(a.out, "w"), indent=2)
        for x in r["checks"]:
            print(f"{'PASS' if x['pass'] else 'FAIL'}  {x['gate']:48s} cand={x['cand']}  base={x['base']}")
        print("GATE", "PASSED" if r["pass"] else "FAILED")
        sys.exit(0 if r["pass"] else 2)


if __name__ == "__main__":
    main()
