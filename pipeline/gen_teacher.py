"""Generate verified training data with a June-2026+ teacher (default: Qwen3.8-27B-FP8 served by vLLM).

Modes (all resumable; outputs are JSONL under --out):
  synth_tasks   teacher invents AgentLoop-style repo tasks (files + hidden tests + reference solution);
                kept only if the task fails as shipped and the reference passes. -> tasks_train.jsonl
  solve_tasks   teacher solves those tasks in the real tool-calling harness, k attempts each; kept only if
                hidden tests pass AND the agent ran python/pytest after its last edit. -> agent_traj.jsonl
  code          teacher solves deepmind/code_contests problems; kept only if all tests pass. -> code_verified.jsonl
  regen         teacher writes fresh answers for tool-use / security / IF / chat prompts. -> regen.jsonl

    python gen_teacher.py --mode synth_tasks --n 600 --out /content/drive/MyDrive/Threnody/teacher
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval", "agentloop"))
sys.path.insert(0, HERE)
import harness  # noqa: E402
from tasks import TASKS as EVAL_TASKS  # noqa: E402
from validate import grade  # noqa: E402

URL = os.environ.get("TEACHER_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("TEACHER_MODEL", "teacher")
LOCK = threading.Lock()

CATEGORIES = {
    "debugging": "a bug reported only by its symptom (traceback or wrong output); the root cause lives in a different "
                 "module from where it surfaces",
    "feature": "a feature spanning 2-3 files that must integrate with existing code and keep old behaviour intact",
    "concurrency": "a threading or asyncio correctness problem (race, deadlock, ordering, cancellation)",
    "security": "a defensive security fix (injection, traversal, unsafe deserialization, secrets in logs, SSRF "
                "allow-listing, timing-safe comparison, input validation)",
    "performance": "an algorithmic performance problem with a hidden test that times out on the slow version",
    "refactor": "a behaviour-preserving refactor (rename/extract/split module) with backwards compatibility",
    "data": "parsing/serialization edge cases (CSV, JSON, dates/timezones, unicode, decimals)",
    "cli": "writing or extending a small command-line tool with exact output format and exit codes",
    "sql": "sqlite schema/migration/query work with transactions and idempotency",
    "spec": "implementing a function from a precise docstring contract with many edge cases",
    "insight": "a trap task: the obvious fix makes the visible test pass but breaks a hidden invariant; solving it "
               "needs noticing a non-obvious constraint in other code, docs or data",
    "algorithmic": "a problem where the straightforward solution is far too slow or wrong on edge cases and a clever "
                   "algorithm or data structure is required (hidden tests include large inputs with a time limit)",
    "persistence": "a bug with a misleading first symptom: the first plausible fix fails a hidden test and the agent must "
                   "keep investigating (logs, minimal repro, bisecting inputs) to reach the real root cause",
}
DOMAINS = ["billing", "inventory", "auth", "scheduling", "logging", "caching", "messaging queue", "file storage",
           "metrics", "rate limiting", "search index", "config loading", "event sourcing", "workflow engine",
           "notification service", "feature flags", "geometry", "text processing", "matrix utils", "graph routing",
           "bank ledger", "chat moderation", "package resolver", "markdown renderer", "job retry policy"]

SYNTH_PROMPT = """Design ONE realistic, self-contained Python coding-agent task for training an AI coding agent.

Category: {cat} — {cat_desc}
Domain: {domain}
Difficulty: {diff}

Requirements:
- A small repository (2-6 files, standard library only, Python 3.11+) that a CLI coding agent must inspect, modify,
  run and verify. The bug/gap must require reading the code; it must NOT be solvable from the instruction alone.
- `instruction`: what a developer would type to their coding agent (1-5 sentences). Describe symptoms/requirements,
  never the exact fix.
- `files`: the starting repository (path -> content). May include a few visible tests under tests/.
- `hidden`: 2-4 pytest files under tests/ named test_hidden_*.py that fully check the requirement including edge cases.
  Tests must be deterministic, run in < 5 s, need no network, and only use the standard library + pytest.
- `solution`: path -> FULL new content for every file the reference fix changes or creates.
- The starting repo MUST fail at least one hidden test; applying `solution` MUST make all tests pass.
- Do not reuse these well-known tasks: token-bucket rate limiter, dict-changed-size cache cleanup, saga compensation,
  CSV amount stats CLI, todo --json flag, duplicate e-mail finder, safe_join path traversal, inventory reserve/commit,
  calc->compute_total rename, asyncio fetch_all with semaphore, orders status migration, deep_merge, log scanner CLI.

Return ONLY a JSON object with keys: id, category, instruction, files, hidden, solution."""


def chat(messages, tools=None, max_tokens=16384, temperature=0.7, effort="medium", timeout=1800):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"reasoning_effort": effort, "enable_thinking": True}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]


def append(path, rec):
    with LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_ids(path, key="id"):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(l).get(key) for l in f if l.strip()}


def _eval_grams():
    from decontam import _ngrams, _words
    g = set()
    for t in EVAL_TASKS.values():
        g |= _ngrams(_words(t["instruction"] + "\n" + "\n".join(t["files"].values())), 13)
    return g


# ---------------------------------------------------------------- synth_tasks
def _materialize(task, with_solution):
    d = tempfile.mkdtemp(prefix="syn_")
    files = dict(task["files"])
    if with_solution:
        files.update(task["solution"])
    files.update(task["hidden"])
    for rel, content in files.items():
        if rel.startswith("/") or ".." in rel:
            raise ValueError("bad path")
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8", newline="\n").write(content)
    return d


def synth_one(i, out, eval_grams, diff=None):
    from decontam import _ngrams, _words
    rng = random.Random(i)
    cat = rng.choice(list(CATEGORIES))
    prompt = SYNTH_PROMPT.format(cat=cat, cat_desc=CATEGORIES[cat], domain=rng.choice(DOMAINS),
                                 diff=diff or rng.choice(["medium", "hard", "hard", "very hard (multi-step, 2 interacting bugs)"]))
    try:
        msg = chat([{"role": "user", "content": prompt}], max_tokens=24000, temperature=0.9)
        m = re.search(r"\{.*\}", msg.get("content") or "", re.S)
        task = json.loads(m.group(0))
        for k in ("instruction", "files", "hidden", "solution"):
            assert isinstance(task[k], (str, dict)) and task[k]
        task["id"] = f"syn{i:05d}_" + re.sub(r"\W+", "_", str(task.get("id", "task")))[:40]
        task["category"] = cat
        text = task["instruction"] + "\n" + "\n".join(task["files"].values())
        if _ngrams(_words(text), 13) & eval_grams:
            return "contaminated"
        d0, d1 = _materialize(task, False), _materialize(task, True)
        try:
            p0, _ = grade(d0, timeout=60)
            p1, _ = grade(d1, timeout=60)
        finally:
            shutil.rmtree(d0, ignore_errors=True)
            shutil.rmtree(d1, ignore_errors=True)
        if p0 or not p1:
            return "invalid"
        append(out, task)
        return "ok"
    except Exception as e:
        return f"error:{type(e).__name__}"


# ---------------------------------------------------------------- solve_tasks
SYSTEMS = [
    harness.SYSTEM,
    "You are Threnody, an autonomous coding agent operating in the user's repository through tools. Investigate "
    "before you edit, find the root cause, make minimal changes, and always run the code or tests to prove the fix "
    "before you report back.",
    "You are an expert software engineer working in a terminal. The working directory is the project root. Use the "
    "tools to explore, edit and test. Do not stop until you have evidence (a passing run) that the task is done; then "
    "give a concise summary.",
]
VERIFY_RE = re.compile(r"\b(pytest|python3?|unittest)\b")


def verified_before_finish(messages):
    """True if a bash call running python/pytest happened after the last file-modifying tool call."""
    last_edit, last_run = -1, -1
    for i, m in enumerate(messages):
        for c in m.get("tool_calls") or []:
            name = c["function"]["name"]
            args = c["function"].get("arguments") or "{}"
            if name in ("write_file", "edit_file"):
                last_edit = i
            elif name == "bash":
                try:
                    cmd = json.loads(args).get("command", "")
                except Exception:
                    cmd = ""
                if VERIFY_RE.search(cmd):
                    last_run = i
                if re.search(r"(>|sed -i|tee |cat <<)", cmd):
                    last_edit = max(last_edit, i)
    return last_run > last_edit >= 0 or (last_edit < 0 and last_run >= 0)


class _A:
    pass


def solve_one(task, attempt, out):
    a = _A()
    a.base_url, a.think, a.max_turns, a.max_tokens = URL, True, 40, 12000
    a.out = tempfile.mkdtemp(prefix="trajout_")
    sys_prompt = SYSTEMS[(hash(task["id"]) + attempt) % len(SYSTEMS)]
    teacher_chat = lambda base_url, messages, think, max_tokens, tools=harness.TOOLS: {"message": chat(  # noqa: E731
        messages, tools=tools, max_tokens=max_tokens, temperature=0.8)}
    try:
        res = harness.run_task(a, task, system=sys_prompt, chat_fn=teacher_chat)
        traj = json.load(open(os.path.join(a.out, f"traj_{task['id']}.json")))
    finally:
        shutil.rmtree(a.out, ignore_errors=True)
    ok = res["passed"] and verified_before_finish(traj)
    if ok:
        append(out, {"id": f"{task['id']}#{attempt}", "task": task["id"], "category": task["category"],
                     "turns": res["turns"], "messages": traj, "tools": harness.TOOLS})
    return res["passed"], ok


# ---------------------------------------------------------------- code (code_contests, verified)
def run_io_tests(code, tests, timeout=6):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sol.py")
        open(p, "w").write(code)
        for inp, exp in tests:
            try:
                r = subprocess.run([sys.executable, p], input=inp, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
            if r.returncode != 0 or r.stdout.split() != exp.split():
                return False
    return True


def code_one(row, out):
    prompt = (row["description"].strip() + "\n\nWrite a complete Python 3 program that reads from standard input and "
              "writes to standard output. Put the final program in a single ```python code block.")
    tests = list(zip(row["public_tests"]["input"], row["public_tests"]["output"])) + \
        list(zip(row["private_tests"]["input"], row["private_tests"]["output"]))
    tests = tests[:25]
    if not tests:
        return False
    msg = chat([{"role": "user", "content": prompt}], max_tokens=24000, temperature=0.7, effort="xhigh")
    m = re.findall(r"```python\n(.*?)```", msg.get("content") or "", re.S)
    if not m or not run_io_tests(m[-1], tests):
        return False
    append(out, {"id": row["name"], "messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": msg["content"], "reasoning_content": msg.get("reasoning_content") or ""}],
        "tools": None})
    return True


# ---------------------------------------------------------------- regen
def regen_prompts(n):
    from datasets import load_dataset
    rows = []
    sec = load_dataset("Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset", split="train", streaming=True)
    for r in sec.shuffle(seed=1, buffer_size=2000).take(n // 3):
        rows.append({"kind": "security", "messages": [{"role": "system", "content": r["system"]},
                                                      {"role": "user", "content": r["user"]}], "tools": None})
    w2c = load_dataset("nvidia/When2Call", "train_sft", split="train", streaming=True)
    for r in w2c.shuffle(seed=1, buffer_size=2000).take(n // 3):
        user = [m for m in r["messages"] if m["role"] == "user"][:1]
        if user:
            from build_data import _tool
            rows.append({"kind": "tool_judgment", "messages": user, "tools": [_tool(t) for t in r["tools"]]})
    ifr = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split="multi_turn_reasoning_if_think", streaming=True)
    for r in ifr.shuffle(seed=1, buffer_size=2000).take(n - len(rows)):
        first = [m for m in r["messages"] if m["role"] in ("system", "user")][:2]
        if first and first[-1]["role"] == "user":
            rows.append({"kind": "instruction_following", "messages": first, "tools": None})
    return rows


def regen_one(i, row, out):
    msg = chat(row["messages"], tools=row["tools"], max_tokens=8192, temperature=0.7,
               effort="low" if row["kind"] == "tool_judgment" else "medium")
    am = {"role": "assistant", "content": msg.get("content") or "", "reasoning_content": msg.get("reasoning_content") or ""}
    if msg.get("tool_calls"):
        am["tool_calls"] = msg["tool_calls"]
    if not am["content"] and not am.get("tool_calls"):
        return False
    append(out, {"id": f"regen{i:05d}", "kind": row["kind"], "messages": row["messages"] + [am], "tools": row["tools"]})
    return True


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["synth_tasks", "solve_tasks", "code", "regen"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=3, help="attempts per task (solve_tasks)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stats = {}

    def tally(x):
        stats[x] = stats.get(x, 0) + 1
        if sum(stats.values()) % 10 == 0:
            print(stats, flush=True)

    if a.mode == "synth_tasks":
        out = os.path.join(a.out, "tasks_train.jsonl")
        start = len(done_ids(out))
        grams = _eval_grams()
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for f in cf.as_completed([ex.submit(synth_one, start * 3 + i, out, grams) for i in range(a.n)]):
                tally(f.result())

    elif a.mode == "solve_tasks":
        tasks = [json.loads(l) for l in open(os.path.join(a.out, "tasks_train.jsonl"), encoding="utf-8")]
        out = os.path.join(a.out, "agent_traj.jsonl")
        have = done_ids(out)
        jobs = [(t, k) for t in tasks for k in range(a.k) if f"{t['id']}#{k}" not in have]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for f in cf.as_completed([ex.submit(solve_one, t, k, out) for t, k in jobs]):
                try:
                    passed, kept = f.result()
                    tally("kept" if kept else ("passed_unverified" if passed else "failed"))
                except Exception as e:
                    tally(f"error:{type(e).__name__}")

    elif a.mode == "code":
        from datasets import load_dataset
        out = os.path.join(a.out, "code_verified.jsonl")
        have = done_ids(out)
        # code_contests rows carry every reference solution; a large shuffle buffer of full rows hits a pyarrow
        # "offset overflow". Keep only the columns we use and a small buffer.
        ds = load_dataset("deepmind/code_contests", split="train", streaming=True).select_columns(
            ["name", "description", "public_tests", "private_tests", "cf_rating"]).shuffle(seed=3, buffer_size=200)
        rows = [r for r in ds.take(a.n) if r["name"] not in have and (r.get("cf_rating") or 0) >= 1400]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for f in cf.as_completed([ex.submit(code_one, r, out) for r in rows]):
                try:
                    tally("kept" if f.result() else "failed")
                except Exception as e:
                    tally(f"error:{type(e).__name__}")

    elif a.mode == "regen":
        out = os.path.join(a.out, "regen.jsonl")
        n_have = len(done_ids(out))
        rows = regen_prompts(a.n)[n_have:]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for f in cf.as_completed([ex.submit(regen_one, n_have + i, r, out) for i, r in enumerate(rows)]):
                try:
                    tally("kept" if f.result() else "empty")
                except Exception as e:
                    tally(f"error:{type(e).__name__}")
    print("FINAL", stats, flush=True)


if __name__ == "__main__":
    main()
