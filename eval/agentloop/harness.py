"""AgentLoop harness: a minimal CLI-agent loop (like Claude Code / OpenCode, but tiny) driving any
OpenAI-compatible endpoint with native tool calling. The model gets a sandboxed copy of the task
repo, four tools, and up to --max_turns turns. When it stops calling tools, hidden tests grade it.

    python harness.py --base-url http://127.0.0.1:8090/v1 --name "Threnody-9B" --out results/ [--think]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schemas  # noqa: E402
from tasks import TASKS  # noqa: E402
from validate import grade  # noqa: E402

SYSTEM = ("You are a coding agent working in a repository on the user's machine. The current working directory is "
          "the repository root. Use the tools to inspect files, make changes and run commands (python and pytest are "
          "available). Verify your work by running code or tests. When the task is complete, reply with a short "
          "summary and no tool calls.")

TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command in the repository root. Returns exit code, stdout and stderr (truncated).",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file (relative path). Returns its content with line numbers.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file with the given content.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exactly one occurrence of old_str with new_str in a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}},
                    "required": ["path", "old_str", "new_str"]}}},
]

MAX_OUT = 6000


def clip(s):
    return s if len(s) <= MAX_OUT else s[:MAX_OUT // 2] + "\n...[truncated]...\n" + s[-MAX_OUT // 2:]


def inside(root, rel):
    p = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([os.path.realpath(root), p]) != os.path.realpath(root):
        raise PermissionError("path escapes the repository")
    return p


def run_tool(root, name, args):
    try:
        if name == "bash":
            r = subprocess.run(args["command"], shell=True, cwd=root, capture_output=True, text=True, timeout=60)
            return clip(f"exit={r.returncode}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}")
        if name == "read_file":
            lines = open(inside(root, args["path"]), encoding="utf-8").read().splitlines()
            return clip("\n".join(f"{i + 1:4d}| {l}" for i, l in enumerate(lines)) or "(empty file)")
        if name == "write_file":
            p = inside(root, args["path"])
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w", encoding="utf-8", newline="\n").write(args["content"])
            return f"wrote {args['path']} ({len(args['content'])} chars)"
        if name == "edit_file":
            p = inside(root, args["path"])
            text = open(p, encoding="utf-8").read()
            n = text.count(args["old_str"])
            if n != 1:
                return f"error: old_str found {n} times (must be exactly 1)"
            open(p, "w", encoding="utf-8", newline="\n").write(text.replace(args["old_str"], args["new_str"], 1))
            return f"edited {args['path']}"
        return f"error: unknown tool {name}"
    except subprocess.TimeoutExpired:
        return "error: command timed out after 60s"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def chat(base_url, messages, think, max_tokens, tools=TOOLS, temperature=None):
    body = {"model": "m", "messages": messages, "temperature": (0.0 if not think else 0.6) if temperature is None
            else temperature, "top_p": 0.95, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]


def run_task(a, task, system=None, chat_fn=None, schema=None):
    """Run one task in a schema (default: the original AgentLoop tools). chat_fn(base_url, messages, think,
    max_tokens, tools) -> {"message": ...}. Result includes `verified` (tests ran after the last edit) and
    `max_repeat` (longest run of identical consecutive actions)."""
    chat_fn = chat_fn or chat
    sc = schema or schemas.get("agentloop")
    root = tempfile.mkdtemp(prefix=f"agent_{task['id']}_")
    for rel, content in task["files"].items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8", newline="\n").write(content)
    instruction = task["instruction"]
    if sc["text_protocol"]:
        instruction += "\n\nRepository files:\n" + schemas.op_list(root)
    messages = [{"role": "system", "content": system or sc["system"]}, {"role": "user", "content": instruction}]
    t0, turns, tool_calls, error, ops = time.time(), 0, 0, None, []
    try:
        for turns in range(1, a.max_turns + 1):
            choice = chat_fn(a.base_url, messages, a.think, a.max_tokens, sc["tools"])
            msg = choice["message"]
            calls = msg.get("tool_calls") or []
            out = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("reasoning_content"):
                out["reasoning_content"] = msg["reasoning_content"]
            if calls and not sc["text_protocol"]:
                out["tool_calls"] = calls
            messages.append(out)
            if sc["text_protocol"]:
                step_ops, feedback = schemas.editblock_step(root, out["content"])
                if step_ops is None:
                    break
                ops += [o for o, _ in step_ops]
                tool_calls += len(step_ops)
                messages.append({"role": "user", "content": feedback})
                continue
            if not calls:
                break
            finished = False
            for c in calls:
                tool_calls += 1
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                    assert isinstance(args, dict)
                except Exception:
                    args = None
                if args is None:
                    op, res = "other", "error: arguments were not valid JSON"
                else:
                    try:
                        op, res = sc["dispatch"](root, c["function"]["name"], args)
                    except KeyError as e:
                        op, res = "other", f"error: missing argument {e}"
                    except Exception as e:
                        op, res = "other", f"error: {type(e).__name__}: {e}"
                ops.append(op)
                finished |= op == "finish"
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": res})
            if finished:
                break
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    # Hidden tests are copied in only now (overwriting anything the model wrote at those paths).
    for rel, content in task["hidden"].items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8", newline="\n").write(content)
    passed, tail = grade(root)
    res = {"id": task["id"], "category": task["category"], "passed": passed, "turns": turns,
           "tool_calls": tool_calls, "seconds": round(time.time() - t0, 1), "error": error, "grade_tail": tail,
           "schema": sc["name"], "verified": schemas.verified(ops), "max_repeat": schemas.max_repeat(messages)}
    json.dump(messages, open(os.path.join(a.out, f"traj_{task['id']}.json"), "w"), indent=1)
    shutil.rmtree(root, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max_turns", type=int, default=30)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--only", default="")
    ap.add_argument("--schema", default="agentloop", help="tool schema from schemas.py (e.g. opencode, codex)")
    ap.add_argument("--tasks_file", default="", help="JSONL of extra tasks to use instead of the built-in 14")
    ap.add_argument("--max_tasks", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = TASKS
    if a.tasks_file:
        tasks = {t["id"]: t for t in (json.loads(l) for l in open(a.tasks_file, encoding="utf-8") if l.strip())}
    sc = schemas.get(a.schema)
    ids = [i for i in tasks if not a.only or i in a.only.split(",")]
    if a.max_tasks:
        ids = ids[:a.max_tasks]
    results = []
    for tid in ids:
        r = run_task(a, tasks[tid], schema=sc)
        results.append(r)
        print(f"{tid:18s} {'PASS' if r['passed'] else 'FAIL'}  turns={r['turns']:2d} tools={r['tool_calls']:2d} "
              f"rep={r['max_repeat']:2d} {r['seconds']:6.1f}s {r['error'] or ''}", flush=True)
    n = sum(r["passed"] for r in results)
    summary = {"model": a.name, "think": a.think, "schema": a.schema, "passed": n, "total": len(results),
               "pass_rate": round(100 * n / len(results), 1),
               "max_repeat": max((r["max_repeat"] for r in results), default=0), "results": results}
    json.dump(summary, open(os.path.join(a.out, "agentloop_summary.json"), "w"), indent=2)
    print(f"\n{a.name} [{'think' if a.think else 'nothink'}]: {n}/{len(results)} passed ({summary['pass_rate']}%)")


if __name__ == "__main__":
    main()
