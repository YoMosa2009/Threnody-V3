"""Build the V3 SFT dataset mix from verified sources.
Implements the 6 Pillars of Threnody V3 (docs/03_V3_PLAN.md):
  1. Complete reasoning restoration (<think> / reasoning_content / reasoning).
  2. Zero-tolerance repetition filter: drop any record with max_repeat >= 2.
  3. Sort clean 0-repeat trajectories first (no recovery loop trap).
  4. Multi-curriculum balance:
       agent  ~35%  verified multi-schema trajectories (traj_v2, traj_v3, agent_traj)
       code   ~20%  verified algorithmic replay (code_verified.jsonl, CodeContests)
       ost    ~20%  Open-SWE-Traces (claudecode, minisweagent, openhands, sweagent)
       sci    ~15%  verified BioChem mechanism reasoning (sci_v2.jsonl)
       misc   ~10%  cybersecurity, tool judgment (regen.jsonl) + identity
  5. 13-gram decontamination against OmniAgent-Bench and AgentLoop.
  6. ~25% nothink share for fast CLI deterministic execution; 75% interleaved thinking.

Usage:
  python pipeline/build_v3.py --out /content/drive/MyDrive/Threnody/data_v3 --total_tokens 35e6
"""
import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eval", "agentloop"))
import build_data as bd  # noqa: E402
import schemas  # noqa: E402
from decontam import _ngrams, _words  # noqa: E402

SHARES = {"agent": 0.35, "code": 0.20, "ost": 0.20, "sci": 0.15, "misc": 0.10}
NOTHINK_SHARE = 0.25
PER_TASK = 3
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_reasoning(m):
    """Robust reasoning extraction across vLLM, transformers, and OpenAI format."""
    rc = m.get("reasoning_content") or m.get("reasoning") or ""
    if not rc and "<think>" in (m.get("content") or ""):
        match = THINK_RE.search(m["content"])
        if match:
            rc = match.group(1).strip()
    return rc


def clean_content(m, rc):
    """Strip explicit <think> tags from content if reasoning_content is extracted."""
    content = m.get("content") or ""
    if rc and "<think>" in content:
        content = THINK_RE.sub("", content).strip()
    return content


def _convert(r):
    msgs = []
    for m in r["messages"]:
        msg = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "assistant":
            rc = extract_reasoning(m)
            if rc:
                msg["reasoning_content"] = rc
                msg["content"] = clean_content(m, rc)
            if m.get("tool_calls"):
                msg["tool_calls"] = [bd._call(tc["function"]["name"], tc["function"]["arguments"])
                                     for tc in m["tool_calls"]]
        msgs.append(msg)
    return {"messages": msgs, "tools": [bd._tool(t) for t in r["tools"]] if r.get("tools") else None}


def src_agent():
    """Verified multi-schema trajectories.
    Clean 0-repeat trajectories come first; at most PER_TASK per task."""
    p_v3 = os.path.join(bd.TEACHER_DIR, "traj_v3.jsonl")
    p_v2 = os.path.join(bd.TEACHER_DIR, "traj_v2.jsonl")
    rows = []
    for p in (p_v3, p_v2):
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    by_task = {}
    for r in rows:
        by_task.setdefault(r.get("task", r.get("id", "unknown")), []).append(r)
    picked = []
    for rs in by_task.values():
        # Pillar 2: prioritize 0 repeats, then shortest turn count
        rs.sort(key=lambda r: (r.get("max_repeat", 0) > 0, r.get("turns", 99)))
        seen_schema, chosen = set(), []
        for r in rs:
            sc_key = r.get("schema", "default").split("~")[0]
            if sc_key not in seen_schema:
                chosen.append(r)
                seen_schema.add(sc_key)
        chosen += [r for r in rs if r not in chosen]
        picked += chosen[:PER_TASK]
    random.Random(42).shuffle(picked)
    for r in picked:
        yield _convert(r)
    yield from bd._teacher_file("agent_traj.jsonl")


def src_code():
    """Verified competitive programming (CodeContests, all tests pass)."""
    yield from bd._teacher_file("code_verified.jsonl")
    # Also stream from code reasoning if available
    yield from bd.src_code_reasoning()


def src_ost():
    """Open-SWE-Traces across diverse tool schemas (excluding held-out opencode)."""
    gens = [bd._ost(h) for h in ("claudecode", "minisweagent", "openhands", "sweagent")]
    while gens:
        for g in list(gens):
            try:
                yield next(g)
            except StopIteration:
                gens.remove(g)


def src_sci():
    """Verified BioChem reasoning with mechanism scratchpads."""
    yield from bd._teacher_file("sci_v2.jsonl")


def src_misc():
    """Cybersecurity, tool judgment, and instruction following."""
    yield from bd._teacher_file("regen.jsonl")
    yield from bd.src_when2call()


def holdout_grams(teacher_dir):
    p = os.path.join(teacher_dir, "hard_eval.jsonl")
    g = set()
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            t = json.loads(l)
            g |= _ngrams(_words(t.get("instruction", "") + "\n" + "\n".join(t.get("files", {}).values())), 13)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--total_tokens", type=float, default=35e6)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--max_scan", type=int, default=30000, help="max records read per source")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok = bd.AutoTokenizer.from_pretrained(bd.BASE)
    decon = bd.Decontaminator(os.path.join(HERE, "omniagent_tasks.json"))
    import gen_teacher
    grams = gen_teacher._eval_grams() | holdout_grams(bd.TEACHER_DIR)
    rng = random.Random(42)

    pools, stats = {}, {}
    for name in SHARES:
        recs, seen, drop = [], 0, {"contaminated": 0, "repetitive": 0, "too_long": 0, "render_error": 0}
        for rec in globals()["src_" + name]():
            seen += 1
            if seen > a.max_scan:
                break
            text = bd.record_text(rec)
            if decon.is_contaminated(text) or (_ngrams(_words(text), 13) & grams):
                drop["contaminated"] += 1
                continue
            # Pillar 2: Zero-tolerance repetition check (strict max_repeat < 2)
            if schemas.max_repeat(rec["messages"]) >= 2:
                drop["repetitive"] += 1
                continue
            try:
                n = bd.render_len(tok, rec)
            except Exception:
                drop["render_error"] += 1
                continue
            if n > a.max_tokens:
                drop["too_long"] += 1
                continue
            rec["source"], rec["n_tok"] = name, n
            recs.append(rec)
        rng.shuffle(recs)
        pools[name] = recs
        stats[name] = dict(available=len(recs), available_tokens=sum(r["n_tok"] for r in recs), seen=seen, **drop)
        print(f"[{name}] {stats[name]}", flush=True)

    # Scale total tokens by available agent data
    avail_agent_tokens = stats["agent"]["available_tokens"]
    if avail_agent_tokens > 0:
        total = min(a.total_tokens, avail_agent_tokens / SHARES["agent"])
    else:
        total = a.total_tokens

    picked = []
    for name, share in SHARES.items():
        budget, used = share * total, 0
        for r in pools[name]:
            if used >= budget:
                break
            picked.append(r)
            used += r["n_tok"]
        stats[name]["picked_tokens"] = used
        stats[name]["picked"] = sum(1 for r in picked if r["source"] == name)

    # Dual-mode formatting: retain reasoning in ~75%, strip in ~25% for fast nothink CLI
    has_reason = [r for r in picked if any(m.get("reasoning_content") for m in r["messages"])]
    target_nothink = int(NOTHINK_SHARE * len(picked))
    already_nothink = len(picked) - len(has_reason)
    need_strip = max(0, target_nothink - already_nothink)

    if need_strip > 0 and has_reason:
        for r in rng.sample(has_reason, min(need_strip, len(has_reason))):
            for m in r["messages"]:
                m.pop("reasoning_content", None)

    # Add identity records
    picked += bd.identity_records(60)
    rng.shuffle(picked)

    n_val = max(25, min(200, len(picked) // 40))
    for fname, part in (("val.jsonl", picked[:n_val]), ("train.jsonl", picked[n_val:])):
        with open(os.path.join(a.out, fname), "w", encoding="utf-8") as f:
            for r in part:
                r.pop("n_tok", None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    reasoning_count = sum(1 for r in picked if any(m.get("reasoning_content") for m in r["messages"]))
    stats["_total"] = {
        "records": len(picked),
        "target_tokens": total,
        "tokens": sum(s.get("picked_tokens", 0) for s in stats.values() if isinstance(s, dict)),
        "reasoning_records": reasoning_count,
        "nothink_records": len(picked) - reasoning_count,
        "reasoning_ratio": round(reasoning_count / max(1, len(picked)), 3)
    }
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=2)
    print("=== V3 DATASET SUMMARY ===")
    print(json.dumps(stats["_total"], indent=2), flush=True)


if __name__ == "__main__":
    main()
