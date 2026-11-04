"""Build the V2 SFT mix from verified sources, by token share (see docs/02_V2_PLAN.md section 2).

  agent  ~40%  traj_v2.jsonl (multi-schema, teacher + on-policy student) + V1 verified agent_traj.jsonl
  ost    <=30% Open-SWE-Traces (Qwen3.8-27B; claudecode, minisweagent, openhands, sweagent).
               opencode is excluded: it is a held-out evaluation harness.
  term   ~10% Qwen3.8 terminal traces
  sci    ~8%  sci_v2.jsonl (verified biology/chemistry)
  misc   ~12% regen.jsonl (tool judgment, security, instruction following) + identity
Every record: decontaminated (OmniAgent-Bench, AgentLoop, hard_eval), no action repeated >= 3 times in a row,
<= max_tokens. ~30% of records without a reasoning trace are produced by stripping reasoning (nothink format).

    python build_v2.py --out /content/drive/MyDrive/Threnody/data_v2b --total_tokens 25e6
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eval", "agentloop"))
import build_data as bd  # noqa: E402
import schemas  # noqa: E402
from decontam import _ngrams, _words  # noqa: E402

SHARES = {"agent": 0.38, "ost": 0.27, "term": 0.08, "sci": 0.08, "code": 0.07, "misc": 0.12}
NOTHINK_SHARE = 0.30
PER_TASK = 3


def _convert(r):
    msgs = []
    for m in r["messages"]:
        msg = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "assistant":
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            if m.get("tool_calls"):
                msg["tool_calls"] = [bd._call(tc["function"]["name"], tc["function"]["arguments"]) for tc in m["tool_calls"]]
        msgs.append(msg)
    return {"messages": msgs, "tools": [bd._tool(t) for t in r["tools"]] if r.get("tools") else None}


def src_agent():
    """Verified multi-schema trajectories, best first: per task, recovered-from-failure runs (determination) come
    first, then the fewest turns (efficiency); at most PER_TASK per task so easy tasks don't dominate."""
    p = os.path.join(bd.TEACHER_DIR, "traj_v2.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    picked = []
    for rs in by_task.values():
        rs.sort(key=lambda r: (not r.get("recovered"), r.get("turns", 99)))
        seen_schema, chosen = set(), []
        for r in rs:  # prefer different harnesses for the same task
            if r["schema"].split("~")[0] not in seen_schema:
                chosen.append(r)
                seen_schema.add(r["schema"].split("~")[0])
        chosen += [r for r in rs if r not in chosen]
        picked += chosen[:PER_TASK]
    random.Random(7).shuffle(picked)
    for r in picked:
        yield _convert(r)
    yield from bd._teacher_file("agent_traj.jsonl")


def src_code():
    """Verified competitive programming (code_contests, all tests pass): algorithmic cleverness."""
    yield from bd._teacher_file("code_verified.jsonl")


def src_ost():
    gens = [bd._ost(h) for h in ("claudecode", "minisweagent", "openhands", "sweagent")]
    while gens:  # round-robin so no single foreign harness dominates
        for g in list(gens):
            try:
                yield next(g)
            except StopIteration:
                gens.remove(g)


def src_term():
    yield from bd.src_qwen38_terminus()


def src_sci():
    yield from bd._teacher_file("sci_v2.jsonl")


def src_misc():
    yield from bd._teacher_file("regen.jsonl")


def holdout_grams(teacher_dir):
    p = os.path.join(teacher_dir, "hard_eval.jsonl")
    g = set()
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            t = json.loads(l)
            g |= _ngrams(_words(t["instruction"] + "\n" + "\n".join(t["files"].values())), 13)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--total_tokens", type=float, default=25e6)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--max_scan", type=int, default=20000, help="max records read per source")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = bd.AutoTokenizer.from_pretrained(bd.BASE)
    decon = bd.Decontaminator(os.path.join(HERE, "omniagent_tasks.json"))
    import gen_teacher
    grams = gen_teacher._eval_grams() | holdout_grams(bd.TEACHER_DIR)
    rng = random.Random(42)

    # The agent slice sets the scale: if verified trajectories are scarce, the whole mix shrinks so that
    # foreign-harness data can never exceed its share.
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
            if schemas.max_repeat(rec["messages"]) >= 3:
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
        print(name, stats[name], flush=True)

    total = min(a.total_tokens, stats["agent"]["available_tokens"] / SHARES["agent"])
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

    # nothink format: strip reasoning from enough records to reach ~30% of records without it
    has_reason = [r for r in picked if any(m.get("reasoning_content") for m in r["messages"])]
    need = int(NOTHINK_SHARE * len(picked)) - (len(picked) - len(has_reason))
    for r in rng.sample(has_reason, max(0, min(need, len(has_reason)))):
        for m in r["messages"]:
            m.pop("reasoning_content", None)
    picked += bd.identity_records(60)

    rng.shuffle(picked)
    n_val = max(20, min(150, len(picked) // 50))
    for fname, part in (("val.jsonl", picked[:n_val]), ("train.jsonl", picked[n_val:])):
        with open(os.path.join(a.out, fname), "w", encoding="utf-8") as f:
            for r in part:
                r.pop("n_tok", None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["_total"] = {"records": len(picked), "target_tokens": total,
                       "tokens": sum(s.get("picked_tokens", 0) for s in stats.values() if isinstance(s, dict)),
                       "nothink_records": sum(1 for r in picked if not any(m.get("reasoning_content") for m in r["messages"]))}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=2)
    print("TOTAL", stats["_total"], flush=True)
    if stats["agent"]["picked"] < 300:
        print("!!! fewer than 300 verified agent trajectories; mix too small", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
