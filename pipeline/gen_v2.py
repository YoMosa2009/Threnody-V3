"""V2 data generation: verified multi-harness trajectories (teacher + on-policy student), a held-out hard eval
split, and verified biology/chemistry Q&A. All modes are resumable JSONL writers under --out.

Endpoints (OpenAI-compatible, vLLM):  TEACHER_URL (Qwen3.8-27B-FP8, served as "teacher")
                                      STUDENT_URL (Qwen3.5-9B base, served as "student")
Modes:
  synth      more repo tasks (gen_teacher.synth_one) -> tasks_v2.jsonl
  hard       very-hard tasks reserved for evaluation only -> hard_eval.jsonl (never sampled for training)
  sample     solve tasks in the TRAIN schemas (some with renamed tools), by teacher and student, think and nothink;
             keep only: hidden tests pass AND tests ran after the last edit AND no action repeated >= 3 times
             -> traj_v2.jsonl
  sci        biology/chemistry multiple-choice questions written by the teacher, solved independently by the
             teacher; kept only when the answers agree and the hazard filter passes -> sci_v2.jsonl

    python gen_v2.py --mode sample --out /content/drive/MyDrive/Threnody/teacher --k 3 --workers 48
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval", "agentloop"))
sys.path.insert(0, HERE)
import gen_teacher  # noqa: E402
import harness  # noqa: E402
import schemas  # noqa: E402

TEACHER_URL = os.environ.get("TEACHER_URL", "http://127.0.0.1:8000/v1")
TEACHER_EFFORT = os.environ.get("TEACHER_EFFORT", "medium")
STUDENT_URL = os.environ.get("STUDENT_URL", "http://127.0.0.1:8001/v1")
LOCK = threading.Lock()
MAX_REPEAT = 3
NOTHINK_SHARE = 0.3
RENAMED_SHARE = 0.2


def append(path, rec):
    with LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}


def chat(url, model, messages, think, max_tokens, tools=None, temperature=0.7, timeout=1800):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": think}}
    if model == "teacher":  # Qwen3.8 defaults to xhigh effort: far too slow for bulk sampling
        body["chat_template_kwargs"]["reasoning_effort"] = TEACHER_EFFORT
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]


# ---------------------------------------------------------------- hazard filter (bio/chem and trajectories)
HAZARD_RE = re.compile(
    r"\b(nerve agent|sarin|soman|tabun|novichok|\bvx\b|mustard gas|ricin|abrin|botulinum|anthrax|bacillus anthracis|"
    r"smallpox|variola|ebola|marburg|yersinia pestis|plague bacter|gain[- ]of[- ]function|enhanced transmissib|"
    r"aerosoli[sz]|weaponi[sz]|bioweapon|chemical weapon|select agent|toxin production|pathogen enhancement|"
    r"fentanyl synth|carfentanil|methamphetamine synth|meth synth|mdma synth|heroin synth|lsd synth|"
    r"explosive synth|tatp|hmtd|petn|rdx|nitroglycerin synth|pipe bomb|detonat)", re.I)


def hazardous(text):
    return bool(HAZARD_RE.search(text or ""))


# ---------------------------------------------------------------- tasks
def load_tasks(out, names=("tasks_train.jsonl", "tasks_v2.jsonl")):
    tasks = []
    for n in names:
        p = os.path.join(out, n)
        if os.path.exists(p):
            tasks += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    hard = os.path.join(out, "hard_eval.jsonl")
    held = {json.loads(l)["id"] for l in open(hard, encoding="utf-8")} if os.path.exists(hard) else set()
    return [t for t in tasks if t["id"] not in held]


# ---------------------------------------------------------------- sample (rejection sampling across schemas)
class _A:
    pass


def plan_attempts(task, k, k_start=0):
    """k attempts per (task, solver): rotate schemas, ~30% nothink, ~20% renamed tools. Deterministic per task."""
    rng = random.Random(task["id"])
    out = []
    for by in ("teacher", "student"):
        names = schemas.TRAIN_SCHEMAS[:]
        rng.shuffle(names)
        for i in range(k_start, k):
            out.append({"by": by, "schema": names[i % len(names)], "think": rng.random() >= NOTHINK_SHARE,
                        "renamed": rng.random() < RENAMED_SHARE, "seed": rng.randrange(1 << 30),
                        "id": f"{task['id']}#{by}{i}"})
    return out


FAIL_RE = re.compile(r"(\d+ failed|FAILED|Traceback \(most recent call last\)|AssertionError|exit=[1-9])")


def recovered(traj):
    """True if a test/python run failed at some point and the agent still reached a passing solution:
    the determination we want to teach (diagnose, change approach, keep going)."""
    runs = [m.get("content") or "" for m in traj if m.get("role") in ("tool", "user")][1:]
    return any(FAIL_RE.search(r) for r in runs)


def sample_one(task, att, out):
    sc = schemas.get(att["schema"])
    if att["renamed"]:
        sc = schemas.renamed(sc, att["seed"])
    url, model = (TEACHER_URL, "teacher") if att["by"] == "teacher" else (STUDENT_URL, "student")
    a = _A()
    a.base_url, a.think, a.max_turns = url, att["think"], 40
    a.max_tokens = 12000 if att["think"] else 4096
    a.out = tempfile.mkdtemp(prefix="v2out_")
    temp = 0.8 if att["by"] == "teacher" else 0.7

    def fn(base_url, messages, think, max_tokens, tools):
        return chat(url, model, messages, think, max_tokens, tools=tools, temperature=temp)
    try:
        res = harness.run_task(a, task, schema=sc, chat_fn=fn)
        traj = json.load(open(os.path.join(a.out, f"traj_{task['id']}.json")))
    finally:
        shutil.rmtree(a.out, ignore_errors=True)
    if not res["passed"]:
        return "failed"
    if res["error"]:
        return "error_in_loop"
    if not res["verified"]:
        return "passed_unverified"
    if res["max_repeat"] >= MAX_REPEAT:
        return "passed_repetitive"
    if hazardous(json.dumps(traj)):
        return "hazard"
    if not att["think"]:
        for m in traj:
            m.pop("reasoning_content", None)
    append(out, {"id": att["id"], "task": task["id"], "category": task.get("category"), "by": att["by"],
                 "schema": sc["name"], "think": att["think"], "turns": res["turns"], "recovered": recovered(traj),
                 "messages": traj, "tools": sc["tools"]})
    return f"kept_{att['by']}"


# ---------------------------------------------------------------- sci (biology + chemistry)
SCI_TOPICS = {
    "biology": ["molecular biology (DNA replication, transcription, translation)", "Mendelian and population genetics",
                "cell biology and signalling", "evolution and phylogenetics", "ecology and population dynamics",
                "human physiology", "plant biology and photosynthesis", "biochemistry and enzyme kinetics",
                "immunology fundamentals", "neurobiology", "developmental biology", "bioinformatics and sequence analysis"],
    "chemistry": ["stoichiometry and limiting reagents", "chemical equilibrium and Le Chatelier", "thermochemistry",
                  "reaction kinetics and rate laws", "electrochemistry and cell potentials", "acid-base and buffers",
                  "organic reaction mechanisms", "stereochemistry", "spectroscopy (NMR/IR/MS) interpretation",
                  "periodic trends and bonding", "solutions and colligative properties", "gas laws",
                  "coordination chemistry", "quantum chemistry basics"],
}
SCI_GEN = """Write ONE difficult, original, university-level {field} multiple-choice question about {topic}.
It must require multi-step reasoning or calculation, have exactly one correct answer, and 10 options labelled A-J
with plausible distractors. Safety: nothing about weapons, toxins, dangerous pathogens or illicit drug synthesis.
Return ONLY JSON: {{"question": "...", "options": ["...", ... 10 items], "answer": "<letter>"}}"""
LETTERS = "ABCDEFGHIJ"


def sci_prompt(q):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    return (f"{q['question']}\n\n{opts}\n\nThink step by step, then finish with \"The answer is (X)\" where X is the "
            "letter of the correct option.")


ANS_RE = re.compile(r"answer is \(?([A-J])\)?", re.I)


def sci_one(i, out, eval_grams):
    from decontam import _ngrams, _words
    rng = random.Random(10_000 + i)
    field = rng.choice(list(SCI_TOPICS))
    topic = rng.choice(SCI_TOPICS[field])
    msg = chat(TEACHER_URL, "teacher", [{"role": "user", "content": SCI_GEN.format(field=field, topic=topic)}], True,
               12000, temperature=0.9)["message"]
    m = re.search(r"\{.*\}", msg.get("content") or "", re.S)
    q = json.loads(m.group(0))
    assert len(q["options"]) == 10 and q["answer"] in LETTERS
    text = q["question"] + " " + " ".join(q["options"])
    if hazardous(text):
        return "hazard"
    if _ngrams(_words(text), 13) & eval_grams:
        return "contaminated"
    think = rng.random() >= NOTHINK_SHARE
    user = sci_prompt(q)
    sol = chat(TEACHER_URL, "teacher", [{"role": "user", "content": user}], think, 12000 if think else 3000,
               temperature=0.6)["message"]
    got = ANS_RE.findall(sol.get("content") or "")
    if not got or got[-1].upper() != q["answer"]:
        return "disagree"
    am = {"role": "assistant", "content": sol["content"]}
    reasoning = sol.get("reasoning_content") or sol.get("reasoning")
    if think and reasoning:
        am["reasoning_content"] = reasoning
    append(out, {"id": f"sci{i:05d}", "field": field, "topic": topic, "think": think,
                 "messages": [{"role": "user", "content": user}, am], "tools": None})
    return f"kept_{field}"


def sci_eval_grams():
    """13-grams of the MMLU-Pro biology/chemistry test questions (the science gate)."""
    from datasets import load_dataset
    from decontam import _ngrams, _words
    g = set()
    for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test"):
        if r["category"] in ("biology", "chemistry"):
            g |= _ngrams(_words(r["question"] + " " + " ".join(r["options"])), 13)
    return g


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["synth", "hard", "sample", "sci"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=3, help="attempts per task per solver (sample)")
    ap.add_argument("--k_start", type=int, default=0, help="skip the first attempts (extra samplers)")
    ap.add_argument("--solvers", default="teacher,student")
    ap.add_argument("--max_tasks", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stats = {}

    def tally(x):
        with LOCK:
            stats[x] = stats.get(x, 0) + 1
            if sum(stats.values()) % 20 == 0:
                print(stats, flush=True)

    def run(jobs):
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for f in cf.as_completed([ex.submit(*j) for j in jobs]):
                try:
                    tally(f.result())
                except Exception as e:
                    tally(f"error:{type(e).__name__}")

    if a.mode in ("synth", "hard"):
        out = os.path.join(a.out, "tasks_v2.jsonl" if a.mode == "synth" else "hard_eval.jsonl")
        start = 500_000 if a.mode == "hard" else 100_000
        start += len(done_ids(out)) * 3
        diff = "very hard (multi-step, 2 interacting bugs across modules)" if a.mode == "hard" else None
        grams = gen_teacher._eval_grams()
        run([(gen_teacher.synth_one, start + i, out, grams, diff) for i in range(a.n)])
    elif a.mode == "sample":
        out = os.path.join(a.out, "traj_v2.jsonl")
        have = done_ids(out)
        solvers = set(a.solvers.split(","))
        tasks = load_tasks(a.out)
        if a.max_tasks:
            tasks = tasks[:a.max_tasks]
        print(f"tasks={len(tasks)}", flush=True)
        jobs = [(sample_one, t, att, out) for t in tasks for att in plan_attempts(t, a.k, a.k_start)
                if att["by"] in solvers and att["id"] not in have]
        random.Random(0).shuffle(jobs)
        run(jobs)
    elif a.mode == "sci":
        out = os.path.join(a.out, "sci_v2.jsonl")
        n0 = len(done_ids(out))
        grams = sci_eval_grams()
        run([(sci_one, n0 * 2 + i, out, grams) for i in range(a.n)])
    print("FINAL", stats, flush=True)


if __name__ == "__main__":
    main()
