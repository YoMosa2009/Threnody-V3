"""Prove every AgentLoop task is (a) failing as shipped and (b) solvable by its reference solution."""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import TASKS  # noqa: E402


def materialize(task, overrides=None):
    d = tempfile.mkdtemp(prefix=f"al_{task['id']}_")
    files = dict(task["files"])
    files.update(overrides or {})
    files.update(task["hidden"])
    for rel, content in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    return d


def grade(d, timeout=90):
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"],
                           cwd=d, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return r.returncode == 0, r.stdout[-400:]


if __name__ == "__main__":
    bad = 0
    for tid, t in TASKS.items():
        d0 = materialize(t)
        p0, _ = grade(d0)
        d1 = materialize(t, t["solution"])
        p1, out1 = grade(d1)
        ok = (not p0) and p1
        bad += not ok
        print(f"{tid:18s} starts_failing={not p0} solvable={p1} {'OK' if ok else 'PROBLEM'}")
        if not p1:
            print(out1)
        shutil.rmtree(d0, ignore_errors=True)
        shutil.rmtree(d1, ignore_errors=True)
    sys.exit(1 if bad else 0)
