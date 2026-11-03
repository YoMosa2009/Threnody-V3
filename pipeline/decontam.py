"""Decontamination against OmniAgent-Bench.

Drops any training sample whose text shares a long n-gram with a benchmark
prompt or test, or that mentions a benchmark entry-point function name in a
definition. Fable-Coder-V4 trained on near-copies of the benchmark; Threnody
must not.

Usage (library):  is_contaminated(text) -> bool
Usage (CLI):      python decontam.py tasks.json data.jsonl out.jsonl
"""
import json
import re
import sys

NGRAM = 13  # word 13-grams, the GPT-3/Llama-style decontamination window


def _words(s):
    return re.findall(r"[a-z0-9_]+", s.lower())


def _ngrams(words, n=NGRAM):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def load_bench(tasks_path):
    data = json.load(open(tasks_path, encoding="utf-8"))
    tasks = data if isinstance(data, list) else data.get("tasks", data)
    grams, entry_points, phrases = set(), set(), set()
    for t in tasks:
        meta = t.get("metadata", {})
        for field in (t.get("prompt", ""), meta.get("test_code", ""), meta.get("rubric", "")):
            grams |= _ngrams(_words(field))
        if meta.get("entry_point"):
            entry_points.add(meta["entry_point"])
    # Distinctive literals that should never appear in training targets.
    phrases |= {"/app/data/../etc/passwd", "user_comments.txt", "post /v1/checkout",
                "i am an enterprise subscriber and my production server is crashing"}
    return grams, entry_points, phrases


class Decontaminator:
    def __init__(self, tasks_path):
        self.grams, self.entry_points, self.phrases = load_bench(tasks_path)
        self.def_re = re.compile(r"def\s+(" + "|".join(map(re.escape, self.entry_points)) + r")\s*\(") \
            if self.entry_points else None
        self.dropped = 0

    def is_contaminated(self, text):
        low = text.lower()
        if any(p in low for p in self.phrases):
            return True
        if self.def_re and self.def_re.search(text):
            return True
        return bool(_ngrams(_words(text)) & self.grams)

    def filter(self, records, text_fn):
        for r in records:
            if self.is_contaminated(text_fn(r)):
                self.dropped += 1
                continue
            yield r


def record_text(r):
    return "\n".join(str(m.get("content", "")) for m in r.get("messages", []))


if __name__ == "__main__":
    tasks, src, dst = sys.argv[1:4]
    d = Decontaminator(tasks)
    kept = 0
    with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
        for r in d.filter((json.loads(l) for l in fi), record_text):
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept={kept} dropped={d.dropped}")
