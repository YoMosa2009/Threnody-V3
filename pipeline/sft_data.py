"""Tokenize records with assistant-only loss masks.

The Qwen3.5 template renders every assistant turn as
    <|im_start|>assistant\n ... <|im_end|>\n
We render the whole conversation once, find those character spans, and set
labels only on tokens inside them (the role header itself is excluded; the
closing <|im_end|> is included so the model learns to stop).
"""
import json

ASSIST_OPEN = "<|im_start|>assistant\n"
END = "<|im_end|>"


def assistant_spans(text):
    spans, i = [], 0
    while True:
        s = text.find(ASSIST_OPEN, i)
        if s < 0:
            return spans
        start = s + len(ASSIST_OPEN)
        e = text.find(END, start)
        if e < 0:
            spans.append((start, len(text)))
            return spans
        spans.append((start, e + len(END)))
        i = e + len(END)


def encode(tok, rec, max_len):
    text = tok.apply_chat_template(rec["messages"], tools=rec.get("tools") or None, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True,
              truncation=True, max_length=max_len)
    spans = assistant_spans(text)
    labels, j = [], 0
    for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        inside = j < len(spans) and spans[j][0] <= a and b <= spans[j][1] and b > a
        labels.append(tid if inside else -100)
    return {"input_ids": enc["input_ids"], "labels": labels}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]
