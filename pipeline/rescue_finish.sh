#!/usr/bin/env bash
# Last resort under a hard credit limit (user rule: <= 1 h): stop failing evals, test softer merges s0.5/s0.3 on the
# fast comparison (AgentLoop + held-out harness + MMLU-Pro bio/chem 400, same build as base), pick the best candidate,
# export Q4_K_M + Q8_0 (+ mmproj), upload to Hugging Face, write eval_v2/final.json. Never blocks on optional parts.
set -u
DRIVE=${DRIVE:-/content/drive/MyDrive/Threnody}
L=$DRIVE/logs; E=$DRIVE/eval_v2; RUN=$DRIVE/sft_v3; DATA=$DRIVE/data_v2b
export DRIVE TEACHER_DIR=$DRIVE/teacher LLAMA=/content/llama.cpp BENCH=/content/OmniAgent-Bench
cd /content/Threnody; P=pipeline
pkill -f "bash pipeline/[r]un_v2_lean"; pkill -f "[g]ates.py eval"; pkill -f "[h]arness.py"; pkill -f "[r]un_bench.py"; pkill -f "[l]lama-server"; sleep 5
echo "=== $(date -u +%H:%M) rescue start"
for s in 0.5 0.3; do
  python $P/merge.py --adapter "$RUN/final_adapter" --out /content/merged_s$s --strength $s >> "$L/v2_rescue.log" 2>&1 \
    && python $P/gates.py pin --dir /content/merged_s$s
done
for s in 0.5 0.3; do python $P/gates.py build --src /content/merged_s$s --name s$s --quant Q8_0 --outdir /content/models 2>>"$L/v2_rescue.log" | tail -1 > /content/g_s$s.txt & done; wait
echo "=== $(date -u +%H:%M) rescue builds done"
python - "$E" <<'EOF' >> "$L/v2_rescue.log" 2>&1
import json, os, sys, threading
sys.path.insert(0, "pipeline"); import gates
e = sys.argv[1]
def run(name, port):
    gguf = open(f"/content/g_{name}.txt").read().strip()
    out = f"{e}/cand_{name}"; os.makedirs(out, exist_ok=True)
    srv = gates.Server(gguf, False, port, f"{out}/server_nothink.log")
    try:
        assert srv.ready(), "server failed"
        r = {"al": gates.agentloop(srv.url, f"{out}/al_agentloop_nothink", False, "agentloop"),
             "al_opencode": gates.agentloop(srv.url, f"{out}/al_opencode_nothink", False, "opencode"),
             "sci": gates.sci(srv.url, out, False, 400)}
        json.dump({"nothink": r}, open(f"{out}/results.json", "w"), indent=2)
        print(name, r, flush=True)
    finally:
        srv.stop()
ts = [threading.Thread(target=run, args=a) for a in (("s0.5", 8090), ("s0.3", 8091))]
[t.start() for t in ts]; [t.join() for t in ts]
EOF
echo "=== $(date -u +%H:%M) rescue evals done"
# choose: highest AgentLoop + held-out passes, ties broken by bio/chem; s1.0/s0.75 use whatever they finished
python - "$E" > /content/choice.txt <<'EOF'
import json, os, sys
e = sys.argv[1]
def load(n):
    p = f"{e}/cand_{n}/results.json"
    if os.path.exists(p):
        return json.load(open(p))["nothink"]
    r = {}
    for k, d in (("al", "agentloop"), ("al_opencode", "opencode")):
        f = f"{e}/cand_{n}/al_{d}_nothink/agentloop_summary.json"
        if os.path.exists(f):
            s = json.load(open(f)); r[k] = {"passed": s["passed"], "total": s["total"], "max_repeat": s["max_repeat"]}
    return r
base = json.load(open(f"{e}/base_mid/results.json"))["nothink"]
rows = []
for n in ("s1.0", "s0.75", "s0.5", "s0.3"):
    r = load(n)
    if "al" in r and "al_opencode" in r:
        rows.append((r["al"]["passed"] + r["al_opencode"]["passed"], r.get("sci", {}).get("overall", 0), n, r))
rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
_, _, best, cand = rows[0]
passed = cand["al"]["passed"] >= base["al"]["passed"] and cand["al_opencode"]["passed"] >= base["al_opencode"]["passed"] \
    and cand.get("sci", {}).get("overall", 0) >= base["sci"]["overall"] and max(cand["al"]["max_repeat"], cand["al_opencode"]["max_repeat"]) < 5
json.dump({"chosen": best, "passed": passed, "base": {"nothink": base}, "cand": {"nothink": cand},
           "all": {n: r for _, _, n, r in rows}}, open(f"{e}/final.json", "w"), indent=2)
print(best, int(passed))
EOF
read -r BEST PASSED < /content/choice.txt
echo "CHOSEN $BEST passed=$PASSED"
S=${BEST#s}
[ -d /content/merged_$BEST ] || python $P/merge.py --adapter "$RUN/final_adapter" --out /content/merged_$BEST --strength $S >> "$L/v2_export.log" 2>&1
python $P/gates.py pin --dir /content/merged_$BEST
mkdir -p "$DRIVE/gguf_v2"
[ -f /content/models/$BEST-Q8_0.gguf ] && cp /content/models/$BEST-Q8_0.gguf "$DRIVE/gguf_v2/Threnody-9B-Q8_0.gguf"
python $P/export_gguf.py --hf /content/merged_$BEST --llama /content/llama.cpp --calib "$DATA/val.jsonl" --out "$DRIVE/gguf_v2" \
  --name Threnody-9B --quants Q4_K_M --skip_kld --no_imatrix >> "$L/v2_export.log" 2>&1 || echo "!!! Q4_K_M export failed"
python /content/llama.cpp/convert_hf_to_gguf.py /content/merged_$BEST --mmproj --outfile "$DRIVE/gguf_v2/mmproj-Threnody-9B-F16.gguf" \
  --outtype f16 >> "$L/v2_export.log" 2>&1 || echo "mmproj export failed"
echo "=== $(date -u +%H:%M) export done"
python $P/publish_hf.py --gguf_dir "$DRIVE/gguf_v2" --final "$E/final.json" > "$L/v2_upload.log" 2>&1 && echo "uploaded" || echo "!!! HF upload failed"
echo "=== V2 FINISHED $(date -u) chosen=$BEST passed=$PASSED"
