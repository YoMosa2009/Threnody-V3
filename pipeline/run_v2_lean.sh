#!/usr/bin/env bash
# Threnody V2, budget-constrained finish (~2.2 GPU-hours). Picks up after generation (traj_v2/sci_v2 in Drive).
#   data mix || base reference  ->  SFT  ->  candidates s1.0 / s0.75 (parallel)  ->  gate vs same-build base
#   -> [rescue: s0.5]  ->  export Q4_K_M/Q5_K_M/Q8_0 + mmproj  ->  Hugging Face upload  ->  final.json
# Every comparison: Q8_0, pinned chat template, nothink, same harnesses. Markers: markers/v2l_<stage>.done
set -u
DRIVE=${DRIVE:-/content/drive/MyDrive/Threnody}
T=$DRIVE/teacher; M=$DRIVE/markers; L=$DRIVE/logs; E=$DRIVE/eval_v2; RUN=$DRIVE/sft_v3; DATA=$DRIVE/data_v2b
export DRIVE TEACHER_DIR=$T LLAMA=/content/llama.cpp BENCH=/content/OmniAgent-Bench HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$E" /content/models; cd /content/Threnody; P=pipeline
stage() { [ -f "$M/v2l_$1.done" ] && { echo "skip $1"; return 1; }; echo "=== $(date -u +%H:%M) start $1"; return 0; }
mark() { touch "$M/v2l_$1.done"; echo "=== $(date -u +%H:%M) done $1"; }
fail() { echo "!!! STAGE $1 FAILED $(date -u +%H:%M) ($2)"; exit 1; }
BASE_HF=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-9B'))")

# ---- generation is over: free the GPU
pkill -f "[g]en_v2.py"; pkill -f "[v]llm serve"; sleep 15
# resume after a relaunch: let orphaned workers from the previous copy finish instead of duplicating them
while pgrep -f "[b]uild_v2.py" >/dev/null || pgrep -f "[g]ates.py eval" >/dev/null; do sleep 20; done
[ -f "$DATA/stats.json" ] && [ -f "$DATA/train.jsonl" ] && [ ! -f "$M/v2l_data.done" ] && mark data

# ---- 1. data mix (CPU) || base reference (GPU)
stage data && { python $P/build_v2.py --out "$DATA" --total_tokens 25e6 --max_scan 6000 > "$L/v2_data.log" 2>&1 & DPID=$!; }
stage base && {
  G8=$(python $P/gates.py build --src "$BASE_HF" --name base --quant Q8_0 --outdir /content/models 2>>"$L/v2_base.log" | tail -1)
  python $P/gates.py eval --gguf "$G8" --suite mid --out "$E/base_mid" --tag base >> "$L/v2_base.log" 2>&1 \
    && mark base || fail base "see v2_base.log"; }
[ -n "${DPID:-}" ] && { wait $DPID && mark data || fail data "see v2_data.log"; }

# ---- 2. SFT (small, clean, mostly on-policy data)
stage sft && { python $P/train_sft.py --data "$DATA" --out "$RUN" --max_len 32768 --epochs 1 --lr 1e-4 --max_hours ${SFT_HOURS:-0.4} \
    --save_steps 10 > "$L/v2_sft.log" 2>&1 && mark sft || fail sft "see v2_sft.log"; }

# ---- 3. candidates, evaluated two at a time on separate ports
cand() {  # cand <name> <strength> <port>
  [ -f "$E/cand_$1/results.json" ] && [ -n "$(python -c "import json;print(json.load(open('$E/cand_$1/results.json')).get('nothink',{}).get('omni',''))")" ] && return 0
  python $P/merge.py --adapter "$RUN/final_adapter" --out /content/merged_$1 --strength $2 >> "$L/v2_cand.log" 2>&1 \
    && python $P/gates.py pin --dir /content/merged_$1 \
    && g=$(python $P/gates.py build --src /content/merged_$1 --name $1 --quant Q8_0 --outdir /content/models 2>>"$L/v2_cand.log" | tail -1) \
    && python $P/gates.py eval --gguf "$g" --suite mid --out "$E/cand_$1" --tag $1 --port $3 >> "$L/v2_cand_$1.log" 2>&1
}
judge() {  # sets BEST (passing first, else best fast score) and PASSED=1/0
  python - "$E" "$@" > /content/judge.txt <<'EOF'
import json, subprocess, sys
sys.path.insert(0, "pipeline"); import gates
e, names = sys.argv[1], sys.argv[2:]
base = json.load(open(f"{e}/base_mid/results.json"))
rows = []
for n in names:
    try:
        c = json.load(open(f"{e}/cand_{n}/results.json"))
    except FileNotFoundError:
        continue
    g = gates.gate(c, base, "mid"); json.dump(g, open(f"{e}/gate_{n}.json", "w"), indent=2)
    rows.append((g["pass"], gates.fast_score(c), n))
    print("#", n, "PASS" if g["pass"] else "FAIL", [x["gate"] for x in g["checks"] if not x["pass"]])
rows.sort(reverse=True)
print(rows[0][2], int(rows[0][0]))
EOF
  cat /content/judge.txt; read -r BEST PASSED < <(tail -1 /content/judge.txt)
}
stage cands && {
  merge_one() { python $P/merge.py --adapter "$RUN/final_adapter" --out /content/merged_$1 --strength $2 >> "$L/v2_cand.log" 2>&1 \
    && python $P/gates.py pin --dir /content/merged_$1; }
  merge_one s1.0 1.0 && merge_one s0.75 0.75 || fail cands "merge failed"
  for n in s1.0 s0.75; do python $P/gates.py build --src /content/merged_$n --name $n --quant Q8_0 --outdir /content/models \
      2>>"$L/v2_cand.log" | tail -1 > /content/g_$n.txt & done; wait
  python $P/gates.py eval --gguf "$(cat /content/g_s1.0.txt)" --suite mid --out "$E/cand_s1.0" --tag s1.0 --port 8090 > "$L/v2_cand_s1.0.log" 2>&1 &
  python $P/gates.py eval --gguf "$(cat /content/g_s0.75.txt)" --suite mid --out "$E/cand_s0.75" --tag s0.75 --port 8091 > "$L/v2_cand_s0.75.log" 2>&1 &
  wait; mark cands; }
judge s1.0 s0.75
# ---- 4. last resort (user rule: <= 1 hour): a softer merge, then pick the best of all
if [ "$PASSED" != 1 ] && [ "${RESCUE:-0}" = 1 ] && stage rescue; then
  echo "GATE FAILED for s1.0 and s0.75 -> rescue s0.5"
  cand s0.5 0.5 8090; mark rescue
fi
judge s1.0 s0.75 s0.5
python - "$E" "$BEST" "$PASSED" <<'EOF'
import json, sys
e, best, passed = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
json.dump({"chosen": best, "passed": passed, "base": json.load(open(f"{e}/base_mid/results.json")),
           "cand": json.load(open(f"{e}/cand_{best}/results.json")), "gate": json.load(open(f"{e}/gate_{best}.json"))},
          open(f"{e}/final.json", "w"), indent=2)
EOF
echo "CHOSEN $BEST passed=$PASSED"

# ---- 5. export + upload
stage export && { S=${BEST#s}
  [ -d /content/merged_$BEST ] || python $P/merge.py --adapter "$RUN/final_adapter" --out /content/merged_$BEST --strength $S >> "$L/v2_export.log" 2>&1
  python $P/gates.py pin --dir /content/merged_$BEST
  python $P/export_gguf.py --hf /content/merged_$BEST --llama /content/llama.cpp --calib "$DATA/val.jsonl" --out "$DRIVE/gguf_v2" \
    --name Threnody-9B --quants Q4_K_M,Q5_K_M,Q8_0 --skip_kld --no_imatrix > "$L/v2_export.log" 2>&1 || fail export "see v2_export.log"
  python /content/llama.cpp/convert_hf_to_gguf.py /content/merged_$BEST --mmproj --outfile "$DRIVE/gguf_v2/mmproj-Threnody-9B-F16.gguf" \
    --outtype f16 >> "$L/v2_export.log" 2>&1 || echo "mmproj export failed (text model unaffected)" >> "$L/v2_export.log"
  mark export; }
stage upload && { python $P/publish_hf.py --gguf_dir "$DRIVE/gguf_v2" --final "$E/final.json" > "$L/v2_upload.log" 2>&1 \
  && mark upload || echo "!!! HF upload failed (files are in Drive gguf_v2); see v2_upload.log"; }
echo "=== V2 FINISHED $(date -u) chosen=$BEST passed=$PASSED"
