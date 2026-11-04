#!/usr/bin/env bash
# Threnody V3 unattended automated pipeline (Colab A100-80GB / G4-96GB).
# Implements the 6 Pillars of Threnody V3 (docs/03_V3_PLAN.md).
# Every stage writes markers/v3_<stage>.done on success and resumes automatically.
#
# Usage:
#   DRIVE=/content/drive/MyDrive/Threnody nohup bash pipeline/run_v3.sh >> $DRIVE/logs/run_v3.log 2>&1 &
# DRY=1 runs every stage on a mini test slice to verify the entire pipeline before spending GPU credits.
set -u
DRIVE=${DRIVE:-/content/drive/MyDrive/Threnody}
DRY=${DRY:-0}
T=$DRIVE/teacher; M=$DRIVE/markers; L=$DRIVE/logs; E=$DRIVE/eval_v3
RUN=$DRIVE/sft_v3; DATA=$DRIVE/data_v3
if [ "$DRY" = 1 ]; then
  T=$DRIVE/dry/teacher; M=$DRIVE/dry/markers; L=$DRIVE/dry/logs; E=$DRIVE/dry/eval_v3
  RUN=$DRIVE/dry/sft_v3; DATA=$DRIVE/dry/data_v3
fi
mkdir -p "$T" "$M" "$L" "$E" "$RUN" "$DATA" /content/models
export DRIVE GATES_DRY=$DRY TEACHER_DIR=$T LLAMA=/content/llama.cpp BENCH=/content/OmniAgent-Bench HF_HUB_ENABLE_HF_TRANSFER=0
cd /content/Threnody
P=pipeline

stage() { [ -f "$M/v3_$1.done" ] && { echo "skip $1 (already done)"; return 1; }; echo "=== $(date -u +%H:%M) start $1"; return 0; }
mark() { touch "$M/v3_$1.done"; echo "=== $(date -u +%H:%M) done $1"; }
fail() { echo "!!! STAGE $1 FAILED $(date -u +%H:%M) ($2)"; exit 1; }
n() { [ "$DRY" = 1 ] && echo "$2" || echo "$1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

echo "=========================================================="
echo " Starting Threnody V3 Automated Pipeline"
echo " Time: $(date -u)"
echo " Drive: $DRIVE | DRY: $DRY"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "=========================================================="

# ---- 0. Environment & Hardware Acceleration
pip uninstall -y -q torchaudio torchao 2>/dev/null || true
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)
CC=${CC:-80}
echo "Detected CUDA compute capability: sm_$CC"

if [ ! -f /content/llama.cpp/build/.arch_$CC ]; then
  CACHE=$DRIVE/llama_bin_sm$CC.tgz
  if [ -f "$CACHE" ]; then
    echo "Restoring cached llama.cpp build for sm_$CC from Drive..."
    mkdir -p /content/llama.cpp/build && tar xzf "$CACHE" -C /content/llama.cpp/build && touch /content/llama.cpp/build/.arch_$CC
  else
    echo "Building llama.cpp for sm_$CC..."
    (cd /content/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC -DLLAMA_CURL=OFF \
      -DBUILD_SHARED_LIBS=OFF > /dev/null && cmake --build build -j 12 --target llama-server llama-quantize \
      llama-imatrix llama-perplexity llama-cli > "$L/llama_build.log" 2>&1 && touch build/.arch_$CC \
      && tar czf "$CACHE" -C build bin) &
  fi
fi

# Precompiled DeltaNet kernel speedup
python -c "import causal_conv1d" 2>/dev/null || \
  (timeout 25m pip install -q --no-build-isolation causal-conv1d flash-attn > "$L/causal_conv1d.log" 2>&1 || \
   echo "causal_conv1d / flash-attn fallback active" >> "$L/causal_conv1d.log") &

# ---- 1. Teacher & Ingest Check
# If files were transferred or shared from mosaalsaegh@gmail.com, verify their presence
if [ -d "$DRIVE/teacher" ] && [ -f "$DRIVE/teacher/traj_v2.jsonl" ]; then
  echo "Found transferred/existing teacher files in Drive: $(lines $DRIVE/teacher/traj_v2.jsonl) trajectories."
  [ -f "$T/traj_v2.jsonl" ] || cp -r "$DRIVE/teacher"/* "$T/" 2>/dev/null || true
fi

wait # ensure llama.cpp and cuda extensions build finishes before training

# ---- 2. Build V3 Data Mix (Anti-loop filter, full reasoning restoration)
stage data && {
  python $P/build_v3.py --out "$DATA" --total_tokens $(n 35e6 3e5) --max_scan $(n 30000 300) > "$L/v3_data.log" 2>&1 \
    && mark data || fail data "see v3_data.log"
}

# ---- 3. Base Ground-Truth Reference Evaluation (Q8_0 on mid suite)
stage base && {
  BASE_HF=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-9B'))")
  G8=$(python $P/gates.py build --src "$BASE_HF" --name base --quant Q8_0 --outdir /content/models 2>>"$L/v3_base.log" | tail -1)
  python $P/gates.py eval --gguf "$G8" --suite $(n mid fast) --out "$E/base_mid" --tag base >> "$L/v3_base.log" 2>&1 \
    && mark base || fail base "see v3_base.log"
}

# ---- 4. Supervised Fine-Tuning (SFT) with Checkpoint Gate Watcher
stage sft && {
  rm -f /content/STOP_TRAINING
  python $P/ckpt_gate.py --run_dir "$RUN" --base_fast "$E/base_mid/results.json" --stop_file /content/STOP_TRAINING \
    --every $(n 25 3) > "$L/v3_gate.log" 2>&1 &
  GATE_PID=$!

  python $P/train_sft.py --data "$DATA" --out "$RUN" --max_len 32768 --epochs $(n 1.2 0.1) --lr 1e-4 \
    --max_hours $(n 3.5 0.2) --save_steps $(n 25 2) --stop_file /content/STOP_TRAINING \
    $( [ "$DRY" = 1 ] && echo "--max_steps 4 --grad_accum 2" ) \
    > "$L/v3_sft.log" 2>&1 || fail sft "see v3_sft.log"

  wait $GATE_PID 2>/dev/null || true
  mark sft
}

# ---- 5. Candidate Evaluation & Release Gates (Target s1.0 full strength)
stage select && {
  echo "Evaluating release candidates (targeting s1.0 full adapter strength)..."
  merge_cand() {
    cand_name=$1; strength=$2; port=$3
    m_dir="/content/merged_${cand_name}"
    [ -d "$m_dir" ] || python $P/merge.py --adapter "$RUN/final_adapter" --out "$m_dir" --strength "$strength" >> "$L/v3_select.log" 2>&1
    python $P/gates.py pin --dir "$m_dir" >> "$L/v3_select.log" 2>&1
    g_cand=$(python $P/gates.py build --src "$m_dir" --name "$cand_name" --quant Q8_0 --outdir /content/models 2>>"$L/v3_select.log" | tail -1)
    python $P/gates.py eval --gguf "$g_cand" --suite $(n mid fast) --out "$E/cand_${cand_name}" --tag "$cand_name" --port "$port" >> "$L/v3_cand_${cand_name}.log" 2>&1
    rm -rf "$g_cand"
  }

  merge_cand "s1.0" 1.0 8090 &
  C1_PID=$!
  merge_cand "s0.75" 0.75 8091 &
  C2_PID=$!
  wait $C1_PID; wait $C2_PID

  # Evaluate and judge against Base
  python - "$E" s1.0 s0.75 > /content/judge_v3.txt <<'EOF'
import json, sys
sys.path.insert(0, "pipeline")
import gates
eval_dir, cands = sys.argv[1], sys.argv[2:]
base = json.load(open(f"{eval_dir}/base_mid/results.json"))
results = []
for c in cands:
    p = f"{eval_dir}/cand_{c}/results.json"
    try:
        cand_data = json.load(open(p))
    except Exception:
        continue
    g = gates.gate(cand_data, base, "mid")
    json.dump(g, open(f"{eval_dir}/gate_{c}.json", "w"), indent=2)
    results.append((g["pass"], gates.fast_score(cand_data), c))
    print(f"Candidate {c}: PASS={g['pass']}, checks={[x['gate'] for x in g['checks'] if not x['pass']]}")
results.sort(reverse=True)
chosen = results[0][2]
passed = int(results[0][0])
print(f"CHOSEN_CANDIDATE: {chosen} (passed={passed})")
open(f"{eval_dir}/chosen.txt", "w").write(chosen)
EOF
  cat /content/judge_v3.txt
  CHOSEN=$(cat "$E/chosen.txt" 2>/dev/null || echo "s1.0")
  mark select
}

# ---- 6. Export Final GGUF Ladder + Vision mmproj
stage export && {
  CHOSEN=$(cat "$E/chosen.txt" 2>/dev/null || echo "s1.0")
  STRENGTH=${CHOSEN#s}
  M_DIR="/content/merged_${CHOSEN}"
  [ -d "$M_DIR" ] || python $P/merge.py --adapter "$RUN/final_adapter" --out "$M_DIR" --strength "$STRENGTH" >> "$L/v3_export.log" 2>&1
  python $P/gates.py pin --dir "$M_DIR"

  echo "Exporting imatrix quant ladder for $CHOSEN..."
  python $P/export_gguf.py --hf "$M_DIR" --llama /content/llama.cpp --calib "$DATA/val.jsonl" \
    --out "$DRIVE/gguf_v3" --name "Threnody-9B-V3" $( [ "$DRY" = 1 ] && echo "--quants Q4_K_M" ) \
    > "$L/v3_export.log" 2>&1 || fail export "see v3_export.log"

  python /content/llama.cpp/convert_hf_to_gguf.py "$M_DIR" --mmproj \
    --outfile "$DRIVE/gguf_v3/mmproj-Threnody-9B-V3-F16.gguf" --outtype f16 >> "$L/v3_export.log" 2>&1 || true

  mark export
}

# ---- 7. Hugging Face Upload (Optional)
stage upload && {
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "Uploading to Hugging Face Hub (MalxTech/Threnody-9B-V3-GGUF)..."
    python $P/publish_hf.py --gguf_dir "$DRIVE/gguf_v3" --final "$E/gate_s1.0.json" > "$L/v3_upload.log" 2>&1 \
      && mark upload || echo "HF upload warning (files saved in Drive gguf_v3); see v3_upload.log"
  else
    echo "HF_TOKEN not set; skipping upload. All GGUF binaries are preserved at $DRIVE/gguf_v3"
    mark upload
  fi
}

echo "=========================================================="
echo " THRENODY V3 TRAINING PIPELINE COMPLETED SUCCESSFULLY!"
echo " Finished at: $(date -u)"
echo " Chosen candidate: $(cat $E/chosen.txt 2>/dev/null || echo 's1.0')"
echo " Output files in Drive: $DRIVE/gguf_v3"
echo "=========================================================="
