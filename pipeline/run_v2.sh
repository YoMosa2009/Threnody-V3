#!/usr/bin/env bash
# Threnody V2 unattended pipeline (Colab G4 / A100-80GB). See docs/02_V2_PLAN.md.
# Every stage writes markers/v2_<stage>.done on success and is skipped on re-run. A failing stage prints
# "!!! STAGE <x> FAILED" and stops the pipeline (the watchdog turns that into an alert); it never writes its marker.
#   DRIVE=/content/drive/MyDrive/Threnody nohup bash pipeline/run_v2.sh >> $DRIVE/logs/run_v2.log 2>&1 &
# DRY=1 runs every stage on a tiny slice (minutes) to prove the pipeline before spending credits.
set -u
DRIVE=${DRIVE:-/content/drive/MyDrive/Threnody}
DRY=${DRY:-0}
T=$DRIVE/teacher; M=$DRIVE/markers; L=$DRIVE/logs; E=$DRIVE/eval_v2
RUN=$DRIVE/sft_v3; DATA=$DRIVE/data_v2b
if [ "$DRY" = 1 ]; then T=$DRIVE/dry/teacher; M=$DRIVE/dry/markers; L=$DRIVE/dry/logs; E=$DRIVE/dry/eval_v2
  RUN=$DRIVE/dry/sft_v3; DATA=$DRIVE/dry/data_v2b; fi
mkdir -p "$T" "$M" "$L" "$E" /content/models
export DRIVE GATES_DRY=$DRY TEACHER_DIR=$T LLAMA=/content/llama.cpp BENCH=/content/OmniAgent-Bench HF_HUB_ENABLE_HF_TRANSFER=0
cd /content/Threnody
P=pipeline
stage() { [ -f "$M/v2_$1.done" ] && { echo "skip $1"; return 1; }; echo "=== $(date -u +%H:%M) start $1"; return 0; }
mark() { touch "$M/v2_$1.done"; echo "=== $(date -u +%H:%M) done $1"; }
fail() { echo "!!! STAGE $1 FAILED $(date -u +%H:%M) ($2)"; exit 1; }
n() { [ "$DRY" = 1 ] && echo "$2" || echo "$1"; }   # n <full> <dry>
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

# ---- 0. environment (CPU work in parallel): vLLM, llama.cpp for this GPU, causal_conv1d
pip uninstall -y -q torchaudio 2>/dev/null
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
if [ ! -f /content/llama.cpp/build/.arch_$CC ]; then
  CACHE=$DRIVE/llama_bin_sm$CC.tgz
  if [ -f "$CACHE" ]; then tar xzf "$CACHE" -C /content/llama.cpp/build && touch /content/llama.cpp/build/.arch_$CC
  else (cd /content/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC -DLLAMA_CURL=OFF \
          -DBUILD_SHARED_LIBS=OFF > /dev/null && cmake --build build -j 12 --target llama-server llama-quantize \
          llama-imatrix llama-perplexity llama-cli > "$L/llama_build.log" 2>&1 && touch build/.arch_$CC \
          && tar czf "$CACHE" -C build bin) & fi
fi
[ "$DRY" = 1 ] || python -c "import causal_conv1d" 2>/dev/null || \
  (timeout 40m pip install -q causal-conv1d --no-build-isolation > "$L/causal_conv1d.log" 2>&1 || echo "causal_conv1d not installed (slow path)" >> "$L/causal_conv1d.log") &
VLLM_BIN=$(command -v vllm || echo /content/vllm-env/bin/vllm)
if [ ! -f "$M/v2_sample.done" ] && [ ! -x "$VLLM_BIN" ]; then
  { pip -q install virtualenv && python -m virtualenv /content/vllm-env && /content/vllm-env/bin/pip install "vllm==0.32.0"; } \
    > "$L/vllm_install.log" 2>&1 || fail env "vllm install, see vllm_install.log"
  VLLM_BIN=/content/vllm-env/bin/vllm
fi

# ---- 1. generation: teacher (Qwen3.8-27B-FP8) + student (Qwen3.5-9B base) served together
up() { for i in $(seq 1 90); do curl -sf localhost:$1/health >/dev/null && return 0; sleep 10; done; return 1; }
serve() {
  # The student must start only after the teacher has finished loading: vLLM's memory profiler fails if another
  # process's GPU usage changes while it measures. 0.66 + 0.26 of 96 GB leaves the 27B FP8 teacher ~14 GB of KV cache.
  "$VLLM_BIN" serve Qwen/Qwen3.8-27B-FP8 --served-model-name teacher --language-model-only --max-model-len 65536 \
    --max-num-seqs 48 --gpu-memory-utilization 0.66 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 --port 8000 > "$L/vllm_teacher.log" 2>&1 &
  up 8000 || return 1
  "$VLLM_BIN" serve Qwen/Qwen3.5-9B --served-model-name student --language-model-only --max-model-len 65536 \
    --max-num-seqs 32 --gpu-memory-utilization 0.26 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 --port 8001 > "$L/vllm_student.log" 2>&1 &
  up 8001
}
if [ ! -f "$M/v2_sample.done" ] || [ ! -f "$M/v2_sci.done" ] || [ ! -f "$M/v2_code.done" ]; then
  serve || fail serve "vLLM servers did not come up; see vllm_teacher.log / vllm_student.log"
  G="python $P/gen_v2.py --out $T --workers $(n 48 8)"
  stage hard && { timeout $(n 25m 6m) $G --mode hard --n $(n 60 6) > "$L/v2_hard.log" 2>&1
    [ "$(lines $T/hard_eval.jsonl)" -ge $(n 20 1) ] && mark hard || fail hard "only $(lines $T/hard_eval.jsonl) hard eval tasks"; }
  # V1's validated tasks are reused (copied in once); new tasks add variety.
  [ -f "$T/tasks_train.jsonl" ] || cp "$DRIVE/teacher/tasks_train.jsonl" "$T/" 2>/dev/null
  [ "$DRY" = 1 ] && mark synth
  stage synth && { timeout $(n 35m 8m) $G --mode synth --n $(n 300 3) > "$L/v2_synth.log" 2>&1; mark synth; }
  stage sci && { timeout $(n 80m 6m) $G --mode sci --n $(n 1500 6) > "$L/v2_sci.log" 2>&1 &
                 SCI_PID=$!; }
  stage code && { timeout $(n 75m 6m) python $P/gen_teacher.py --mode code --n $(n 3000 20) --workers $(n 16 4) --out "$T"                     > "$L/v2_code.log" 2>&1 & CODE_PID=$!; }
  stage sample && { timeout $(n 80m 6m) $G --mode sample --k $(n 3 1) --max_tasks $(n 0 4) > "$L/v2_sample.log" 2>&1
    [ "$(lines $T/traj_v2.jsonl)" -ge $(n 300 1) ] && mark sample || fail sample "only $(lines $T/traj_v2.jsonl) verified trajectories"; }
  [ -n "${CODE_PID:-}" ] && { wait $CODE_PID; mark code; }  # optional slice: an empty file just means no code data
  [ -n "${SCI_PID:-}" ] && { wait $SCI_PID; [ "$(lines $T/sci_v2.jsonl)" -ge $(n 200 1) ] && mark sci || fail sci "only $(lines $T/sci_v2.jsonl) sci records"; }
  pkill -f "[v]llm serve"; sleep 20
fi
wait  # llama.cpp build / causal_conv1d

# ---- 2. data mix (CPU) || base fast reference on the GPU
stage data && { [ "$DRY" = 1 ] && [ ! -f "$T/regen.jsonl" ] && cp "$DRIVE/teacher/regen.jsonl" "$T/" 2>/dev/null
  python $P/build_v2.py --out "$DATA" --total_tokens $(n 25e6 3e5) --max_scan $(n 20000 200) > "$L/v2_data.log" 2>&1 &
  DATA_PID=$!; }
stage base_fast && {
  BASE_HF=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-9B'))")
  G8=$(python $P/gates.py build --src "$BASE_HF" --name base --quant Q8_0 --outdir /content/models 2>>"$L/v2_base_fast.log" | tail -1)
  python $P/gates.py eval --gguf "$G8" --suite fast --out "$E/base_fast" --tag base >> "$L/v2_base_fast.log" 2>&1 \
    && mark base_fast || fail base_fast "see v2_base_fast.log"; }
[ -n "${DATA_PID:-}" ] && { wait $DATA_PID && mark data || fail data "see v2_data.log"; }

# ---- 3. SFT with the checkpoint gate beside it, and the base full reference (same GPU, overlapping)
stage sft && { rm -f /content/STOP_TRAINING
  python $P/ckpt_gate.py --run_dir "$RUN" --base_fast "$E/base_fast/results.json" --stop_file /content/STOP_TRAINING \
    --every $(n 40 4) > "$L/v2_gate.log" 2>&1 &
  GATE_PID=$!
  ( sleep 600; BASE_HF=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-9B'))")
    G4=$(python $P/gates.py build --src "$BASE_HF" --name base --quant Q4_K_M --calib "$DATA/val.jsonl" --outdir /content/models | tail -1)
    python $P/gates.py eval --gguf "$G4" --suite $(n full fast) --out "$E/base_full" --tag base --port 8092 ) > "$L/v2_base_full.log" 2>&1 &
  BASEFULL_PID=$!
  python $P/train_sft.py --data "$DATA" --out "$RUN" --max_len 32768 --epochs 1 --lr 1e-4 --max_hours $(n 3.5 0.2) \
    --stop_file /content/STOP_TRAINING $( [ "$DRY" = 1 ] && echo "--max_steps 4 --grad_accum 2 --save_steps 2" ) \
    > "$L/v2_sft.log" 2>&1 || fail sft "see v2_sft.log"
  wait $GATE_PID; wait $BASEFULL_PID || fail sft "base full eval failed, see v2_base_full.log"
  mark sft; }

# ---- 4. candidates: fast screen (Q8_0, same build as base_fast), then full eval of the top 2 (Q4_K_M imatrix) + gate
mk() {  # mk <name> <adapter> <strength> <ties> -> /content/merged_<name> with the pinned chat template
  [ -d /content/merged_$1 ] || { python $P/merge.py --adapter "$2" --out /content/merged_$1 --strength $3 --ties_density $4 \
    >> "$L/v2_select.log" 2>&1 < /dev/null && python $P/gates.py pin --dir /content/merged_$1 < /dev/null; }; }
stage select && {
  python $P/gates.py candidates --run_dir "$RUN" > /content/cands.txt || fail select "no adapters found"
  [ "$DRY" = 1 ] && sed -i '2,$d' /content/cands.txt
  cat /content/cands.txt
  while read -r name adapter s ties; do
    [ -f "$E/cand_$name/results.json" ] && continue
    mk $name "$adapter" $s $ties || continue
    g=$(python $P/gates.py build --src /content/merged_$name --name $name --quant Q8_0 --outdir /content/models 2>>"$L/v2_select.log" < /dev/null | tail -1)
    python $P/gates.py eval --gguf "$g" --suite fast --out "$E/cand_$name" --tag $name >> "$L/v2_select.log" 2>&1 < /dev/null
    rm -rf "$g" /content/merged_$name
  done < /content/cands.txt
  python $P/gates.py rank --eval_dir "$E" | tee /content/ranked.txt
  PASSED=""
  for name in $(head -2 /content/ranked.txt | cut -d' ' -f1); do
    read -r _ adapter s ties < <(grep "^$name " /content/cands.txt)
    mk $name "$adapter" $s $ties || continue
    g=$(python $P/gates.py build --src /content/merged_$name --name $name --quant Q4_K_M --calib "$DATA/val.jsonl" --outdir /content/models | tail -1)
    python $P/gates.py eval --gguf "$g" --suite $(n full fast) --out "$E/full_$name" --tag $name >> "$L/v2_select.log" 2>&1
    if python $P/gates.py gate --cand "$E/full_$name/results.json" --base "$E/base_full/results.json" --suite $(n full fast) \
         --out "$E/gate_$name.json" >> "$L/v2_select.log" 2>&1; then PASSED=$name; break; fi
    echo "GATE FAILED for $name" >> "$L/v2_select.log"; rm -rf /content/merged_$name
  done
  grep -E "^(PASS|FAIL) |^GATE" "$L/v2_select.log" | tail -30
  [ -n "$PASSED" ] || fail select "no candidate passed every gate vs the same-build base (see $E/gate_*.json); nothing will be released"
  echo "$PASSED" > "$E/chosen.txt"; mark select; }

# ---- 5. export the chosen model: imatrix quant ladder + KLD, vision mmproj, report
stage export && { C=$(cat "$E/chosen.txt")
  python $P/export_gguf.py --hf /content/merged_$C --llama /content/llama.cpp --calib "$DATA/val.jsonl" \
    --out "$DRIVE/gguf_v2" --name Threnody-9B $( [ "$DRY" = 1 ] && echo "--quants Q4_K_M" ) > "$L/v2_export.log" 2>&1 \
    || fail export "see v2_export.log"
  python /content/llama.cpp/convert_hf_to_gguf.py /content/merged_$C --mmproj \
    --outfile "$DRIVE/gguf_v2/mmproj-Threnody-9B-F16.gguf" --outtype f16 >> "$L/v2_export.log" 2>&1 \
    || echo "mmproj export failed (text model unaffected)" >> "$L/v2_export.log"
  mark export; }
echo "=== V2 FINISHED $(date -u) chosen=$(cat $E/chosen.txt 2>/dev/null)"
