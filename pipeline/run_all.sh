#!/usr/bin/env bash
# Threnody unattended pipeline (Colab A100). Every stage is resumable and writes a DONE marker to Drive.
# A failing teacher stage never blocks training: SFT falls back to the public 2026-teacher data.
#   nohup bash pipeline/run_all.sh > $DRIVE/logs/run_all.log 2>&1 &
set -u
DRIVE=${DRIVE:-/content/drive/MyDrive/Threnody}
T=$DRIVE/teacher; M=$DRIVE/markers; L=$DRIVE/logs
mkdir -p "$T" "$M" "$L"
export TEACHER_DIR=$T TEACHER_URL=http://127.0.0.1:8000/v1 TEACHER_MODEL=teacher
cd /content/Threnody
stage() { [ -f "$M/$1.done" ] && { echo "skip $1"; return 1; }; echo "=== $(date -u +%H:%M) start $1"; return 0; }
mark() { touch "$M/$1.done"; echo "=== $(date -u +%H:%M) done $1"; }

# ---- 1. teacher server (Qwen3.8-27B-FP8, separate venv so Colab torch stays intact)
if [ ! -f "$M/teacher_all.done" ]; then
  if [ ! -x /content/vllm-env/bin/vllm ]; then
    { python -m venv /content/vllm-env && /content/vllm-env/bin/pip install -U pip &&       /content/vllm-env/bin/pip install "vllm==0.32.0" || /content/vllm-env/bin/pip install vllm; } > "$L/vllm_install.log" 2>&1
  fi
  if [ ! -x /content/vllm-env/bin/vllm ]; then
    echo "!!! vllm install failed (see vllm_install.log); skipping teacher stages"; touch "$M/teacher_failed"
  fi
  [ -f "$M/teacher_failed" ] || /content/vllm-env/bin/vllm serve Qwen/Qwen3.8-27B-FP8 --served-model-name teacher --language-model-only \
    --max-model-len 65536 --max-num-seqs 32 --gpu-memory-utilization 0.92 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --port 8000 > "$L/vllm.log" 2>&1 &
  VPID=$!
  for i in $(seq 1 180); do [ -f "$M/teacher_failed" ] && break; curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 10; done
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then
    stage synth_tasks && { timeout 75m python pipeline/gen_teacher.py --mode synth_tasks --n 400 --workers 32 --out "$T" > "$L/synth_tasks.log" 2>&1; mark synth_tasks; }
    stage solve_tasks && { timeout 150m python pipeline/gen_teacher.py --mode solve_tasks --k 2 --workers 32 --out "$T" > "$L/solve_tasks.log" 2>&1; mark solve_tasks; }
    stage code && { timeout 60m python pipeline/gen_teacher.py --mode code --n 2500 --workers 32 --out "$T" > "$L/code.log" 2>&1; mark code; }
    stage regen && { timeout 40m python pipeline/gen_teacher.py --mode regen --n 1200 --workers 32 --out "$T" > "$L/regen.log" 2>&1; mark regen; }
    mark teacher_all
  else
    echo "!!! teacher server failed to start; see vllm.log. Continuing without teacher data."
  fi
  kill $VPID 2>/dev/null; sleep 20; pkill -f "vllm serve" 2>/dev/null; sleep 10
fi

# ---- 2. dataset v2 (June-2026+ teachers only)
stage data_v2 && { python pipeline/build_data.py --out "$DRIVE/data_v2" --mix v2 --max_tokens 32768 > "$L/data_v2.log" 2>&1 && mark data_v2; }

# ---- 3. SFT (resumable)
stage sft && { python pipeline/train_sft.py --data "$DRIVE/data_v2" --out "$DRIVE/sft_v2" --max_len 32768 --epochs 1.5 --lr 1.5e-4 --max_hours 9 > "$L/sft.log" 2>&1 && mark sft; }

# ---- 4. merge sweep + GGUF + bench (OmniAgent-Bench + AgentLoop, both modes)
if [ -f "$M/sft.done" ] && stage eval; then
  python - <<'EOF'
import json
json.dump([{"id": f"threnody-s{s}", "name": f"Threnody-9B merge {s}", "path": f"/content/models/Threnody-9B-s{s}-Q4_K_M.gguf",
            "modes": ["nothink", "think"]} for s in (1.0, 0.75, 0.5)], open("/content/models_threnody.json", "w"))
EOF
  for s in 1.0 0.75 0.5; do
    g=/content/models/Threnody-9B-s$s-Q4_K_M.gguf
    [ -f "$g" ] && continue
    python pipeline/merge.py --adapter "$DRIVE/sft_v2/final_adapter" --out /content/merged_$s --strength $s >> "$L/eval.log" 2>&1
    python /content/llama.cpp/convert_hf_to_gguf.py /content/merged_$s --outfile /content/tmp_bf16.gguf --outtype bf16 >> "$L/eval.log" 2>&1
    /content/llama.cpp/build/bin/llama-quantize /content/tmp_bf16.gguf "$g" Q4_K_M >> "$L/eval.log" 2>&1
    rm -f /content/tmp_bf16.gguf
  done
  python pipeline/bench_colab.py --llama /content/llama.cpp/build/bin --bench /content/OmniAgent-Bench \
    --models /content/models_threnody.json --out "$DRIVE/bench_threnody" --agentloop >> "$L/eval.log" 2>&1 && mark eval
fi
echo "=== ALL STAGES FINISHED $(date -u)"
