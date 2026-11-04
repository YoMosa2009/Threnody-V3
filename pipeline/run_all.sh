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

# ---- 0. env repair: a stray vLLM install upgraded torch; mismatched torchaudio breaks transformers imports
pip uninstall -y -q torchaudio 2>/dev/null
VLLM_BIN=$(command -v vllm || echo /content/vllm-env/bin/vllm)

# ---- 1. teacher (Qwen3.8-27B-FP8 via vLLM), verified data generation
if [ ! -f "$M/teacher_all.done" ]; then
  if [ ! -x "$VLLM_BIN" ]; then
    { pip -q install virtualenv && python -m virtualenv /content/vllm-env && /content/vllm-env/bin/pip install "vllm==0.32.0"; } > "$L/vllm_install.log" 2>&1
    VLLM_BIN=/content/vllm-env/bin/vllm
  fi
  rm -f "$M/teacher_failed"
  [ -x "$VLLM_BIN" ] || { echo "!!! no vllm binary"; touch "$M/teacher_failed"; }
  [ -f "$M/teacher_failed" ] || "$VLLM_BIN" serve Qwen/Qwen3.8-27B-FP8 --served-model-name teacher --language-model-only \
    --max-model-len 65536 --max-num-seqs 32 --gpu-memory-utilization 0.92 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --port 8000 > "$L/vllm.log" 2>&1 &
  for i in $(seq 1 180); do [ -f "$M/teacher_failed" ] && break; curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 10; done
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then
    stage synth_tasks && { timeout 60m python pipeline/gen_teacher.py --mode synth_tasks --n 400 --workers 32 --out "$T" > "$L/synth_tasks.log" 2>&1; mark synth_tasks; }
    stage solve_tasks && { timeout 120m python pipeline/gen_teacher.py --mode solve_tasks --k 2 --workers 32 --out "$T" > "$L/solve_tasks.log" 2>&1; mark solve_tasks; }
    stage code && { timeout 45m python pipeline/gen_teacher.py --mode code --n 2500 --workers 32 --out "$T" > "$L/code.log" 2>&1; mark code; }
    stage regen && { timeout 30m python pipeline/gen_teacher.py --mode regen --n 1200 --workers 32 --out "$T" > "$L/regen.log" 2>&1; mark regen; }
    mark teacher_all
  else
    echo "!!! teacher server failed to start; see vllm.log. Continuing without teacher data."
  fi
  pkill -f "[v]llm serve" 2>/dev/null; sleep 30
fi

# ---- 2. dataset: public June-2026+ data (data_v2) + verified teacher outputs -> data_final
stage data_v2 && { python pipeline/build_data.py --out "$DRIVE/data_v2" --mix v2 --max_tokens 32768 > "$L/data_v2.log" 2>&1 && mark data_v2; }
stage data_teacher && { python pipeline/build_data.py --out "$DRIVE/data_teacher" --mix v2 --max_tokens 32768 \
    --only teacher_agent,teacher_code,teacher_regen --no_identity > "$L/data_teacher.log" 2>&1; mark data_teacher; }
stage data_final && { python - "$DRIVE" <<'EOF'
import json, os, random, sys
d = sys.argv[1]; out = f"{d}/data_final"; os.makedirs(out, exist_ok=True)
for split in ("train", "val"):
    rows = []
    for src in ("data_v2", "data_teacher"):
        p = f"{d}/{src}/{split}.jsonl"
        if os.path.exists(p):
            rows += [l for l in open(p, encoding="utf-8") if l.strip()]
    random.Random(1).shuffle(rows)
    open(f"{out}/{split}.jsonl", "w", encoding="utf-8").writelines(rows)
    print(split, len(rows))
EOF
  mark data_final; }

# ---- 3. SFT (resumable)
stage sft && { python pipeline/train_sft.py --data "$DRIVE/data_final" --out "$DRIVE/sft_v2" --max_len 32768 --epochs 1 --lr 1.5e-4 --max_hours 9 > "$L/sft.log" 2>&1 && mark sft; }

# ---- 4. merge sweep + GGUF + bench (OmniAgent-Bench + AgentLoop, both modes)
if [ -f "$M/sft.done" ] && stage eval; then
  # the Drive-cached llama.cpp is built for sm_80 (A100); rebuild for any other GPU (e.g. G4 = sm_120)
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  if [ "$CC" != "80" ] && [ ! -f /content/llama.cpp/build/.arch_$CC ]; then
    (cd /content/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF \
      && cmake --build build -j 12 --target llama-server llama-quantize llama-imatrix llama-perplexity llama-cli \
      && touch build/.arch_$CC) >> "$L/eval.log" 2>&1
  fi
  mkdir -p /content/models
  export STRENGTHS=${STRENGTHS:-"1.0 0.75"}
  python - <<'EOF'
import json, os
json.dump([{"id": f"threnody-s{s}", "name": f"Threnody-9B merge {s}", "path": f"/content/models/Threnody-9B-s{s}-Q4_K_M.gguf",
            "modes": ["nothink", "think"]} for s in os.environ["STRENGTHS"].split()], open("/content/models_threnody.json", "w"))
EOF
  for s in $STRENGTHS; do
    g=/content/models/Threnody-9B-s$s-Q4_K_M.gguf
    [ -f "$g" ] && continue
    python pipeline/merge.py --adapter "$DRIVE/sft_v2/final_adapter" --out /content/merged_$s --strength $s >> "$L/eval.log" 2>&1
    python /content/llama.cpp/convert_hf_to_gguf.py /content/merged_$s --outfile /content/tmp_bf16.gguf --outtype bf16 >> "$L/eval.log" 2>&1
    /content/llama.cpp/build/bin/llama-quantize /content/tmp_bf16.gguf "$g" Q4_K_M >> "$L/eval.log" 2>&1
    rm -rf /content/tmp_bf16.gguf /content/merged_$s
    [ -s "$g" ] || { echo "!!! quantize failed for s=$s (see eval.log)"; exit 1; }
  done
  python pipeline/bench_colab.py --llama /content/llama.cpp/build/bin --bench /content/OmniAgent-Bench \
    --models /content/models_threnody.json --out "$DRIVE/bench_threnody" --agentloop >> "$L/eval.log" 2>&1 \
    && ! grep -q "server failed" "$L/eval.log" && mark eval || echo "!!! bench failed (see eval.log); eval not marked done"
fi
echo "=== ALL STAGES FINISHED $(date -u)"
