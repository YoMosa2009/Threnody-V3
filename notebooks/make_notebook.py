"""Generates notebooks/Threnody.ipynb (run from repo root). Every cell is idempotent and resumable."""
import json

CELLS = [
("md", """# Threnody V3 — Automated Training & Verification Notebook
### Dominating the Base Model at Full Adapter Strength (`s1.0`)

**GPU Recommendation:**
- **Recommended: NVIDIA A100-SXM4-80GB (High-RAM)**:
  - Native precompiled wheel support for `causal-conv1d`, `flash-attn`, `triton`, and `flash-linear-attention`.
  - Cheaper credit burn rate (~7.5–8.0 CU/hr vs ~11 CU/hr on G4).
  - 80 GB VRAM is plenty for 32k context LoRA SFT on 9B (~40 GB utilized).
- **Alternative: G4 (RTX PRO 6000 96GB)**:
  - Use if A100 is unavailable or if concurrent two-model vLLM serving is needed. Note that `causal_conv1d` requires source compilation on sm_120.

**Google Drive Sharing (from `mosaalsaegh@gmail.com` to `Moonlightstar1975@gmail.com`):**
1. In `mosaalsaegh@gmail.com`: Right-click `MyDrive/Threnody` in Google Drive -> **Share** -> add `Moonlightstar1975@gmail.com` as **Editor**.
2. In `Moonlightstar1975@gmail.com`: Open Google Drive -> **Shared with me** -> right-click `Threnody` -> **Add shortcut to Drive** -> select **My Drive**.
3. All teacher data (`traj_v2`, `sci_v2`, `code_verified`, `regen`) will instantly be accessible in Colab without spending hours or credits regenerating!"""),

("code", """# 1. GPU & Google Drive Connection
!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
from google.colab import drive
import os, glob

drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/Threnody'
os.makedirs(DRIVE, exist_ok=True)
os.makedirs(f'{DRIVE}/markers', exist_ok=True)
os.makedirs(f'{DRIVE}/logs', exist_ok=True)
os.makedirs(f'{DRIVE}/teacher', exist_ok=True)

# Check for existing/transferred teacher data from mosaalsaegh@gmail.com
teacher_files = glob.glob(f'{DRIVE}/teacher/*.jsonl')
print(f"Mounted Drive at: {DRIVE}")
if teacher_files:
    print(f"Detected {len(teacher_files)} teacher dataset files in Drive:")
    for f in teacher_files:
        print(f"  - {os.path.basename(f)} ({os.path.getsize(f) / (1024*1024):.2f} MB)")
else:
    print("Notice: No teacher files found yet in MyDrive/Threnody/teacher.")
    print("If you shared Threnody from mosaalsaegh@gmail.com, make sure to add it as a shortcut into MyDrive.")"""),

("code", """# 2. Repo Code & Python Dependencies
!rm -rf /content/Threnody && git clone -q https://github.com/YoMosa2009/Threnody /content/Threnody
!test -d /content/OmniAgent-Bench || git clone -q https://github.com/YoMosa2009/OmniAgent-Bench /content/OmniAgent-Bench
!pip -q install -U "transformers>=4.57" "peft>=0.17" "datasets>=3" accelerate flash-linear-attention && pip -q uninstall -y torchaudio torchao 2>/dev/null
!pip -q install --no-build-isolation causal-conv1d flash-attn || echo "Standard PyTorch DeltaNet fallback active"

import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
import transformers, peft, torch
print(f"Environment ready | Transformers: {transformers.__version__} | PEFT: {peft.__version__} | PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")"""),

("code", """# 3. llama.cpp (CUDA) — Built once, cached in Drive by GPU Architecture
import os, subprocess
CC = subprocess.check_output("nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .", shell=True).decode().strip()
CC = CC if CC else "80"
CACHE = f'{DRIVE}/llama_bin_sm{CC}.tgz'

if not os.path.exists('/content/llama.cpp'):
    !git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp

if os.path.exists(CACHE):
    print(f"Restoring cached llama.cpp build for sm_{CC} from Drive...")
    !mkdir -p /content/llama.cpp/build && tar xzf {CACHE} -C /content/llama.cpp/build
else:
    print(f"Compiling llama.cpp for sm_{CC} (takes ~2 minutes)...")
    !cd /content/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={CC} -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF > /dev/null && cmake --build build -j 12 --target llama-server llama-quantize llama-imatrix llama-perplexity llama-cli > /tmp/build.log 2>&1; tail -3 /tmp/build.log
    !cd /content/llama.cpp/build && tar czf {CACHE} bin

!pip -q install /content/llama.cpp/gguf-py
!ls -lh /content/llama.cpp/build/bin/llama-server"""),

("code", """# 4. RUN V3 FULL AUTOMATED PIPELINE (Detached & Unattended)
# Executes: data build -> base eval -> SFT -> checkpoint gating -> s1.0 release gate -> GGUF export
import subprocess
os.makedirs(f'{DRIVE}/logs', exist_ok=True)
subprocess.run(['pkill', '-f', 'bash pipeline/[r]un_v3'])
subprocess.run('cd /content/Threnody && git pull -q && git log --oneline -1', shell=True)
subprocess.Popen(f'cd /content/Threnody && DRIVE={DRIVE} nohup bash pipeline/run_v3.sh >> {DRIVE}/logs/run_v3.log 2>&1', shell=True, start_new_session=True)
print('>>> Threnody V3 training pipeline launched in background! Safe to close or leave running.')"""),

("code", """# 5. WATCHDOG (Background Process)
# Writes heartbeat.txt every 5 min + ALERT.txt on any errors into Google Drive
subprocess.run(['pkill', '-f', 'pipeline/[w]atchdog'])
subprocess.Popen(f'cd /content/Threnody && DRIVE={DRIVE} nohup bash pipeline/watchdog.sh > /tmp/watchdog.out 2>&1', shell=True, start_new_session=True)
print('>>> Watchdog launched.')"""),

("code", """# 6. LIVE DASHBOARD & PROCESS MONITOR
!echo "=== GPU Memory & Utilization ==="
!nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
!echo ""
!echo "=== Active Training Processes ==="
!ps aux | grep -E "run_v3|train_sft|llama-server|ckpt_gate|build_v3" | grep -v grep || echo "No pipeline process currently active"
!echo ""
!echo "=== Watchdog Heartbeat ==="
!cat {DRIVE}/heartbeat.txt 2>/dev/null || echo "Heartbeat pending..."
!cat {DRIVE}/ALERT.txt 2>/dev/null | head -5
!echo ""
!echo "=== Pipeline Log Tail (Last 15 lines) ==="
!tail -15 {DRIVE}/logs/run_v3.log 2>/dev/null || echo "run_v3.log not created yet" """),

("code", """# 7. SFT LOSS & CHECKPOINT GATING PROGRESS
!echo "=== SFT Training Progress ==="
!tail -c 4000 {DRIVE}/logs/v3_sft.log 2>/dev/null | tr '\\r' '\\n' | grep -v 'Loading weights' | tail -12 || echo "No SFT logs yet"
!echo ""
!echo "=== Checkpoint Gate Evaluation ==="
!grep -E "^(GATE|PASS|FAIL)" {DRIVE}/logs/v3_gate.log 2>/dev/null | tail -10 || echo "No checkpoint gate logs yet" """),

("code", """# 8. DRY RUN (Optional Mini-Test Slice: ~15-20 min)
# Use this to verify all stages without burning GPU credits
# import subprocess
# subprocess.Popen(f'cd /content/Threnody && DRY=1 DRIVE={DRIVE} nohup bash pipeline/run_v3.sh >> {DRIVE}/logs/dry_v3.log 2>&1', shell=True, start_new_session=True)
# print('Dry run launched')"""),

("code", """# 9. FINAL GGUF & GATING VERIFICATION SUMMARY
import json, glob
print("=== Released Model & Gate Report ===")
chosen_f = f"{DRIVE}/eval_v3/chosen.txt"
if os.path.exists(chosen_f):
    print("CHOSEN CANDIDATE:", open(chosen_f).read().strip())
gate_files = glob.glob(f"{DRIVE}/eval_v3/gate_*.json")
for gf in gate_files:
    print(f"\n--- {os.path.basename(gf)} ---")
    data = json.load(open(gf))
    print(f"Overall PASS: {data.get('pass')}")
    for chk in data.get("checks", []):
        mark = "✓" if chk.get("pass") else "✗"
        print(f"  [{mark}] {chk['gate']}: cand={chk['cand']} vs base={chk['base']}")
!echo "\n=== GGUF Export Artifacts ==="
!ls -lh {DRIVE}/gguf_v3 2>/dev/null || echo "GGUF exports not generated yet" """),
]


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def build():
    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "machine_shape": "hm"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}
        },
        "cells": [cell(k, s) for k, s in CELLS]
    }
    with open("notebooks/Threnody.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Wrote notebooks/Threnody.ipynb successfully!")


if __name__ == "__main__":
    build()
