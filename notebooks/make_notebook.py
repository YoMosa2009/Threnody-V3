"""Generates notebooks/Threnody.ipynb (run from repo root). Every cell is idempotent and resumable."""
import json

CELLS = [
("md", "# Threnody — training notebook (Colab A100)\nRun cells top to bottom. Every cell is safe to re-run after a disconnect; state lives in Google Drive `MyDrive/Threnody`."),
("code", """# 1. GPU + Drive
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/Threnody'
os.makedirs(DRIVE, exist_ok=True)"""),
("code", """# 2. Code + Python deps
!rm -rf /content/Threnody && git clone -q https://github.com/YoMosa2009/Threnody /content/Threnody
!test -d /content/OmniAgent-Bench || git clone -q https://github.com/YoMosa2009/OmniAgent-Bench /content/OmniAgent-Bench
!pip -q install -U "transformers>=4.57" "peft>=0.17" "datasets>=3" accelerate flash-linear-attention hf_transfer
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
import transformers, peft, torch; print(transformers.__version__, peft.__version__, torch.__version__)"""),
("code", """# 3. llama.cpp (CUDA) — built once, cached in Drive
import os
CACHE = f'{DRIVE}/llama_bin.tgz'
if not os.path.exists('/content/llama.cpp'):
    !git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
if os.path.exists(CACHE):
    !mkdir -p /content/llama.cpp/build && tar xzf {CACHE} -C /content/llama.cpp/build
else:
    !cd /content/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF > /dev/null && cmake --build build -j 12 --target llama-server llama-quantize llama-imatrix llama-perplexity llama-cli > /tmp/build.log 2>&1; tail -3 /tmp/build.log
    !cd /content/llama.cpp/build && tar czf {CACHE} bin
!pip -q install /content/llama.cpp/gguf-py
!ls /content/llama.cpp/build/bin"""),
("code", """# 4. Baseline benchmark: V4's base, Fable-Coder-V4, Threnody's base (same harness, same settings)
from huggingface_hub import hf_hub_download
for repo, f in [('Qwen/Qwen2.5-Coder-7B-Instruct-GGUF', 'qwen2.5-coder-7b-instruct-q4_k_m.gguf'),
                ('MalxTech/Fable-Coder-V4', 'fable-coder-v4-7b.Q4_K_M.gguf'),
                ('unsloth/Qwen3.5-9B-GGUF', 'Qwen3.5-9B-Q4_K_M.gguf')]:
    hf_hub_download(repo, f, local_dir='/content/models')
!cd /content/Threnody && python pipeline/bench_colab.py --llama /content/llama.cpp/build/bin --bench /content/OmniAgent-Bench --models pipeline/models_baseline.json --out {DRIVE}/bench_baseline"""),
("code", """# 5. Build the SFT dataset (streams from the HF Hub, decontaminates, length-filters)
if not os.path.exists(f'{DRIVE}/data/train.jsonl'):
    !cd /content/Threnody && python pipeline/build_data.py --out {DRIVE}/data --scale 1.0
!cat {DRIVE}/data/stats.json"""),
("code", """# 6. Pilot: 40 optimizer steps to measure throughput/memory and sanity-check loss
!cd /content/Threnody && python pipeline/train_sft.py --data {DRIVE}/data --out /content/pilot --max_steps 40 --save_steps 1000"""),
("code", """# 7. Full SFT (resumable — just re-run this cell after a disconnect)
!cd /content/Threnody && python pipeline/train_sft.py --data {DRIVE}/data --out {DRIVE}/sft --epochs 2 --lr 1.5e-4"""),
("code", """# 8. Merge sweep -> GGUF -> benchmark
STRENGTHS = [1.0, 0.75, 0.5]
for s in STRENGTHS:
    out = f'/content/merged_{s}'
    if not os.path.exists(out):
        !cd /content/Threnody && python pipeline/merge.py --adapter {DRIVE}/sft/final_adapter --out {out} --strength {s}
    g = f'/content/models/Threnody-9B-s{s}-Q4_K_M.gguf'
    if not os.path.exists(g):
        !python /content/llama.cpp/convert_hf_to_gguf.py {out} --outfile /content/tmp_bf16.gguf --outtype bf16
        !/content/llama.cpp/build/bin/llama-quantize /content/tmp_bf16.gguf {g} Q4_K_M
import json
json.dump([{"id": f"threnody-s{s}", "name": f"Threnody-9B merge {s}", "path": f"/content/models/Threnody-9B-s{s}-Q4_K_M.gguf",
            "modes": ["nothink", "think"]} for s in STRENGTHS], open('/content/models_threnody.json', 'w'))
!cd /content/Threnody && python pipeline/bench_colab.py --llama /content/llama.cpp/build/bin --bench /content/OmniAgent-Bench --models /content/models_threnody.json --out {DRIVE}/bench_threnody"""),
("code", """# 9. Final export of the chosen merge: imatrix quant ladder + KLD report
BEST = 1.0  # set from cell 8 results
!cd /content/Threnody && python pipeline/export_gguf.py --hf /content/merged_{BEST} --llama /content/llama.cpp --calib {DRIVE}/data/val.jsonl --out {DRIVE}/gguf"""),
]


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [cell(k, s) for k, s in CELLS]}
json.dump(nb, open("notebooks/Threnody.ipynb", "w"), indent=1)
print("wrote notebooks/Threnody.ipynb")
