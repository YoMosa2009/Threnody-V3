# Threnody — Current Status & Release Handoff

_Last updated: 2026-11-05 (UTC)_

## 🚀 THRENODY 3.0 (V3) OFFICIALLY RELEASED

The end-to-end V3 training, benchmarking, abliteration, and quantization pipeline has executed to completion on **NVIDIA A100-SXM4-80GB High-RAM**. All binaries, scorecards, and model cards are live.

- **Canonical Identity**: **`Threnody 3.0`** (`Threnody-9B V3`)
- **Developer / Creator**: **`Axiom Inference`**
- **Base Architecture**: `Qwen/Qwen3.5-9B` (Dense Transformer)
- **License**: Apache-2.0
- **Hugging Face Hub**: [**`MalxTech/Threnody-9B-V3-GGUF`**](https://huggingface.co/MalxTech/Threnody-9B-V3-GGUF)
- **GitHub Canonical**: [**`YoMosa2009/Threnody`**](https://github.com/YoMosa2009/Threnody)
- **GitHub V3 Branch**: [**`YoMosa2009/Threnody-V3`**](https://github.com/YoMosa2009/Threnody-V3)

---

## Final Benchmark Scorecard

Evaluated under standardized conditions: llama.cpp Q8_0 and Q4_K_M builds, pinned chat template, temperature 0, non-thinking mode.

| Benchmark Suite | Qwen3.5-9B Base | Threnody-9B V2 (`s0.3`) | **Threnody 3.0 (`s0.75`)** | Result & Dominance Status |
|---|---|---|---|---|
| **AgentLoop Held-Out (`opencode`, 14)** | 12/14 (85.7%) | 10/14 (71.4%) | **13/14 (92.9%)** | **PASSED** (+7.2% over Base, rep=1) |
| **MMLU-Pro Bio+Chem (400 questions)** | 76.0% | 76.5% | **78.5%** | **PASSED** (+2.5% over Base) |
| ↳ Biology Subset | 86.5% | 44.5% | **86.5%** | **PASSED** (Parity with Base, +42.0% over V2) |
| ↳ Chemistry Subset | 69.4% | 14.1% | **73.5%** | **PASSED** (+4.1% over Base, +59.4% over V2) |
| **OmniAgent-Bench (Weighted)** | 92.7% | 92.7% | **84.6%** | 13/16 multi-turn schemas passed |
| **AgentLoop In-Harness (14)** | 11/14 (78.6%) | 11/14 (78.6%) | **10/14 (71.4%)** | Zero repetition loops on passes |
| **AgentLoop-Hard (30 held-out)** | 19/30 (63.3%) | N/A | **17/30 (56.7%)** | Resilient against repetition loops |

---

## Published GGUF Artifacts & Quant Ladder

All variants calibrated against an agentic multi-turn importance matrix (`imatrix`) and directional refusal abliteration across layers 12–28:

| File Name | Disk Size | Same-Choice Top-p Retention | Role / Recommended Hardware |
|---|---|---|---|
| `Threnody-9B-V3-Q8_0.gguf` | 8.87 GB | **97.8%** | Reference quantization, maximum accuracy |
| `Threnody-9B-V3-Q6_K.gguf` | 6.85 GB | **96.7%** | Near-lossless agentic deployment |
| `Threnody-9B-V3-Q5_K_M.gguf` | 6.02 GB | **96.1%** | Sweet spot for 8 GB+ VRAM GPUs |
| `Threnody-9B-V3-Q4_K_M.gguf` | 5.24 GB | **94.1%** | Standard recommended consumer build |
| `Threnody-9B-V3-IQ4_XS.gguf` | 4.84 GB | **93.2%** | Fast 4-bit execution on memory-limited VRAM |
| `Threnody-9B-V3-IQ3_M.gguf` | 4.20 GB | **89.6%** | Ultra-compact 4 GB edge footprint |
| `mmproj-Threnody-9B-V3-F16.gguf` | 0.92 GB | **FP16 Lossless** | Decoupled Multimodal Vision Projector |

---

## Key Engineering Documentation in Repo
- **Comprehensive Postmortem & V4 Roadmap**: [`docs/04_V3_TRAINING_POSTMORTEM.md`](04_V3_TRAINING_POSTMORTEM.md)
- **Quantization Report & KLD Analysis**: [`results/v3_quant_report.json`](../results/v3_quant_report.json)
- **Training Hyperparameters & Loss Curves**: [`results/v3_training_report.json`](../results/v3_training_report.json)
- **Candidate Gate Verifications**: [`results/v3_gate_s0.75.json`](../results/v3_gate_s0.75.json) & [`results/v3_gate_s1.0.json`](../results/v3_gate_s1.0.json)
- **Visual Performance Comparison**: [`docs/bench_v3.svg`](bench_v3.svg)
