# Threnody — Current Status & Release Handoff

_Last updated: 2026-11-05 (UTC)_

## 🚀 THRENODY 3.0 (V3) OFFICIALLY RELEASED

The V3 training, benchmarking, abliteration, and quantization pipeline ran on an **NVIDIA A100-SXM4-80GB High-RAM** instance, and the binaries are published. Threnody 3.0 is an **experimental agentic-coding fine-tune of Qwen3.5-9B** (supervised fine-tuning using LoRA, merged at strength 0.75). All benchmarks are author-reported. Conflicting records are listed in [`KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md); the published Hugging Face card has known errors, and a corrected card is in [`HF_MODEL_CARD.md`](HF_MODEL_CARD.md).

- **Canonical Identity**: **`Threnody 3.0`** (`Threnody-9B V3`)
- **Developer / Creator**: **`Axiom Inference`**
- **Base Architecture**: `Qwen/Qwen3.5-9B`
- **License**: Apache-2.0
- **Hugging Face Hub**: [**`MalxTech/Threnody-9B-V3-GGUF`**](https://huggingface.co/MalxTech/Threnody-9B-V3-GGUF)
- **GitHub Canonical**: [**`YoMosa2009/Threnody`**](https://github.com/YoMosa2009/Threnody)
- **GitHub V3 Branch**: [**`YoMosa2009/Threnody-V3`**](https://github.com/YoMosa2009/Threnody-V3)

---

## Final Benchmark Scorecard

Conditions per `run_v3.sh`: llama.cpp Q8_0 builds, pinned chat template, temperature 0, non-thinking mode only. Differences in percentage points (pp).

| Benchmark Suite | Qwen3.5-9B Base | Threnody-9B V2 (`s0.3`) | **Threnody 3.0 (`s0.75`)** | V3 vs. Base |
|---|---|---|---|---|
| **AgentLoop Held-Out (`opencode`, 14)** ¹ | 12/14 (85.7%) | 10/14 (71.4%) | **13/14 (92.9%)** | +7.1 pp; `max_repeat` 1 |
| **MMLU-Pro biology + chemistry (400-question subset)** | 76.0% | 76.5% | **78.5%** | +2.5 pp |
| ↳ Biology | 86.5% | not in current record | **86.5%** | 0.0 pp |
| ↳ Chemistry | 69.4% | not in current record | **73.5%** | +4.1 pp |
| **OmniAgent-Bench (Weighted)** | 92.7% (15/16) | 92.7% (15/16) | **84.6% (13/16)** | −8.1 pp |
| **AgentLoop In-Harness (14)** | 11/14 (78.6%) | 11/14 (78.6%) | **10/14 (71.4%)** | −7.1 pp; `max_repeat` 1 (HF card: 27) |
| **AgentLoop-Hard (30 held-out)** | 19/30 (63.3%) | N/A | **17/30 (56.7%)** | −6.7 pp; `max_repeat` 2 |

¹ The same suite was used in checkpoint and candidate selection.

The published gate records (`v3_final.json`, `v3_gate_s0.75.json`) mark this release as passing, but they do not match the gate criteria in the current `pipeline/gates.py` ([discrepancies §4](KNOWN_DISCREPANCIES.md#4-gate-records-vs-gate-code)).

---

## Published GGUF Artifacts & Quant Ladder

All variants use an imatrix computed on `val.jsonl` and were made from the abliterated (layers 12–28) merged model. "Same top token" is agreement with the BF16 GGUF on the same `val.jsonl` text. It measures token distribution, not task performance:

| File Name | Disk Size | Same Top Token vs. BF16 | Median KLD |
|---|---|---|---|
| `Threnody-9B-V3-Q8_0.gguf` | 8.87 GB | 97.756% | 0.000088 |
| `Threnody-9B-V3-Q6_K.gguf` | 6.85 GB | 96.664% | 0.000246 |
| `Threnody-9B-V3-Q5_K_M.gguf` | 6.02 GB | 96.127% | 0.000426 |
| `Threnody-9B-V3-Q4_K_M.gguf` | 5.24 GB | 94.058% | 0.001219 |
| `Threnody-9B-V3-IQ4_XS.gguf` | 4.84 GB | 93.161% | 0.001749 |
| `Threnody-9B-V3-IQ3_M.gguf` | 4.20 GB | 89.615% | 0.007465 |
| `mmproj-Threnody-9B-V3-F16.gguf` | 0.92 GB | N/A (F16) | N/A |

---

## Key Engineering Documentation in Repo
- **Known Discrepancies & Evaluation Limitations**: [`docs/KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md)
- **Corrected Hugging Face model card (not yet uploaded)**: [`docs/HF_MODEL_CARD.md`](HF_MODEL_CARD.md)
- **Postmortem & V4 Roadmap**: [`docs/04_V3_TRAINING_POSTMORTEM.md`](04_V3_TRAINING_POSTMORTEM.md)
- **Quantization Report & KLD Analysis**: [`results/v3_quant_report.json`](../results/v3_quant_report.json)
- **Training Report** (hyperparameters disagree with `train_sft.py`; see discrepancies §3): [`results/v3_training_report.json`](../results/v3_training_report.json)
- **Candidate Gate Records** (do not match current gate code; see discrepancies §4): [`results/v3_gate_s0.75.json`](../results/v3_gate_s0.75.json) & [`results/v3_gate_s1.0.json`](../results/v3_gate_s1.0.json)
- **Visual Performance Comparison**: [`docs/bench_v3.svg`](bench_v3.svg)
