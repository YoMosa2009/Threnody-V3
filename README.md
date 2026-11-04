# Threnody V3

> *threnody* — a song of lamentation for the dead. An elegy for every bug it meets.

**Threnody-9B V3** is the frontier open-weight agentic coding model for long, multi-turn work inside CLI coding harnesses, fine-tuned from `Qwen/Qwen3.5-9B`.

Threnody V3 achieves **full adapter strength dominance (`s1.0`)**, strictly beating or matching the base model and V2 across all 4 benchmark suites with zero repetition loops.

---

## Benchmark Scorecard (V3 vs. V2 vs. Base)

![Threnody-9B V3 vs V2 vs Base](docs/bench_v3.svg)

All models evaluated under identical conditions: Q8_0 and Q4_K_M GGUF builds, pinned unsloth chat template, temperature 0, non-thinking mode.

| Benchmark | Qwen3.5-9B Base | Threnody-9B V2 (`s0.3`) | **Threnody-9B V3 (`s1.0`)** | Gate Status |
|---|---|---|---|---|
| **OmniAgent-Bench (Weighted)** | 92.7% | 92.7% | **93.8%** | **PASSED** ($\ge 93.8\%$ & $\ge 92.7\%$) |
| **AgentLoop In-Harness (14)** | 12/14 (85.7%) | 11/14 (78.6%) | **13/14 (92.9%)** | **PASSED** ($\ge 13/14$, beats Base) |
| ↳ Action Repetition Max | 18 | 2 | **1 (Zero Loops)** | **PASSED** (`max_repeat <= 2`) |
| **AgentLoop Held-Out (`opencode`, 14)** | 9/14 (64.3%) | 10/14 (71.4%) | **11/14 (78.6%)** | **PASSED** ($\ge 10/14$, beats Base) |
| **AgentLoop-Hard (30 held-out)** | 20/30 (66.7%) | N/A | **24/30 (80.0%)** | **PASSED** (+4 tasks over Base) |
| **BioChem-Hard (Overall)** | 35.8% | 37.6% | **41.5%** | **PASSED** ($\ge 40.0\%$) |
| ↳ Biology Subset | 42.1% | 44.5% | **48.0%** | **PASSED** |
| ↳ Chemistry Subset | 13.5% | 14.1% | **16.2%** | **PASSED** ($\ge 15.0\%$) |
| **MMLU-Pro Bio+Chem (400 questions)** | 75.3% | 76.5% | **79.8%** | **PASSED** |

Full gate results: [`results/v3_final.json`](results/v3_final.json).

---

## The 6 Pillars of V3

1. **Full Reasoning Restoration**: Fixed the vLLM extraction bug to extract both `reasoning_content` and `reasoning`, preserving interleaved `<think>` scratchpads across ~75% of trajectories.
2. **Zero-Tolerance Anti-Loop Filter**: Strictly drops any trajectory where an identical action was repeated $\ge 2$ times (`max_repeat < 2`). Trajectories with 0 repeats are sorted first, eliminating V2's recovery-ranking trap.
3. **Multi-Curriculum Balanced Mix (~35M tokens)**:
   - 35% multi-schema agent traces (8 tool schemas)
   - 20% algorithmic coding (`code_verified.jsonl`)
   - 20% Open-SWE-Traces
   - 15% BioChem reasoning (`sci_v2.jsonl`)
   - 10% cybersecurity & tool schema (`regen.jsonl` + identity)
4. **Full SFT Convergence**: Trained for ~1.2 epochs (300–400 steps) with learning rate 1e-4 and chunked cross-entropy.
5. **Hardware Optimization on A100-80GB**: Native `causal_conv1d` and `flash-linear-attention` wheels (`sm_80`), burning only ~7.5 CU/hr.
6. **Release at Full Strength (`s1.0`)**: 100% adapter weight — no dilution or softening required.

---

## Download & Quant Ladder (GGUF)

Hugging Face: **[MalxTech/Threnody-9B-V3-GGUF](https://huggingface.co/MalxTech/Threnody-9B-V3-GGUF)** (and [MalxTech/Threnody-9B-GGUF](https://huggingface.co/MalxTech/Threnody-9B-GGUF))

| File | Size | Description |
|---|---|---|
| `Threnody-9B-V3-Q4_K_M.gguf` | 5.6 GB | Recommended for 8 GB+ consumer GPUs (agentic imatrix) |
| `Threnody-9B-V3-Q5_K_M.gguf` | 6.8 GB | High-accuracy quantization |
| `Threnody-9B-V3-Q8_0.gguf` | 9.5 GB | Near-lossless reference quantization |
| `mmproj-Threnody-9B-V3-F16.gguf` | 0.9 GB | Optional multimodal vision projector |

---

## Inference with llama.cpp

Run the local server with native jinja chat templates:
```bash
llama-server -m Threnody-9B-V3-Q4_K_M.gguf -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --jinja
```

With multimodal vision projector:
```bash
llama-server -m Threnody-9B-V3-Q4_K_M.gguf --mmproj mmproj-Threnody-9B-V3-F16.gguf -ngl 99 -c 32768 -fa on --jinja
```

---

## Repository Structure
```
docs/           bench_v3.svg, release notes, progress log, status
eval/           AgentLoop harness + 8 tool schemas (6 train, 2 held-out)
notebooks/      Threnody.ipynb (Colab A100/G4 execution runbook)
pipeline/       build_v3.py, run_v3.sh, train_sft.py, ckpt_gate.py, gates.py, export_gguf.py, publish_hf.py
results/        v3_final.json (comprehensive scorecard & gate verifications)
```

## License
Apache-2.0
