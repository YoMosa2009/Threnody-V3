---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
pipeline_tag: text-generation
library_name: gguf
language:
- en
tags:
- open-weights
- gguf
- llama.cpp
- agentic-coding
- tool-use
- abliterated
- axiom-inference
- threnody
- v3
---

<!--
Corrected replacement for the model card at https://huggingface.co/MalxTech/Threnody-9B-V3-GGUF (README.md).
Scores are unchanged from results/v3_final.json in the GitHub repo.
Do not regenerate this card with pipeline/publish_hf.py: its hard-coded text would put back the
"STRICT DOMINANCE", "s1.0" and "zero dilution" claims.
-->

# Threnody-9B V3 (GGUF)

**Threnody-9B V3** (Threnody 3.0) is an **experimental agentic-coding fine-tune of `Qwen/Qwen3.5-9B`** by Axiom Inference. It is trained for multi-turn coding and tool calling across 8 harness schemas (6 used in training, 2 held out), plus biology/chemistry reasoning. Training was supervised fine-tuning using LoRA.

- **Source code**: [https://github.com/YoMosa2009/Threnody-V3](https://github.com/YoMosa2009/Threnody-V3)
- **Adapter strength**: `s0.75`. The LoRA update is scaled to 75% when merged into the base. `s1.0` (full strength) was evaluated and rejected.
- **Status**: mixed results vs. the base model. V3 improves on the held-out `opencode` suite and the MMLU-Pro bio/chem subset, and is lower on three agent suites. All results are author-reported.
- **Abliteration**: directional refusal orthogonalization on layers 12–28 (Arditi et al., 2024). No quantitative refusal or safety evaluation is published.

---

## Benchmark Scorecard

Author-reported. Conditions: llama.cpp Q8_0 builds, pinned chat template, temperature 0, non-thinking mode only, single run. Differences are in **percentage points (pp)**.

| Benchmark | Qwen3.5-9B Base | Threnody-9B V2 (s0.3) | Threnody-9B V3 (s0.75) | V3 vs. Base |
|---|---|---|---|---|
| AgentLoop Held-Out (`opencode`, 14) ¹ | 12/14 (85.7%) | 10/14 (71.4%) | **13/14 (92.9%)** | +7.1 pp |
| MMLU-Pro biology + chemistry (400-question subset) ² | 76.0% | 76.5% | **78.5%** | +2.5 pp |
|   ↳ Biology | 86.5% | — ³ | **86.5%** | 0.0 pp |
|   ↳ Chemistry | 69.4% | — ³ | **73.5%** | +4.1 pp |
| OmniAgent-Bench (weighted, 16) | 92.7% (15/16) | 92.7% (15/16) | **84.6% (13/16)** | −8.1 pp |
| AgentLoop In-Harness (14) | 11/14 (78.6%) | 11/14 (78.6%) | **10/14 (71.4%)** | −7.1 pp |
| AgentLoop-Hard (30 held-out) | 19/30 (63.3%) | N/A | **17/30 (56.7%)** | −6.7 pp |

¹ The `opencode` suite also contributed to checkpoint and candidate selection, so it is not a clean held-out estimate. 13/14 suggests transfer to this tested schema; it does not establish general tool understanding.
² A fixed sample of MMLU-Pro biology and chemistry test questions. This is not an overall MMLU-Pro score.
³ The V2 subset values in the previous version of this card (biology 44.5%, chemistry 14.1%, overall 37.6%) come from a different test, BioChem-Hard, and are not comparable with this row.

**Repetition (`max_repeat`)**: the longest run of identical consecutive actions, across all tasks in a suite. For V3 the GitHub record gives **1** on `opencode`, **1** on AgentLoop In-Harness and **2** on AgentLoop-Hard. The previous version of this card listed **27** for AgentLoop In-Harness. That conflict with the GitHub record is unresolved.

---

## Training

- Supervised fine-tuning using LoRA (`pipeline/train_sft.py`; rank 64 / alpha 32 by default) with assistant-only loss and chunked cross-entropy, on an A100-80GB.
- Data: a **35M-token configuration target**. The run reported **2,895,493 tokens across 662 trajectories**.
- Training trajectories with an immediately repeated action (`max_repeat >= 2`) were filtered out. About 75% of trajectories keep `<think>` scratchpads.
- The published training report's batch size and some other hyperparameters disagree with the trainer code. The settings that actually ran are unverified.

## Quantization

Each quant was compared with the BF16 GGUF over the same `val.jsonl` text used for imatrix calibration. "Same top token" is the percentage of positions where the quant's most likely token matches BF16's. These are token-distribution metrics, not task benchmarks.

| File | Size | Same top token vs. BF16 | Median KLD |
|---|---|---|---|
| `Threnody-9B-V3-Q8_0.gguf` | 8.87 GB | 97.756% | 0.000088 |
| `Threnody-9B-V3-Q6_K.gguf` | 6.85 GB | 96.664% | 0.000246 |
| `Threnody-9B-V3-Q5_K_M.gguf` | 6.02 GB | 96.127% | 0.000426 |
| `Threnody-9B-V3-Q4_K_M.gguf` | 5.24 GB | 94.058% | 0.001219 |
| `Threnody-9B-V3-IQ4_XS.gguf` | 4.84 GB | 93.161% | 0.001749 |
| `Threnody-9B-V3-IQ3_M.gguf` | 4.20 GB | 89.615% | 0.007465 |
| `mmproj-Threnody-9B-V3-F16.gguf` | 0.92 GB | N/A (F16 vision projector; no vision eval published) | N/A |

---

## Known discrepancies and limitations

- All benchmarks are author-reported, and V3 per-task outputs and run logs are not published.
- The published gate records mark this release as passing. They do not match the gate criteria in the repository's current `pipeline/gates.py`, which this candidate would fail.
- An earlier, all-passing `s1.0` result set was published in the GitHub repo and later replaced.
- The previous version of this card said "strict dominance", "full strength (`s1.0`)", "zero dilution" and "state-of-the-art". Those claims were not supported by its own table or by any external comparison, and have been removed.

Full list: [docs/KNOWN_DISCREPANCIES.md](https://github.com/YoMosa2009/Threnody-V3/blob/main/docs/KNOWN_DISCREPANCIES.md)

---

## Inference with llama.cpp

Run the local API server:
```bash
llama-server -m Threnody-9B-V3-Q4_K_M.gguf -ngl 99 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --jinja
```

To enable multimodal input:
```bash
llama-server -m Threnody-9B-V3-Q4_K_M.gguf --mmproj mmproj-Threnody-9B-V3-F16.gguf -ngl 99 -c 32768 -fa on --jinja
```
