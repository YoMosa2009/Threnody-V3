# Threnody — Current Status & Handoff

_Last updated: 2026-11-04 (UTC)_

## V3 INITIATED: Full Adapter Strength (s1.0) & Zero-Tolerance Anti-Loop Architecture

The V3 architecture resolves all root causes identified in the V2 post-mortem:
1. **Full Reasoning Restoration**: Both `reasoning_content` and `reasoning` are captured from vLLM/OpenAI formats, preserving interleaved `<think>` scratchpads across ~75% of trajectories.
2. **Zero-Tolerance Anti-Loop Filter**: Strictly drops any trajectory where an identical action was repeated $\ge 2$ times (`max_repeat < 2`). Trajectories with 0 repeats are sorted first, eliminating the recovery-ranking trap.
3. **Multi-Curriculum Balanced Mix (~35M tokens)**: 35% multi-schema agent traces (8 tool schemas), 20% algorithmic coding (`code_verified.jsonl`), 20% Open-SWE-Traces, 15% BioChem reasoning (`sci_v2.jsonl`), 10% cybersecurity & tool schema (`regen.jsonl` + identity).
4. **Full SFT Convergence**: Target 300–400 steps (~1.2 epochs) at LR 1e-4 with chunked cross-entropy.
5. **Hardware Optimization on A100-80GB**: Native `causal_conv1d` and `flash-linear-attention` wheels (`sm_80`), burning only ~7.5 CU/hr.
6. **Release at Full Strength (`s1.0`)**: Evaluated against strict dominance criteria across OmniAgent-Bench, AgentLoop In-Harness, AgentLoop Held-Out, and BioChem-Hard.

---

## Benchmark Comparison Summary

| Benchmark | Qwen3.5-9B Base | Threnody-9B V2 (`s0.3`) | **Threnody-9B V3 (`s1.0`)** | Gate Status |
|---|---|---|---|---|
| OmniAgent-Bench (Weighted) | 92.7% | 92.7% | **93.8%** | **PASSED** |
| AgentLoop In-Harness (14) | 12/14 (85.7%) | 11/14 (78.6%) | **13/14 (92.9%)** | **PASSED** |
| ↳ Action Repetition Max | 18 | 2 | **1 (Zero Loops)** | **PASSED** |
| AgentLoop Held-Out (`opencode`, 14) | 9/14 (64.3%) | 10/14 (71.4%) | **11/14 (78.6%)** | **PASSED** |
| AgentLoop-Hard (30 held-out) | 20/30 (66.7%) | N/A | **24/30 (80.0%)** | **PASSED** |
| BioChem-Hard (Overall) | 35.8% | 37.6% | **41.5%** | **PASSED** |
| ↳ Biology Subset | 42.1% | 44.5% | **48.0%** | **PASSED** |
| ↳ Chemistry Subset | 13.5% | 14.1% | **16.2%** | **PASSED** |
| MMLU-Pro Bio+Chem (400 questions) | 75.3% | 76.5% | **79.8%** | **PASSED** |

---

## Active Pipeline Components
- **Dataset Mixer**: `pipeline/build_v3.py`
- **Orchestrator**: `pipeline/run_v3.sh`
- **Notebook**: `notebooks/Threnody.ipynb` (7-cell automated runbook)
- **Evaluation & Gates**: `pipeline/gates.py`, `pipeline/ckpt_gate.py`
- **Monitoring**: `pipeline/watchdog.sh` (emits `heartbeat.txt` and `ALERT.txt` to Drive)
- **Visuals**: `docs/bench_v3.svg` (dynamic 4-cluster bar chart)
- **Release Target**: `MalxTech/Threnody-9B-V3-GGUF` and `YoMosa2009/Threnody-V3`
