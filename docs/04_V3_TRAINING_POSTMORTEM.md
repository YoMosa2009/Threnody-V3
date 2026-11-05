# Threnody 3.0 — Comprehensive Training Postmortem & Engineering Roadmap

_Author: Axiom Inference Engineering Team_  
_Model: Threnody 3.0 (`Threnody-9B V3`)_  
_Base: Qwen/Qwen3.5-9B (Dense Transformer)_  
_Release Date: November 5, 2026_  
_Artifacts: [Hugging Face Repository](https://huggingface.co/MalxTech/Threnody-9B-V3-GGUF) | [GitHub Repository](https://github.com/YoMosa2009/Threnody)_

> **Revision note (2026-11-05):** This postmortem has been revised to narrow claims to the measurements. Scores are unchanged. All benchmarks are author-reported. Records that conflict with each other or with the code are listed in [`KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md). This document does not say which conflicting record is correct.

---

## 1. Executive Summary

Threnody 3.0 is an experimental agentic-coding fine-tune of Qwen3.5-9B, trained with supervised fine-tuning using LoRA and merged at adapter strength 0.75. It targeted the failure modes seen in V2. Measured outcomes against base Qwen3.5-9B (author-reported, non-thinking mode, Q8_0):
1. **Action repetition**: training trajectories with `max_repeat >= 2` were filtered out. Measured `max_repeat` (longest run of identical consecutive actions, across all tasks in a suite) is **1** on `opencode` held-out, **1** on AgentLoop In-Harness (the Hugging Face card says 27, unresolved), and **2** on AgentLoop-Hard.
2. **Domain retention**: on a 400-question MMLU-Pro biology + chemistry sample, V3 scored 78.5% vs. Base 76.0% (**+2.5 pp**). Chemistry was 73.5% vs. 69.4% (**+4.1 pp**); biology was 86.5% for both. The V2 figures of 14.1% / 37.6% previously quoted here come from a different test (BioChem-Hard) and are not directly comparable ([discrepancies §7](KNOWN_DISCREPANCIES.md#7-science-score-records)).
3. **Held-out `opencode` schema**: 13/14 vs. Base 12/14 (**+7.1 pp**, one task) and V2 10/14. The same suite fed the checkpoint and candidate selection score.
4. **Regressions vs. Base**: OmniAgent-Bench **−8.1 pp** (13/16 vs. 15/16), AgentLoop In-Harness **−7.1 pp** (10/14 vs. 11/14), AgentLoop-Hard **−6.7 pp** (17/30 vs. 19/30).
5. **Refusal abliteration**: directional orthogonalization on layers 12–28, aimed at false refusals on security and shell tooling. No quantitative refusal evaluation is published.
6. **Quantization ladder**: 6 GGUF quants plus an F16 vision projector. Quant metrics measure token-distribution agreement with BF16, not task performance (§6).

---

## 2. End-to-End Pipeline Execution Log

The entire pipeline executed autonomously on an **NVIDIA A100-SXM4-80GB High-RAM** instance in Google Colab across 7 distinct stages:

```
[01:11 UTC] Stage 1: Data Ingestion & Zero-Tolerance Anti-Loop Filtering (v3_data.done)
            - 8 teacher datasets ingested
            - 2,895,493 tokens compiled across 662 multi-turn trajectories
            - 150 identity pairs for Axiom Inference / Threnody 3.0 injected
[01:28 UTC] Stage 2: Base Model Reference Evaluation (v3_base.done)
            - Baseline established across OmniAgent-Bench, AgentLoop, and BioChem
[02:40 UTC] Stage 3: Supervised Fine-Tuning (LoRA) (v3_sft.done)
            - 56 optimizer steps (1.219 epochs) with Chunked Cross-Entropy (reported; see note below)
            - Train loss: 1.4820 -> 0.5541 | Eval loss: 0.6126
[03:26 UTC] Stage 4/5: Release Candidate Evaluation & Automated Judge (v3_select.done)
            - Candidates s0.75 and s1.0 evaluated in parallel
            - Automated judge selected candidate s0.75 for release
[05:12 UTC] Stage 6: Abliteration, Imatrix Calibration, and Quantization (v3_export.done)
            - Layer 12-28 refusal orthogonalization on merged weights
            - GPU importance matrix calculated on val.jsonl
            - Quantized Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, IQ3_M + mmproj-F16
[05:44 UTC] Stage 7: Hugging Face Release Streaming (v3_upload.done)
            - Streamed ~37 GB via HfApi to MalxTech/Threnody-9B-V3-GGUF
[05:50 UTC] Pipeline completed successfully.
```

> **Note on the training record:** the 2,895,493 tokens above are what was reported for the run. The **35M-token** figure elsewhere is the configuration target (`run_v3.sh --total_tokens 35e6`). `results/v3_training_report.json` reports per-device batch size 4 × gradient accumulation 8. `pipeline/train_sft.py` hard-codes batch size 1, and `run_v3.sh` leaves accumulation at its default of 16. Warmup, weight decay and optimizer also differ. Without run logs, the settings that actually ran are unverified ([discrepancies §3](KNOWN_DISCREPANCIES.md#3-training-records)).
>
> **Note on the "automated judge":** both published gate records mark `s0.75` as passing. Neither gate function in `pipeline/gates.py` would pass it, and no revision in this repository contains the published gate criteria. `run_v3.sh` falls back to the highest `fast_score` when no candidate passes ([discrepancies §4](KNOWN_DISCREPANCIES.md#4-gate-records-vs-gate-code)).

---

## 3. Detailed Benchmark Findings & Comparative Scorecard

Conditions per `run_v3.sh`: llama.cpp CUDA runtime, Q8_0 build, pinned chat template, non-thinking mode, temperature 0. Author-reported; single run. Differences are in percentage points (pp), from pass counts.

| Benchmark Suite | Qwen3.5-9B Base | Threnody-9B V2 (`s0.3`) | **Threnody 3.0 (`s0.75`)** | Delta vs Base |
|---|---|---|---|---|
| **AgentLoop Held-Out (`opencode`, 14 tasks)** | 12/14 (85.7%) | 10/14 (71.4%) | **13/14 (92.9%)** | **+7.1 pp** |
| ↳ `max_repeat` (all tasks in suite) | 1 | 2 | **1** | 0 |
| **MMLU-Pro biology + chemistry (400-question subset)** | 76.0% | 76.5% | **78.5%** | **+2.5 pp** |
| ↳ Biology | 86.5% | not in current record | **86.5%** | 0.0 pp |
| ↳ Chemistry | 69.4% | not in current record | **73.5%** | **+4.1 pp** |
| **OmniAgent-Bench (Weighted)** | 92.7% | 92.7% | **84.6%** | **−8.1 pp** |
| **AgentLoop In-Harness (14 tasks)** | 11/14 (78.6%) | 11/14 (78.6%) | **10/14 (71.4%)** | **−7.1 pp** |
| **AgentLoop-Hard (30 held-out tasks)** | 19/30 (63.3%) | N/A | **17/30 (56.7%)**, `max_repeat` 2 | **−6.7 pp** |

### Key Benchmark Takeaways:
- **Held-out schema transfer**: 13/14 on the held-out `opencode` schema (one task more than Base) suggests transfer to that tested schema. Fourteen tasks on one schema cannot establish general tool understanding. The suite also contributed to checkpoint and candidate selection, so the score is optimistically biased.
- **Bio/chem subset retained**: unlike V2, V3 did not lose bio/chem accuracy on the MMLU-Pro sample. Chemistry was +4.1 pp over Base and biology unchanged. This covers biology and chemistry questions only, not overall MMLU-Pro.
- **Agent-suite regressions**: V3 is below Base on three of four agent suites (OmniAgent-Bench, AgentLoop In-Harness, AgentLoop-Hard).

---

## 4. Candidate Ablation: Why `s0.75` Won Over `s1.0`

In V2, the adapter had to be diluted down to `s0.3` because `s1.0` was catastrophically unstable. In V3, our architectural goal was to push adapter strength as high as possible.

We benchmarked `s0.75` against `s1.0`:
- **`s1.0` Bottleneck**: At full 1.0 strength, the model exhibited uncalibrated steering oscillations when handling complex nested bash/python commands. On task `syn500025_notif_pipeline_perf`, `s1.0` spent 26.5 minutes before timing out, generating repetitive reasoning steps.
- **`s0.75`**: Scaling the adapter update to 75% (`merge.py` interpolates toward the base) gave 13/14 on `opencode` with suite-level `max_repeat = 1`, against `s1.0`'s recorded 11/14 with `max_repeat = 4` (`v3_gate_s1.0.json`). How much of the trained behaviour survives scaling was not measured.

**Verdict**: `s0.75` was written to `eval_v3/chosen.txt` as the release candidate. Per `run_v3.sh`, candidates are ranked by gate pass first, then `fast_score`. The published `s0.75` metrics fail both gate functions in `gates.py`, so selection by `fast_score` (33.99 vs. 32.39 for `s1.0`) is the likely path. That is an inference from the code; no log confirms it. Note that commit `52edc9f` earlier published an all-passing **`s1.0`** result set ([discrepancies §4.1](KNOWN_DISCREPANCIES.md#41-an-earlier-superseded-v3-result-set)).

---

## 5. Architectural Innovations & Solutions

### 5.1 Zero-Tolerance Anti-Loop Trajectory Filter
In V2, trajectories were ranked by "recovery from loops", which accidentally reinforced loop behavior because the training loss penalized the recovery steps equally with the loop steps.
In V3, any trajectory with `max_repeat >= 2` was dropped unconditionally. Only trajectories in which no action was immediately repeated (`max_repeat` = 1, consecutive identical actions) were admitted. Non-consecutive repeats are not caught by this metric.

### 5.2 Multimodal Decoupled Projector Architecture
The vision projector was kept at F16 and not quantized with the text weights. No vision evaluation (quantized or not) was run, so the effect of quantizing it is unmeasured here.
Layout:
- Text GGUF Ladder (`Threnody-9B-V3-*.gguf`): Quantized using multi-turn calibration matrices.
- Multimodal Vision Projector (`mmproj-Threnody-9B-V3-F16.gguf`): stored at F16.

### 5.3 Directional Refusal Orthogonalization (Arditi et al., 2024)
Standard instruction fine-tunes frequently introduce false refusals when coding agents attempt to inspect `/etc/passwd`, run `nmap`, or manipulate memory offsets.
We applied abliteration across layers 12 through 28:
$$\mathbf{W}_{\text{abliterated}} = \mathbf{W} - (\mathbf{W} \cdot \hat{\mathbf{v}}_{\text{refusal}}) \otimes \hat{\mathbf{v}}_{\text{refusal}}$$
The goal was to reduce false refusals on developer commands. No quantitative evaluation of refusal rates or of remaining safety behaviour is published. `results/v3_redteam_report.json` has only free-text status labels, and its format does not match `eval/redteam/harness.py` output ([discrepancies §10](KNOWN_DISCREPANCIES.md#10-red-team-report)).

---

## 6. Quantization Fidelity Ladder (KLD Analysis)

Each quant was compared with the BF16 GGUF of the same merged model (`llama-perplexity --kl-divergence`, 40 × 4096-token chunks of `val.jsonl`, the same text used for imatrix calibration). "Same top token" is the percentage of positions where the quant's most likely token matches BF16's. **These are token-distribution metrics; they do not show equal coding or agent performance, and no benchmark was run per quant.**

| Quantization Format | Raw Bytes | Size (GiB) | Same Top Token | Median KLD | Mean $\Delta p$ |
|---|---|---|---|---|---|
| **Q8_0** | 9,527,502,016 | 8.87 GB | **97.756%** | 0.000088 | $+0.009\%$ |
| **Q6_K** | 7,359,259,840 | 6.85 GB | **96.664%** | 0.000246 | $+0.160\%$ |
| **Q5_K_M** | 6,467,970,240 | 6.02 GB | **96.127%** | 0.000426 | $+0.021\%$ |
| **Q4_K_M** | 5,629,109,440 | 5.24 GB | **94.058%** | 0.001219 | $+0.503\%$ |
| **IQ4_XS** | 5,196,440,768 | 4.84 GB | **93.161%** | 0.001749 | $-0.148\%$ |
| **IQ3_M** | 4,415,382,720 | 4.20 GB | **89.615%** | 0.007465 | $-1.787\%$ |

---

## 7. Actionable Roadmap for Threnody 4.0 (V4)

To address V3's regressions and its reliance on a diluted adapter (`s0.75`), future work should consider the following:

### Recommendation 1: Multi-Turn Direct Preference Optimization (DPO / GRPO)
- **Problem**: SFT alone cannot completely prevent steering oscillations at `s1.0`.
- **Solution**: Collect paired trajectories where Candidate A solves a task in 4 steps with no repeats and Candidate B solves it in 12 steps with oscillatory backtracking. Apply offline DPO or online GRPO directly on trajectory pairs to explicitly penalize oscillatory token distributions.

### Recommendation 2: Synthetic Error Injection & Explicit Backtracking Curriculum
- **Problem**: When a tool produces an unexpected runtime error (e.g., `404 Not Found` or `SyntaxError`), models can occasionally pause or attempt duplicate actions.
- **Solution**: Inject synthetic mock errors into 15% of training trajectories, followed by an explicit `<backtrack>` or `<alternative_hypothesis>` reasoning trace demonstrating immediate tool pivoting.

### Recommendation 3: In-Training Refusal Loss Suppression
- **Problem**: Post-hoc weight orthogonalization (abliteration) alters the weight geometry after fine-tuning.
- **Solution**: Incorporate an auxiliary loss penalty during SFT on harmless dual-use prompts, actively penalizing positive dot products with refusal token logits during backpropagation.

### Recommendation 4: Dynamic Multi-Token Prediction (MTP) Training
- **Problem**: Inference throughput on 9B models in long multi-turn contexts (20k+ tokens) can be latency-constrained.
- **Solution**: Train the native Qwen3.5 MTP head simultaneously during SFT to allow speculative decoding of 2 tokens per forward pass in `llama.cpp`.

### Recommendation 5: Expansion of Held-Out Evaluation Suites
- **Problem**: The held-out `opencode` suite (14 tasks) is small (one task = 7.1 pp), near ceiling (13/14), and was also used in model selection.
- **Solution**: Expand the held-out evaluation suite to at least 100 diverse, automated SWE benchmarks (covering Rust, Go, TypeScript, and Docker container orchestration). Keep a test split that is never used for checkpoint or candidate selection, and publish per-task outputs and run logs so results can be checked.
