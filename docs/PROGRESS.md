# Threnody — Progress Log

## 2026-11-02
- Created `G:\Threnody`; cloned `Fable-Coder-V4` and `OmniAgent-Bench` into `refs/`.
- V4 post-mortem written → `docs/01_V4_ANALYSIS.md`. Key: V4 ≈ base; benchmark contamination; LR 10× too low; single-turn only; no loss masking.
- Deep research (4 tracks + synthesis) → `reports/Agentic coding LLM training research.md`, notes in `research_notes/`.
- **Name chosen: Threnody** (base `Qwen/Qwen3.5-9B`). Plan → `docs/00_PLAN.md`.
- Verified Qwen3.5-9B config: hybrid Gated-DeltaNet/full-attention (1 in 4 full), 262K ctx, multimodal wrapper `Qwen3_5ForConditionalGeneration`, MTP head, chat template with interleaved `<think>` and XML `<tool_call><function=..>` format.
- Wrote `pipeline/decontam.py` (13-gram + entry-point + literal filter vs OmniAgent-Bench).
- User approved: public GitHub repo `YoMosa2009/Threnody`; Google Drive for checkpoints. All compute on Colab A100.
- Built + locally tested pipeline (`pipeline/`): data builder (11 sources, native Qwen3.5 tool format, interleaved think), assistant-only masking (verified token-level), chunked-CE LoRA trainer, merge sweep, GGUF/imatrix/KLD export, Colab bench runner.
- Fixed SWE-smith conversion bug: some rows embed `<function=...>` actions inline with observations as user turns → now parsed into real tool calls / tool results.
- Published https://github.com/YoMosa2009/Threnody (public). Notebook: `notebooks/Threnody.ipynb`.
- Colab: runtime = **A100-SXM4-80GB High-RAM**; Drive mounted at `MyDrive/Threnody`. Started cells 2–5 (deps, llama.cpp build, baseline bench, dataset build).
- Scope confirmed by user: **Qwen3.5-9B only** (no larger variants).

### Baseline — OmniAgent-Bench on Colab A100 (llama.cpp CUDA, Q4_K_M, temp 0, q8_0 KV, 32k ctx)
| Model | Pass | Weighted | Coding | Tools | Sec | Long | IF |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct (V4 base) | 12/16 | 73.03% | 13/27 | 14/14 | 8.6/14 | 10/10 | 7/7 |
| Fable-Coder-V4 | 12/16 | 72.69% | 13/27 | 14/14 | 9.8/14 | 10/10 | 5.5/7 |
| **Qwen3.5-9B (Threnody base), non-thinking** | **15/16** | **92.71%** | 23/27 | 14/14 | 12.75/14 | 10/10 | 7/7 |

- V4 reproduces its published 72.69% exactly → harness is faithful. On identical hardware V4 is now *below* its own base (73.03%).
- Qwen3.5-9B already +20 pts over V4 before training. Only CODE-05 (email regex) and SEC-01 partial credit remain → **OmniAgent-Bench is saturated** for this model class. It can show regressions, not improvements.
- Decision: add a harder closed-loop multi-turn agentic eval (`eval/agentloop`) — sandboxed repos, native tools, hidden pytest grading.

### Course correction (user directive): teachers must be June-2026+ models
- Rejected: datasets whose trajectories come from 2025-era teachers (e.g. SWE-smith trajectories = Claude 3.7 Sonnet). Full SFT is on hold until the mix is rebuilt.
- New direction: (1) audit every source's teacher/date; (2) generate our own verified agent trajectories with a 2026 open-weight teacher (leading candidate `Qwen/Qwen3.8-27B`, 2026-08-14, Apache-2.0, same tokenizer family) on the A100; (3) on-policy distillation from that teacher; (4) refresh techniques from 2026 papers.
- Closed-model outputs (Claude/GPT) are excluded: their terms forbid training competing models on outputs.
- Pilot still runs (validates trainer + measures throughput/memory; data-independent).
- Research round 2 launched → `research_notes/Threnody 2026 frontier techniques/`.

### 2026-11-03 — 2026-teacher pipeline launched
- Research round 2 (`research_notes/Threnody 2026 frontier techniques/`): all 9 v1 sources use 2025-era teachers → replaced. Teacher chosen: `Qwen/Qwen3.8-27B-FP8` (tokenizer identical to Qwen3.5-9B, verified by hash). Main public source: `nvidia/Open-SWE-Traces` (Qwen3.8-27B, 5 real harnesses, resolved flag).
- AgentLoop baseline: V4 0/14, Qwen3.5-9B nothink 12/14, think 14/14.
- Trainer bugs fixed: torchao conflict, transformers-5 `warmup_ratio` removal; validated end-to-end on a tiny Qwen3.5 on CPU.
- Open-SWE-Traces trajectories are long (only ~6% ≤16k) → training context raised to 32k.
- Launched unattended `pipeline/run_all.sh` on the A100. See `docs/STATUS.md`.
