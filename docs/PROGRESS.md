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

### 2026-11-03 07:30 UTC — recovery
- data_v2 built 06:19: 2,980 records / 51.4M tokens (Open-SWE-Traces Qwen3.8-27B ×5 harnesses, avg ~23k tok; Qwen3.8 terminal 900).
- SFT crashed at start: the first (uv) vLLM attempt had installed vLLM 0.32 into system Python → torch 2.13+cu132, stale torchaudio (cu130) broke `import transformers/peft`. A100 sat idle ~1 h (~15 CU lost).
- Fix: uninstall torchaudio; use the system vLLM for the teacher; append teacher data to data_v2 (no re-stream); 1 epoch; merge sweep 1.0/0.75. Imports verified OK (torch 2.13, transformers 5.19, peft 0.21, fla).
- 07:13 teacher (Qwen3.8-27B-FP8) serving; synth_tasks keeping ~60% of designed tasks (24 by 07:28).
- Monitoring method: a notebook cell writes `MyDrive/Threnody/status.txt`, read via the Drive API (notebook output panes are not scrollable reliably).

### 2026-11-03 15:50 UTC — SFT crash #2, runtime lost, moved to G4
- Teacher stages done 09:19: synth 08:13 (tasks_train), solve_tasks kept 73 verified trajectories (105 passed-but-unverified rejected, 42 failed), regen 730 answers. `code` mode crashed (pyarrow "offset overflow" in code_contests shuffle buffer) -> 0 records; to fix if the teacher is re-run.
- data_final: 3,707 train / 75 val, 52.4M tokens (teacher_agent 71, teacher_regen 731 + data_v2).
- SFT crashed on step 1 at 09:26: `expected scalar type Float but found BFloat16` in chunked CE (final hidden state fp32 vs bf16 lm_head, no autocast). Fixed: cast h to lm_head dtype; eval is loss-only.
- Colab reclaimed the idle VM. A100 unavailable on reconnect (fell back to L4 24 GB) -> switched to **G4 = RTX PRO 6000 Blackwell 96 GB** (sm_120). run_all now rebuilds llama.cpp for non-sm_80 GPUs before eval.
- Lesson: watch the first SFT steps after every launch; a crash leaves the GPU idle.
