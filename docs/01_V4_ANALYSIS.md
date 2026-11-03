# 01 — Fable-Coder-V4 Post-Mortem (input to Threnody)

Source: `refs/Fable-Coder-V4` (cloned 2026-11-02), `refs/OmniAgent-Bench`.

## What V4 is
- Base: `Qwen/Qwen2.5-Coder-7B-Instruct` (2024 model).
- QLoRA NF4, r=32, alpha=64, all 7 projections, **LR 2e-5**, 350 steps × eff. batch 8 (~2,800 samples seen, <1 epoch), max_len 4096.
- Data: 3,500 single-turn samples — 2,000 CodeFeedback, 1,200 Hermes-FC (collapsed to last user/assistant turn), 1,000 agentic_red_team, 400 "precision templates" (8 prompts × 50 repeats).
- Shipped: Q4_K_M GGUF, 4.36 GiB, 4k context in Modelfile, system prompt still "You are Qwen".

## Results
| | OmniAgent-Bench (16 tasks) | Closed-loop sandbox (4 tasks) |
|---|---|---|
| Qwen2.5-Coder-7B base | 12/16, 70.7% weighted | 1/4 |
| Fable-Coder V4 | 12/16, 72.7% weighted | 1/4 |

The +2.0 pt delta is one partial-credit judge task — within noise on a 16-task suite. **V4 is effectively the base model.**

## Root causes found in the code
1. **Benchmark contamination.** The "precision templates" in `scripts/1_curate_v4.py` are near-verbatim copies of OmniAgent-Bench tasks (CODE-02 merge_intervals, CODE-03 is_within, CODE-04 add_item, SQL tie query, Bayes, injection SAFE). Repeated 50× each. Any V4 gain on those tasks is not real capability. Threnody must use a decontamination filter against `tasks.json`.
2. **Broken template strings.** Those templates use `"\\n"` inside normal Python strings → the model was trained to emit literal backslash-n characters in code. A small but real poisoning.
3. **Wrong reference answer.** The `is_within` template uses `Path.resolve()`, which touches the filesystem — the exact thing the task forbids — and V4 still fails CODE-03.
4. **LR too low for LoRA.** 2e-5 is a full-fine-tune LR; LoRA typically needs ~1e-4–2e-4 (≈10× full-FT LR). Combined with <1 epoch, weights barely moved (consistent with the ~0 delta).
5. **No assistant-only loss masking.** Loss computed on system+user tokens too; the long tool-schema system prompt dominates the gradient for agentic samples.
6. **Single-turn only.** Hermes multi-turn conversations were flattened to one user/assistant pair; no tool-result → fix → retest loops. This is precisely why the closed-loop sandbox failed (traceback → repeated edit loops).
7. **Identity/system prompt left as Qwen**, 4k context in the Modelfile — far too short for long-horizon CLI sessions.

## Benchmark observations (OmniAgent-Bench)
- 16 tasks; 6 coding (unit tests), 3 tool-schema, 3 sec + 2 long-horizon (heuristic keyword judge unless an LLM judge is configured), 2 constraint.
- Default `--max-tokens 1024`. A thinking model will be truncated → must either run non-thinking mode or raise max-tokens and strip `<think>` blocks before evaluation.
- Heuristic judge is keyword-based → saturates (everyone gets 100% on long_horizon). The coding category is where real differentiation lives (all three models 3/6).
- Small N ⇒ report per-task results and run ≥3 seeds / temperature 0 for fairness.

## Implications for Threnody
- Change the base to a 2025–26 open model with native agentic training (see research report).
- Train on **multi-turn trajectories with execution feedback**, assistant-only loss.
- Proper LoRA LR, enough steps, replay to prevent forgetting.
- Strict decontamination against OmniAgent-Bench.
- Ship multiple quant sizes + long-context settings with quantized KV cache.
