# Threnody — Current Status & Handoff

_Last updated: 2026-11-03 19:16 (UTC)_

## Session 4: SFT DONE 00:09 (11-04), eval (merge + benchmarks) RUNNING on G4
- Colab account = project account, opened with `?authuser=1` (Chrome's default account is a different one with an empty Drive).
- SFT 19:08 -> 00:09: 232 steps, train_loss 0.547, final eval_loss 0.567. Adapter in `sft_v2/final_adapter`; checkpoints 120-220 adapters in `sft_v2_keep/`.
- Eval stage running: results -> `bench_threnody/summary.json`.
- After benchmarks: post-SFT polish list in docs/00_PLAN.md (merge sweep 0.85/0.5, checkpoint averaging, TIES/DARE, imatrix ladder, DPO, MTP speculative decoding).
- Keep the Chrome tab open. Watch `heartbeat.txt` / `ALERT.txt`; LOG TAIL cell for sft progress.

## Session 3 (15:50 UTC): teacher data done, SFT relaunching on G4 (RTX PRO 6000 96 GB)
- SFT crash (fp32 vs bf16 in chunked CE) fixed; runtime was reclaimed; A100 unavailable so moved to G4. data_final = 3,707 records, 52.4M tokens.

## Session 2 (07:30 UTC): env repaired, teacher running
- Training crash root cause: stray system vLLM install -> torch 2.13 cu132 vs old torchaudio. Fixed (torchaudio removed), imports verified.
- Teacher Qwen3.8-27B-FP8 served by system vLLM since 07:13; then data_teacher -> data_final -> SFT (1 epoch, 9 h cap) -> eval (merge 1.0/0.75).
- Check progress: run the STATUS cell, then read `MyDrive/Threnody/status.txt`.

## Session 1 outcome (2026-11-03 05:21 UTC)
- Teacher stage was SKIPPED: Colab's Python lacks `ensurepip`, so `python -m venv` failed and vLLM never installed
  (fixed in repo: now uses `virtualenv`). `markers/teacher_failed` exists; `teacher_all.done` does not.
- The pipeline fell back as designed and is running data_v2 -> SFT -> eval on the public June-2026+ data only
  (Open-SWE-Traces Qwen3.8-27B + Qwen3.8 terminal traces).
- Next session: `git pull` in Colab, delete markers/teacher_failed, data_v2.done, sft.done, eval.done, then re-run
  the RUN-ALL cell. Teacher stages run, then SFT retrains with teacher data (or continue from sft_v2 + on-policy distillation).

## Running now (Colab A100-80GB, unattended)
`pipeline/run_all.sh`, launched from the notebook, logs to `MyDrive/Threnody/logs/`. Each stage writes
`MyDrive/Threnody/markers/<stage>.done` and is skipped on re-run, so to resume after a disconnect just
re-run the RUN-ALL cell.

| Stage | What | Cap | Output |
|---|---|---|---|
| teacher server | vLLM 0.32.0 serving `Qwen/Qwen3.8-27B-FP8` (2026-08, Apache-2.0) in `/content/vllm-env` | 30 min to start | `logs/vllm.log` |
| synth_tasks | teacher designs repo tasks (files + hidden tests + reference); kept only if they fail as shipped and the reference passes; eval tasks n-gram excluded | 75 min | `teacher/tasks_train.jsonl` |
| solve_tasks | teacher solves them in the AgentLoop harness (k=2); kept only if hidden tests pass AND it ran python/pytest after its last edit | 150 min | `teacher/agent_traj.jsonl` |
| code | teacher solves `deepmind/code_contests` (cf_rating ≥ 1400); kept only if ≤25 official tests pass | 60 min | `teacher/code_verified.jsonl` |
| regen | fresh teacher answers for security / tool-judgment / instruction-following prompts | 40 min | `teacher/regen.jsonl` |
| data_v2 | mix of teacher outputs + `nvidia/Open-SWE-Traces` (Qwen3.8-27B teacher, 5 harnesses, resolved only) + `ukisai/Qwen3.8-27B-multi-turn-agent-sft`; ≤32k tokens; decontaminated | — | `data_v2/` |
| sft | bf16 LoRA r=64/α=32 on all 248 LM linear layers, LR 1.5e-4, 1.5 epochs, 32k ctx, assistant-only loss | 9 h | `sft_v2/final_adapter` |
| eval | merge at 1.0/0.75/0.5 → Q4_K_M GGUF → OmniAgent-Bench + AgentLoop, think & nothink | — | `bench_threnody/summary.json` |

If the teacher server fails to start, the script logs it and continues: SFT then uses only the public 2026-teacher data.

## Baselines (same A100, Q4_K_M, temp 0)
| Model | OmniAgent-Bench | AgentLoop (14) |
|---|---|---|
| Qwen2.5-Coder-7B (V4 base) | 73.0% | 0/14 |
| Fable-Coder-V4 | 72.7% | 0/14 (no native tool calls) |
| Qwen3.5-9B nothink | 92.7% | 12/14 |
| Qwen3.5-9B think | 84.8% | 14/14 |

Both suites are near saturation for the 9B base, so a harder eval is still needed (next steps).

## Next steps (after the pipeline finishes)
1. Read `bench_threnody/summary.json` and pick the best merge strength; must not regress vs base on either suite.
2. **Harder eval**: add a held-out AgentLoop-Hard set and a recent LiveCodeBench subset; run base vs Threnody.
3. **On-policy distillation** from Qwen3.8-27B (research: `research_notes/Threnody 2026 frontier techniques/`), phased: student vLLM sampling → teacher prompt-logprobs → LoRA update; STRIDE-style early stop.
4. Final export: `pipeline/export_gguf.py` (imatrix ladder Q8_0…IQ3_M + KLD report), HF upload, README results.

## Credit budget
About 25 CU used so far (A100-80GB ≈ 7.5 CU/h). The pipeline should use about 120 CU (~16 h).
