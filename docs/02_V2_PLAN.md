# Threnody V2 — Plan

_Written 2026-11-04 after V1 SFT regressed vs base. Goal unchanged: **strictly better than Qwen3.5-9B** on every gate below, or nothing ships._

## Why V1 failed (facts)
- Every V1 config scored below base: nothink OmniAgent 81.7% (s1.0) / 83.5% (s0.75) vs 92.7%; AgentLoop 5/14 / 8/14 vs 12/14.
- s0.75 > s1.0 under an identical build ⇒ the training delta itself was harmful.
- Dominant failure: identical failing tool call repeated until the 30-turn cap (learned behaviour; base doesn't do it).
- ~85% of tokens were Open-SWE-Traces from 5 foreign harnesses with their own tool habits; only 71 in-harness verified trajectories.
- Every training turn had reasoning; nothink inference gets an empty think block (mode mismatch).
- Eval flaw: base numbers came from unsloth's GGUF (imatrix quant, different template) vs our plain Q4_K_M + HF template. No acceptance gate existed.

## 1. Same-build evaluation (fixes the eval flaw)
- Every model compared (base and every Threnody candidate) is built by **one** pipeline: our convert → imatrix Q4_K_M → **one** pinned chat template (unsloth's fixed Qwen3.5 template, explicit `enable_thinking` in both modes).
- The base reference is re-measured with that build (control run `STRENGTHS=0.0`, 2026-11-04). Its scores are the bar.

## 2. Data, rebuilt around verified in-harness behaviour
| Slice | Share (tokens) | Source | Filter |
|---|---|---|---|
| Multi-harness verified agent trajectories | ~40% | Rejection sampling from Qwen3.8-27B teacher **and** the Qwen3.5-9B base on synthesized repo tasks, run in **6+ harness tool schemas** (see §3) | hidden tests pass AND tests run after last edit; no repeated identical action |
| Open-SWE-Traces (Qwen3.8-27B) | ≤30% | existing | drop trajectories with repeated identical actions or ≥3 consecutive identical errors; resolved only |
| Terminal / CLI | ~10% | Qwen3.8 terminal traces + teacher | same repeat filter |
| Biology + chemistry | ~8% | Teacher answers to bio/chem questions with checkable answers (MCQ / numeric / reaction balancing), kept only when correct | **hazard filter**: no pathogen enhancement, toxin/weapon synthesis, or controlled-substance routes |
| Tool judgment / security / IF / general knowledge replay | ~12% | existing regen 731 + new | — |
- **~30% of examples rendered in nothink format** (empty think block), the rest with reasoning.
- Teacher rule unchanged: June-2026+ open-weight only (Qwen/Qwen3.8-27B-FP8). No 2025-era data, no closed-model outputs.
- Fix before teacher runs: `gen_teacher.py --mode code` pyarrow overflow (done in c73e610).

## 3. Harness generalization (no overfitting to one harness)
- Training trajectories are generated in **many tool schemas**, each with its own system prompt and tool names/arguments:
  our AgentLoop (read_file/write_file/bash), OpenHands-style (str_replace_editor + execute_bash), SWE-agent-style, mini-swe-agent (bash only), Aider-style (search/replace edit blocks), Claude-Code/OpenCode-style (Read/Edit/Write/Bash/Grep/Glob), plain OpenAI function-calling.
- Tool names and argument names are randomized per sample in ~20% of cases so the model reads the schema instead of memorizing it.
- **Held-out harness eval**: at least 2 schemas never seen in training are used only for evaluation (AgentLoop tasks re-skinned).

## 3b. Cleverness and determination (user goal: a real leap over the base, not just "no regression")
These are selection pressures in the data, measured by the hard gates, not prompt tricks:
- **Harder task families** for synthesis: `insight` (the obvious fix breaks a hidden invariant), `algorithmic` (naive
  solution too slow; hidden large-input tests), `persistence` (misleading first symptom; first plausible fix fails).
- **Determination**: per task, trajectories that hit a failing test/traceback and still reached a verified pass
  (`recovered`) are preferred over ones that passed first try. Trajectories that stop without verifying are dropped.
- **Cleverness/efficiency**: among passing runs, fewer turns rank higher; at most 3 per task, preferring different harnesses.
- **Algorithmic reasoning**: ~7% verified competitive programming (code_contests, all tests pass; offset-overflow bug fixed).
- **On-policy**: student (base 9B) successes are kept alongside teacher ones, so the model learns from its own
  distribution, not only an imitation of a 27B model.
- Measured by AgentLoop-Hard (strictly better than base required) and the held-out harnesses.

## 4. Training
- bf16 LoRA r=64/α=32, **LR 1e-4** (was 1.5e-4), 1 epoch, 32k ctx, assistant-only loss, checkpoint every 20 steps, `keep_ckpt.sh` preserves all adapters.
- Install `causal_conv1d` matching wheel before the run (DeltaNet conv speed-up).

## 5. Hard gates (automatic, nothing ships without passing all)
Against the same-build base, both think and nothink:
1. OmniAgent-Bench ≥ base.
2. AgentLoop (in-harness) ≥ base.
3. AgentLoop held-out harnesses ≥ base.
4. AgentLoop-Hard / recent LiveCodeBench subset > base (the actual "better" claim).
5. Bio/chem: GPQA-Diamond bio+chem and MMLU-Pro bio+chem ≥ base.
6. Repetition check: no AgentLoop trajectory with ≥5 identical consecutive tool calls.
- **Gate during training**: a fast eval (AgentLoop nothink + 1 held-out harness + 200-question bio/chem subset) on every 40-step checkpoint; first regression vs the previous best stops training and keeps the best checkpoint.
- Final candidate = best of {checkpoint averaging, merge-strength sweep 1.0/0.85/0.75/0.5, TIES/DARE} that passes all gates.

## 6. Automation + alerts
- `pipeline/run_v2.sh`: one resumable orchestrator (markers per stage) — teacher serve → task synth → multi-harness rejection sampling (teacher + base) → bio/chem gen → filters/mix → SFT with checkpoint gate → merge candidates → same-build GGUF → full gates → export (imatrix ladder, KLD, mmproj) → report.
- Any stage failure exits non-zero and never writes its marker (V1's eval stage marked done after failing; fixed in 977380c for eval, applied to all stages in V2).
- `watchdog.sh` writes `heartbeat.txt` every 5 min and `ALERT.txt` on: traceback, OOM, idle GPU > 10 min, silent log 45 min, stage failure, **gate failure**, pipeline finished.
- Claude checks heartbeat/ALERT every 30 min and posts a status every 1.5 h; emails the user on stage failure, gate failure, Drive re-auth, or completion.
- Optional instant push channel (e.g. a private ntfy topic) needs the user's OK before use.

## 7. Release (only after all gates pass)
imatrix quant ladder Q8_0…IQ3_M + KLD report, optional vision `mmproj`, MTP speculative-decoding notes, HF upload, README with same-build base vs Threnody tables.

## 8. Implementation (2026-11-04)
| Piece | File |
|---|---|
| 8 tool schemas (6 train, 2 held-out), renamed variants, verified/repetition checks | `eval/agentloop/schemas.py` |
| Harness runs any schema; `--schema`, `--tasks_file`, `--max_tasks` | `eval/agentloop/harness.py` |
| Generation: hard eval split, task synth, multi-schema rejection sampling (teacher+student), bio/chem | `pipeline/gen_v2.py` |
| Mix by token share, decontam, repeat filter, nothink share, determination ranking | `pipeline/build_v2.py` |
| Same-build GGUF, eval suites, gates, candidates, ranking, template pinning | `pipeline/gates.py` |
| Checkpoint gate beside training (CPU merge, Q8_0, fast suite, STOP file) | `pipeline/ckpt_gate.py` |
| MMLU-Pro bio+chem eval | `pipeline/sci_eval.py` |
| Merge: strength, CPU, TIES trim, checkpoint averaging | `pipeline/merge.py` |
| Orchestrator (resumable, fail-fast, `DRY=1` mode) | `pipeline/run_v2.sh` |
| Watchdog: follows V2 logs, alerts on stage/gate failure, ntfy push | `pipeline/watchdog.sh` |
| Pinned chat template (unsloth Qwen3.5, explicit enable_thinking) | `pipeline/templates/qwen35_chat_template.jinja` |
Local CPU tests passed: all 8 schemas solve a task with scripted actions (minisweagent's heredoc only fails on
Windows), mixer end-to-end (decontam rejects eval-derived data, caps, nothink), gate/rank/candidate logic.

## 9. Budget (G4 ~6 CU/h, 57 CU left on 2026-11-04)
Generation ~2.5 h, base references + data ~0.5 h, SFT ~2.5-3.5 h (gates and base full eval overlap on the same GPU),
selection ~1.5 h, export ~0.5 h: **~7.5-8 h, ~45-48 CU**. Optional dry run (`DRY=1`, tiny slice of every stage): ~1 h, ~6 CU.
