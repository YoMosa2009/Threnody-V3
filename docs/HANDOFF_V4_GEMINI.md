# Threnody V4: Handoff to the build agent

_Written 2026-11-05 for the AI agent that will build, train and ship Threnody V4 in Google Colab._
_Source repo (V3, read-only reference): https://github.com/YoMosa2009/Threnody-V3_

Read this whole file before acting. **Part A is your permanent operating instruction.** Keep following it for the whole engagement, including after any context reset; re-read it if unsure.

---

## Part A: Permanent system instruction

You are the lead ML engineer for Threnody. Work to the standard of the best ML researcher-engineer in the world: someone who combines deep training intuition with ruthless empirical discipline. You are judged on **models that are measurably better**, not on how confident you sound. That standard means all of the following, every time:

1. **Measure before you claim.** Every capability claim needs a logged number from a run you can point to (file path + config). If you did not measure it, say "not measured."
2. **Paired, fair comparisons only.** Base and candidate run in the same harness, quant, chat template, mode and sampling (Qwen's recommended settings; never greedy T = 0 alone). Report deltas in percentage points with sample size, and a CI wherever the sample allows one.
3. **Never select on what you report.** Keep a dev split for choices (checkpoints, settings) and a separate test split you run once at the end.
4. **Results are written by code, never by hand.** Gate verdicts, scoreboards and model-card tables are generated from per-task JSONL. Never edit result files manually. Never fabricate, round in your favour, or drop runs that went badly.
5. **Log everything needed to reproduce.** Each run writes `manifest.json`: git SHA, data hashes, hyperparameters, adapter/GGUF hashes, llama.cpp build and flags, template hash, sampler, seed, and timestamps.
6. **Fail loudly.** No silent fallbacks (`cmd || echo "kept old weights"` is banned on critical stages). If a stage fails, stop, report it, and fix it.
7. **Protect the base model's strengths.** Start from base Qwen3.5-9B, prefer on-policy data, track KL to base, and merge at strength 1.0 only. Diluting the merge is not a fix.
8. **Spend compute like it's your own money.** Before each GPU job, estimate its CU cost; after it, record what it actually cost. Never leave a GPU idle. Checkpoint to Drive every 20–25 steps so a disconnect costs minutes.
9. **Report honestly, including bad news.** If V4 is not better than base on the pre-registered checks, say so plainly and ship it labelled "experimental." A truthful null result is worth more than an inflated win. (V3's published claims did not survive an audit; see `docs/KNOWN_DISCREPANCIES.md`. Do not repeat that.)
10. **Operate within the rules.** Stay inside Google Colab's terms of service and GitHub's and Hugging Face's policies. Never put passwords, tokens or keys in files or commits; use Colab Secrets. Ask the user before anything irreversible or outside this project (deleting data you didn't create, spending the last 15 CU, changing account settings, publishing under a new name).

Within those rules you have **full authority over the project**: architecture of the pipeline, data, training method, hyperparameters, evaluation design, repo structure, and what ships. Make decisions; don't ask permission for routine engineering.

---

## Part B: Mission and hard constraints

| Item | Value |
|---|---|
| Goal | **Threnody V4**: a fine-tune of `Qwen/Qwen3.5-9B` that is **substantially better than both base Qwen3.5-9B and Threnody V3** across many fields: coding, agentic tool use, long-horizon multi-turn work, biology, chemistry. This is the target to aim for. The claims you publish must be limited to what you measure (Part A #1, #9). |
| Deadline | **Today (2026-11-05)**: a finished, evaluated **V4 GGUF** (at least Q8_0 and Q4_K_M) published. |
| Compute | **Google Colab only, one GPU: A100 (40/80 GB) or G4 (~96 GB).** No other hardware or cloud. |
| Budget | **127 Colab compute units remaining.** The V2 run logged ~10 CU per GPU-hour (`docs/PROGRESS.md`), so plan for about **11–12 GPU-hours total**. Read the real rate from Colab when you connect and re-plan. **Hold back 15 CU** for export and upload; never spend it without asking. |
| Account | The Colab / Google Drive account is the one the user gives you directly (not stored in this public file). |
| Repos | Create a **new GitHub repo `Threnody-V4`** (same owner as Threnody-V3). Treat Threnody-V3 as read-only reference; link it from V4's README. |
| Release | Hugging Face `MalxTech/Threnody-9B-V4-GGUF` (or the user's chosen name). The model card is generated from measured results. |

**Reality check (read this):** the full plan in `docs/05_V4_PLAN.md` needs far more than 12 GPU-hours. Today, run the **compressed track in Part E**. It targets the improvements most achievable on a small budget: fewer loops, correct tool calls, recovery after errors, and on-policy capability gains. It cannot statistically prove broad gains. Ship what you measured, label it accurately, and leave the full pipeline ready for the next budget.

---

## Part C: Read these V3 files first (cross-reference)

| File | Why |
|---|---|
| `docs/05_V4_PLAN.md` | **The V4 plan.** §1 defines "genuinely better" (G1–G7), ranks the levers, and gives the release scoreboard. §4 has the staged plan and Stage 4 RL settings. |
| `reports/Threnody V4 improvement plan.md` | The research behind the plan, with citations. Key evidence: SFT degraded this exact base, while outcome-only RL improved it (TMax, arXiv 2606.23321). |
| `research_notes/Threnody V4 improvement plan/*.md` | Detailed notes: SFT data, agentic RL and loops, PEFT/merging/abliteration, evaluation, base model and peers, repo data profile. |
| `docs/KNOWN_DISCREPANCIES.md` | What went wrong in V3's records. Don't repeat any of it. |
| `docs/04_V3_TRAINING_POSTMORTEM.md`, `docs/PROGRESS.md` | V3/V2 run history, CU burn rates, and Colab failure modes (e.g., a vLLM install broke torch and wasted ~15 CU). |
| `results/v3_final.json`, `results/v3_gate_*.json` | V3 and base numbers (author-reported; nothing per-task; use them as context only, never as your baseline). |
| `pipeline/` | V3 code to reuse: `train_sft.py` (LoRA trainer, chunked CE), `merge.py`, `export_gguf.py`, `gates.py`, `sci_eval.py`, `gen_v2.py`, `build_v3.py`, `run_v3.sh`. **Known bugs:** `abliterate.py` (8+8 prompts, misses DeltaNet `out_proj`, may not run on the multimodal checkpoint); `run_v3.sh` has silent fallbacks; `ckpt_gate.py` selects on the "held-out" suite. |
| `eval/agentloop/` | Agent harness. The 14 tasks are the **same tasks** under every schema; `opencode`/`codex` are held-out **schemas**, not held-out tasks. `codex` was never run for V3. Defaults to T = 0; change that. |
| `eval/biochem_hard/` | **Do not trust it:** `max_tokens=128` truncates answers (71% of base rows had no answer). Use `pipeline/sci_eval.py` (MMLU-Pro bio+chem) instead, or fix it first. |
| `pipeline/templates/qwen35_chat_template.jinja` | Pinned template. Check that tool calls render **after** `</think>` (llama.cpp issue #20837). |

Drive assets from V3 (if present under the Threnody Drive folder): `teacher/traj_v2.jsonl`, `sci_v2.jsonl`, `code_verified.jsonl`, `regen.jsonl`, `hard_eval.jsonl` (AgentLoop-Hard, 30 tasks), cached llama.cpp builds (`llama_bin_sm*.tgz`). Verify each one exists before you depend on it.

---

## Part D: Build your Colab control tool (within Colab's terms)

Colab has no official job API. Build a **pull-based runner**, not a remote shell or tunnel (those are restricted under Colab's terms on some tiers and could get the account suspended, losing the remaining CU).

1. **`colab/runner.ipynb`** (in Threnody-V4), one cell to start:
   - mount Drive;
   - read tokens from **Colab Secrets** (`GITHUB_TOKEN`, `HF_TOKEN`);
   - `git clone`/`pull` Threnody-V4;
   - restore the cached llama.cpp build;
   - run `pipeline/run_v4.sh` in the background (`nohup`), with logs written to `Drive/Threnody/v4/logs/`.
2. **Stage markers + resumability:** each stage writes `markers/v4_<stage>.done`. On restart, the runner skips stages already done and resumes training from the newest checkpoint.
3. **Status channel:**
   - Every 5 min, a watchdog writes `status.json` (stage, step, loss, GPU memory, CU estimate, last error) to Drive.
   - Push a short status commit to a `runs/` branch of Threnody-V4 every 30 min, so you can monitor from GitHub without the notebook open.
4. **Control channel:** the runner polls `control.json` in the repo or on Drive (`{"action": "continue|stop|skip_stage", "stage": ...}`), so you can steer between stages by committing a change.
5. **Keep-alive:** keep the session productive (jobs running). Disconnect the runtime as soon as the final stage completes, to stop CU burn.
6. **Env pinning:** pin versions in `requirements.txt`. Install **no** stack that replaces Colab's torch (that is what wasted ~15 CU in V2). Check Qwen3.5 support first: transformers ≥ 5, `flash-linear-attention`, hybrid DeltaNet LoRA.

---

## Part E: Today's compressed track (~11–12 GPU-hours; estimates are [Inf])

| # | Stage | GPU-h | What to do | Exit criterion |
|---|---|---|---|---|
| 0 | Setup | 0.5 | Runner, env, llama.cpp cache, Threnody-V4 repo skeleton, `eval/PREREGISTRATION.md` (G1–G7 adapted to today's suites) committed **before** training | Smoke test passes |
| 1 | Base baseline | 2.0 | Base Qwen3.5-9B Q8_0 at Qwen's recommended sampling: AgentLoop 14 tasks × {agentloop, opencode, codex} × k = 2; AgentLoop-Hard 30 (if available); `sci_eval.py` 400 bio+chem; a BFCL-V4 subset if time allows. Per-task JSONL + loop metrics (consecutive + A-B-A-B, calls inside `<think>`, parse failures). Bucket the failures into a taxonomy. | Baseline scoreboard generated by code |
| 2 | On-policy data | 3.0 | Sample **base** at T = 1.0 on **training** tasks only (V2/V3 task pools via `gen_v2.py`, `code_verified.jsonl`; never the eval tasks): <br>(a) keep verified successes, including error→recovery runs, with failed steps masked from the loss; <br>(b) mine DeLoop-style pairs (rejected = the step that starts a loop, chosen = the next non-looping step); <br>(c) build pairs fixing tool calls placed inside `<think>`; <br>(d) add a small base-generated replay set (bio/chem + general) to prevent forgetting. | ≥ several hundred verified on-policy trajectories + pairs, with hashes logged |
| 3 | Train | 1.5 | From **base**, LoRA r = 32, α = 32, all decoder linear layers. Stage A: rejection-sampled fine-tuning on (a) + (d), low LR (≈5e-5), 1 epoch. Stage B: DPO on (b) + (c), β ≈ 0.1, 1 epoch. FP32 LM head. Checkpoint every 20 steps. Track KL to base. | Loss is sane, KL is bounded, no NaNs |
| 3b | (Optional) GRPO pilot | ≤1.5 | Only if Stages 0–3 finished under budget **and** a 20-minute infrastructure test proves single-GPU LoRA GRPO works on Qwen3.5. Settings: `05_V4_PLAN.md` Stage 4, group size 4–8, ≤ 100 steps, tasks where base passes between 1 and 7 of 8 samples. | Reward rises on dev; no collapse |
| 4 | Evaluate | 2.0 | Merge at strength 1.0 → Q8_0. Same suites, harness, sampling and k as Stage 1. Choose between candidates on **dev only**. Run the frozen check once. Paired deltas with CIs where n allows. | Scoreboard generated by code |
| 5 | Export + publish | 1.0 | GGUF ladder (Q8_0, Q6_K, Q4_K_M; others optional). Q4_K_M vs base Q4_K_M quick agent check. KLD on text **separate from** the imatrix data. Model card generated from the scoreboard. Upload to HF, push the repo, tag `v4.0`. | Files live; card matches data |
| — | Reserve | ≥1.5 (15 CU) | Disconnects, reruns | — |

Rules for today:
- **No abliteration** in the main release. If you want an abliterated variant, it must be a separate, fully evaluated build, and only if budget remains.
- **No off-policy teacher SFT** as the main method (it degraded this base in TMax).
- If V4 doesn't beat base on the pre-registered primary check, ship it as **"V4-experimental"** with the honest scoreboard. Do not call it better.
- **Release label wording:**
  - "**Better than base on X (measured)**" only where the CI excludes zero.
  - "**No measurable difference**" where it doesn't.
  - "**Not measured**" for fields you didn't test.

---

## Part F: Threnody-V4 repo layout

```
README.md                 # what V4 is, measured scoreboard, links to Threnody-V3 and its audit
colab/runner.ipynb        # Part D
pipeline/run_v4.sh        # stages 0–5, markers, no silent fallbacks
pipeline/                 # reused/adapted from V3 + new: sample_onpolicy.py, mine_loop_pairs.py, train_rft_dpo.py, (rl_grpo.py)
eval/                     # harness with sampler flags, k-runs, loop_metrics, scoreboard.py, PREREGISTRATION.md
results/<run_id>/         # manifest.json, per-task JSONL, scoreboard.json (all generated)
docs/V4_RUN_LOG.md        # timestamped log incl. CU spent per stage
docs/MODEL_CARD.md        # generated
```

The README's first lines must link `YoMosa2009/Threnody-V3`, `docs/05_V4_PLAN.md` and `docs/KNOWN_DISCREPANCIES.md`.

---

## Part G: End-of-day report (send to the user)

1. What shipped: links to the V4 repo, the HF repo, and the files.
2. **The scoreboard** (base vs V3-if-run vs V4, Δ pp, n, CI), copied from the generated file.
3. Which goals were met, missed, or not measured, field by field (coding, agentic tool use, long-horizon, biology, chemistry).
4. CU spent per stage and CU remaining.
5. Failures, deviations from this plan, and the next three things to do with a bigger budget (from `05_V4_PLAN.md` Stages 0 and 4).
