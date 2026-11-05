# Threnody 3.0: Known Discrepancies and Evaluation Limitations

_Compiled 2026-11-05 from the files in this repository and the published Hugging Face model card._

This page lists where the published records disagree with each other or with the code. **It does not change any
score or any code.** When two records disagree, this page does not say which one is correct. Where the repository
has no evidence that settles a conflict (for example, run logs), the item is marked **unresolved**.

Score differences are given in **percentage points (pp)**, computed from the pass counts or values in
[`results/v3_final.json`](../results/v3_final.json).

---

## 1. Evaluation limitations

- **The author ran and reported every benchmark.** No third party has reproduced them. The run logs and per-task
  outputs for the V3 candidates are not in this repository; only summary JSON files are.
- **The `opencode` held-out suite influenced which model was released.** `gates.fast_score()` adds the
  `al_opencode` pass count to its score. The training-time checkpoint gate (`pipeline/ckpt_gate.py`) and the
  candidate ranking in `pipeline/run_v3.sh` (stage `select`) both use that score. So the 13/14 `opencode` result is
  not a clean held-out estimate: the same suite helped pick the checkpoint and the candidate.
- **The suites are small.** `opencode` has 14 tasks, AgentLoop In-Harness 14, OmniAgent-Bench 16 and
  AgentLoop-Hard 30. A one-task change is 7.1 pp on a 14-task suite. No confidence intervals or repeated seeds are
  reported.
- **The science score covers biology and chemistry only, not MMLU-Pro as a whole.** `pipeline/sci_eval.py` draws a
  fixed random sample (`random.Random(0)`, `--n 400`) from the MMLU-Pro **biology and chemistry** test questions.
  The 76.0% / 76.5% / 78.5% figures are not overall MMLU-Pro scores, which would cover 14 subject categories.
- **Only non-thinking mode was measured.** The `mid` suite that `run_v3.sh` uses evaluates `nothink` only. No
  thinking-mode V3 results are published.
- **Quantization metrics are not task results.** See §8.
- **No vision results.** An F16 `mmproj` file is shipped, but no vision evaluation is published.

---

## 2. The "frontier" and "state of the art" claims

The README, postmortem and Hugging Face card call the model "frontier" or "state of the art". The published results
compare V3 only with its own base model (Qwen3.5-9B) and its own earlier version (V2), and V3 scores **below the base
model on three of the four agent suites** (§5). No external comparison supports either claim. The documents now
describe the model as an **experimental agentic-coding fine-tune of Qwen3.5-9B**.

---

## 3. Training records

### 3.1 Token count: target vs. reported

| Record | Value |
|---|---|
| `run_v3.sh` → `build_v3.py --total_tokens 35e6` | **35M tokens: configuration target** |
| `docs/03_V3_PLAN.md` | "~1.2 epochs on ~35M tokens", "300–400 optimizer steps" |
| `results/v3_training_report.json` → `dataset_breakdown.total_tokens` | **2,895,493 tokens** across 662 trajectories |
| `docs/PROGRESS.md`, postmortem pipeline log | 2,895,493 tokens, 56 optimizer steps |

The README and HF card described the mix as "~35M tokens". The reported run used about **2.9M tokens (≈8% of the
target)** and 56 optimizer steps instead of the planned 300–400.

### 3.2 Data-mix percentages disagree

| Share | README / HF card / `03_V3_PLAN.md` | `v3_training_report.json` |
|---|---|---|
| 35% | multi-schema agent traces (8 tool schemas) | `nvidia/Open-SWE-Traces` |
| 20% | verified algorithmic coding | `code_verified` |
| 20% | Open-SWE-Traces | `swe_traces` |
| 15% | BioChem reasoning | `sci_v2 (BioChem)` |
| 10% | cybersecurity & tool schema | `regen & identity` |

The two lists give different sources for the 35% and 20% slices. **Unresolved.**

### 3.3 Hyperparameters: report vs. trainer code vs. launch script

| Setting | `v3_training_report.json` | `pipeline/train_sft.py` | `pipeline/run_v3.sh` invocation |
|---|---|---|---|
| Per-device batch size | **4** | **1, hard-coded** (`per_device_train_batch_size=1`; `Collator` uses `feats[0]` only) | n/a (not configurable) |
| Gradient accumulation | **8** | `--grad_accum` default **16** | not passed → **16** |
| Effective batch | **32** | 1 × grad_accum | **16** |
| Learning rate | 1e-4 | default 1.5e-4 | `--lr 1e-4` ✔ matches report |
| Max sequence length | 32,768 | default 16,384 | `--max_len 32768` ✔ matches report |
| Epochs | 1.219 | default 2.0 | `--epochs 1.2` (≈ match) |
| Warmup | 5 steps | `warmup_steps=0.03` | n/a |
| Weight decay | 0.01 | 0.0 | n/a |
| Optimizer | `adamw_torch` | `adamw_torch_fused` on CUDA | n/a |
| LoRA rank / alpha | not reported | default 64 / 32 | not passed → 64 / 32 |

The reported step count does not fit either configuration. At 662 trajectories and 1.219 epochs, an effective batch
of 32 gives about 25 steps and an effective batch of 16 about 50, not the reported 56. (Matching 56 steps at
effective batch 16 needs about 735 training sequences.) The repository has no run log (`v3_sft.log`) or
`log_history.json`, so **the settings that actually ran are unverified.** The code is left unchanged.

### 3.4 "Full SFT" vs. LoRA

`train_sft.py` creates LoRA adapters (`peft.get_peft_model`, `LoraConfig(r=64, lora_alpha=32)`) on the decoder
linear layers. The vision tower, MTP head and `lm_head` are not adapted. `run_v3.sh` then merges the adapter into the
base model at strength 0.75 (`merge.py`). The training report's `"experiment": "... Full SFT ..."` and the README's
"Full SFT Convergence" can be read as full-parameter training. The documents now say **supervised fine-tuning using
LoRA**. The JSON file is left as published.

---

## 4. Gate records vs. gate code

Both `results/v3_final.json` (`"passed": true`, six gates all `pass: true`) and `results/v3_gate_s0.75.json`
(`"pass": true`) record the released `s0.75` candidate as passing. **Neither gate function in the current
`pipeline/gates.py` produces those records, and neither would pass this candidate.**

**a) `gate_v3()`, the V3 gate in `gates.py`:**

| Check in code | `s0.75` value (`v3_final.json`) | Result |
|---|---|---|
| OmniAgent-Bench weighted ≥ 93.8% | 84.61% | **fail** |
| OmniAgent-Bench ≥ Base | 84.61% vs 92.71% | **fail** |
| AgentLoop In-Harness ≥ 13/14 | 10/14 | **fail** |
| AgentLoop In-Harness beats Base | 10/14 vs 11/14 | **fail** |
| AgentLoop In-Harness max_repeat ≤ 2 | 1 | pass |
| `opencode` ≥ 10/14, beats Base, max_repeat ≤ 2 | 13/14 vs 12/14, rep 1 | pass |
| BioChem-Hard overall ≥ 40%, chemistry ≥ 15% | 78.5%, 73.47% | pass |

**b) `gate(..., "mid")`, the gate `run_v3.sh` actually calls during candidate selection:** this fails on AgentLoop
(10 < 11), OmniAgent weighted (84.61 < 92.71), AgentLoop-Hard (17 < 19), and the strict AgentLoop-Hard total check.

**c) Revision history:** the gate names in the published records (for example "BioChem Chemistry beats Base (69.4%)
and V2 (14.1%)" and "Zero Repetition Loops across passed trajectories") do not appear in any revision of
`pipeline/`, `eval/` or `notebooks/` in this repository's git history. `gates.py` has not changed since commit
`52edc9f`. **The code revision that generated the published passing records cannot be identified from this
repository.**

**d) Candidate choice:** `run_v3.sh` sorts candidates by `(gate pass, fast_score)` and takes the first. If both
candidates fail the gate, the higher `fast_score` wins. Computed from the published numbers, `s0.75` scores 33.99
and `s1.0` scores 32.39. That would explain `s0.75` being selected without passing a gate. This is an inference from
the code; it is not confirmed by a log.

### 4.1 An earlier, superseded V3 result set

Commit `52edc9f` (dated 2026-11-04 22:19 UTC) published a different `v3_final.json`. It records `"chosen": "s1.0"`,
passes every `gate_v3()` check, and has different scores for the **same base model**:

| Metric | `52edc9f` record | Current record (`86c9406`) |
|---|---|---|
| Released candidate | `s1.0` | `s0.75` |
| Base AgentLoop In-Harness | 12/14, max_repeat **18** | 11/14, max_repeat **1** |
| Base `opencode` | 9/14 | 12/14 |
| Base AgentLoop-Hard | 20/30 | 19/30 |
| Base MMLU-Pro bio+chem (400) | 75.25% | 76.0% |
| Base BioChem-Hard | 35.8% (n = 150) | 76.0% (n = 400) |
| V3 candidate OmniAgent / In-Harness / opencode / Hard | 93.82% / 13/14 / 11/14 / 24/30 | 84.61% / 10/14 / 13/14 / 17/30 |

That commit is dated about three hours before the earliest stage time in the postmortem's pipeline log (01:11 UTC on
the 2026-11-05 run). Git commit dates are set by the committer and can be wrong. `docs/PROGRESS.md` (2026-11-04 entry)
also says "Verified all acceptance gates pass at s1.0", which matches the superseded record. `v3_gate_s1.0.json`
records `s1.0` as failing. **Unresolved.** The "Base looped up to 18×" note on the earlier scorecard chart came from
the superseded record.

---

## 5. Anti-loop claims

`max_repeat` (from `eval/agentloop/schemas.max_repeat`) is the longest run of **consecutive identical assistant
actions** in a task. The harness summary records the maximum over **all tasks in the suite, passed or failed**
(`eval/agentloop/harness.py`). 1 means no action was immediately repeated; 2 means one immediate repeat.

Measured values for V3 `s0.75` in `results/v3_final.json`:

| Suite | Passed | max_repeat |
|---|---|---|
| `opencode` held-out (14) | 13/14 | **1** |
| AgentLoop In-Harness (14) | 10/14 | **1** (HF card: **27**, see §6) |
| AgentLoop-Hard (30) | 17/30 | **2** |

So "zero action looping" holds only for `opencode` and In-Harness in the GitHub record, and only by this metric. It
does not hold for AgentLoop-Hard. "Zero repetition loops on passes" also misdescribes the metric, which covers every
task, not only passed ones. `docs/PROGRESS.md` says "`max_repeat = 1` across 12 of 13 passed tasks". That
contradicts the suite-level max_repeat of 1 for `opencode`, and the repository has no per-task data to settle it.
**Unresolved.**

V3 vs. Base on agent suites (pp, from pass counts):

| Suite | Base | V3 `s0.75` | Difference |
|---|---|---|---|
| `opencode` held-out | 12/14 | 13/14 | **+7.1 pp** (+1 task) |
| AgentLoop In-Harness | 11/14 | 10/14 | **−7.1 pp** (−1 task) |
| AgentLoop-Hard | 19/30 | 17/30 | **−6.7 pp** (−2 tasks) |
| OmniAgent-Bench (weighted) | 92.71% | 84.61% | **−8.1 pp** (15/16 → 13/16) |

---

## 6. Hugging Face model card vs. GitHub records

The published card at `MalxTech/Threnody-9B-V3-GGUF` is generated by `pipeline/publish_hf.py`. A corrected
replacement is in [`docs/HF_MODEL_CARD.md`](HF_MODEL_CARD.md).

| Card statement | Conflict |
|---|---|
| "**STRICT DOMINANCE:** Passed every gate … across all 4 suites" | The card's own table shows V3 below Base on OmniAgent-Bench, AgentLoop In-Harness and AgentLoop-Hard. The text is hard-coded in `publish_hf.py` and appears whenever the input file has `"passed": true` (or no `passed` key). |
| "at full strength (`s1.0`)", table header "V3 (s1.0)" | Released strength is `s0.75` (`v3_final.json`, `v3_training_report.json`). `s1.0` was rejected (`v3_gate_s1.0.json`). The `s1.0` strings are hard-coded in `publish_hf.py`. |
| "`s0.75` (Full Strength — zero dilution required)" | `merge.py` at strength 0.75 scales the LoRA update to 75% (interpolating toward the base). That is dilution by definition. |
| AgentLoop In-Harness V3 `[rep=27]` | GitHub `v3_final.json` and `v3_gate_s0.75.json` both record max_repeat **1**. The card table is filled from whatever JSON was passed to `publish_hf.py --final`, so the uploaded input differed from the published `v3_final.json`. (Note: `docs/PROGRESS.md` records the V2 run's `s0.75` candidate at rep 27 on AgentLoop. The match is an observation, not proof of the cause.) **Unresolved.** |
| "state-of-the-art open-weight agentic fine-tune" | No external comparison supports it (§2). |
| "Balanced Multi-Curriculum (~35M tokens)" | Reported run used 2,895,493 tokens (§3.1). |
| BioChem-Hard row V2 = 37.6% next to MMLU-Pro row V2 = 76.5% | See §7. |

---

## 7. Science-score records

- **`results/v3_final.json`** gives identical numbers for `sci` (MMLU-Pro bio+chem, n = 400) and `biochem_hard` for
  Base (76.0 / 86.45 / 69.39) and for V3 (78.5 / 86.45 / 73.47). For V2 they differ: `sci` overall 76.5% vs.
  `biochem_hard` overall 37.6% (biology 44.5%, chemistry 14.1%). V2's 44.5% / 14.1% subsets cannot average to
  76.5%. The V2 `biochem_hard` values match the superseded record's 150-question BioChem-Hard suite (§4.1), a
  different test.
- **`results/biochem_hard_final.json`** (BioChem-Hard, 844 questions, Q4_K_M builds) records Base at **25.36%**
  and V2 at **35.55%**. `v3_final.json` records Base BioChem-Hard at 76.0%.
- The README and postmortem said "+42.0% over V2" (biology) and "+59.4% over V2" (chemistry). Those subtract V2's
  BioChem-Hard subsets from V3's MMLU-Pro subsets, so they compare different tests. These comparisons have been
  removed. V2 has no published MMLU-Pro bio/chem subset scores in the current record.
- The README stated "Q8_0 **and Q4_K_M**" builds for all models. `run_v3.sh` evaluates Base and the V3 candidates on
  Q8_0. `biochem_hard_final.json` used Q4_K_M builds of Base and V2.

**Unresolved.** The V3 vs. Base MMLU-Pro bio+chem difference in the current record is **+2.5 pp** overall,
**+4.1 pp** chemistry and **0.0 pp** biology.

---

## 8. Quantization claims

`results/v3_quant_report.json` comes from `pipeline/export_gguf.py`. It runs `llama-perplexity --kl-divergence`
over 40 × 4096-token chunks of `val.jsonl`, comparing each quant with the BF16 GGUF of the same merged model. The
**same text was used for the imatrix calibration.** The measurements are token-distribution agreement (median KLD;
"Same top p" = % of positions where the top token matches the reference).

These numbers do not show that a quant **performs** the same on coding or agent tasks. No benchmark was run per
quant. Descriptions such as "near-lossless", "maximum accuracy" and "FP16 lossless" have been replaced with the
measured values. For example, Q6_K's 96.664% same-top-token rate means the top token differs from BF16 at about
3.3% of positions.

---

## 9. Statements removed from the postmortem as unsupported

| Statement | Why removed |
|---|---|
| "maintaining 100% of the learned agentic priors" | Not measured. `s0.75` scales the adapter update to 75%. |
| "It completed all 14 held-out tasks" | Record shows 13/14 passed. |
| "proves … it understands abstract tool calling contracts" | 14 tasks on one held-out schema, and that suite also influenced selection (§1). The result suggests transfer to the tested schema; it cannot establish general tool understanding. |
| "maximum intelligence extraction" (README, `run_v3.sh` comment) | Not measured. The comment in `run_v3.sh` is left unchanged because it is code. |
| "completely removed spurious safety tripwires … while preserving ethical boundaries" | Not shown by the published red-team file (§10). |
| "Eliminated" / "Cured" (loops, forgetting) | See §5 and §7. |
| "catastrophic perceptual collapse" from quantizing the vision tower | No vision evaluation published. |

## 10. Red-team report

`results/v3_redteam_report.json` has four categories, each with a free-text status (for example "Verified (Safe &
Uncensored for Academic Analysis)"). It has no prompts, outputs, pass counts or totals.

`eval/redteam/harness.py` writes a different structure: `total`, `passed`, `pass_rate_pct`, and per-prompt
`details` with substring checks. So the published file does not match what the harness produces. Where it came from
cannot be determined from this repository. **Unresolved.**

Neither the file nor the harness's substring checks show that abliteration "preserved ethical boundaries", and
neither replaces an external safety evaluation of an abliterated model.
