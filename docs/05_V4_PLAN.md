# Threnody V4: Engineering Plan

_Written 2026-11-05. **Hard constraint:** all training and evaluation runs in **Google Colab on one GPU**, either an A100 (40/80 GB) or a G4 (~96 GB). No multi-GPU or multi-node work and no external clusters._

Evidence base:
- Full research report: [`reports/Threnody V4 improvement plan.md`](../reports/Threnody%20V4%20improvement%20plan.md)
- Research notes: [`research_notes/Threnody V4 improvement plan/`](../research_notes/Threnody%20V4%20improvement%20plan/)
- V3 record audit: [`docs/KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md)

Labels used in this document:
- **[Repo]**: checked in this repository's code or data.
- **[Lit]**: from a cited source in the report or notes.
- **[Inf]**: my inference or estimate. Treat it as a hypothesis to test.

---

## 0. The thesis in five lines

1. **V3's instruments could not see a real improvement, and they could not see a real regression either.** Each suite has 14–30 tasks, which gives a 95% interval of about ±18–26 pp. The "held-out" suite was also used for selection, and BioChem-Hard mostly measured answer truncation. **[Repo]**
2. **On this specific base model, off-policy SFT hurts.** TMax reported that SFT on agent trajectories, even with a Qwen3.6-27B teacher, cut Qwen3.5-9B on Terminal-Bench Lite from 41.9 to 35.5 (and to 31.3 with a larger mix). Outcome-only RL from the raw base raised SWE-bench Verified from 46.2 to 55.3. **[Lit, arXiv 2606.23321]** V3's pattern fits that result. **[Inf]**
3. **Many of the loops are likely caused by decoding.** V3 evaluated at temperature 0 (greedy). Qwen's model cards warn that greedy decoding causes endless repetition. **[Repo/Lit]**
4. **So V4 has three steps, in this order:** fix the yardstick, fix decoding and loops, then train on-policy against executable tests. No more large off-policy SFT, fractional merges, or unmeasured abliteration. **[Inf]**
5. **Success is defined in advance, before any training** (full definition and levers in §1): V4 minus base pass@1 on a frozen 300–400-task, Docker-free local repo-task test split, k = 3 runs per task, with a paired-bootstrap 95% CI that excludes zero. On top of that, V4 must be no more than 2 pp worse than base on BFCL-V4, tau2 and LiveCodeBench (Holm-corrected). **[Inf]**

The realistic target on one Colab GPU is a **few points of statistically supported gain plus clearly better loop and tool-call behavior**. It is not TMax's +9 points, which took 8 H100 nodes for 1–2 days per run. **[Inf]**

---

## 1. What would make V4 *genuinely* better

V3 was not shown to be better than base. On the published record it scores lower on 3 of 4 agent suites, and every difference sits inside the noise ([`KNOWN_DISCREPANCIES.md`](KNOWN_DISCREPANCIES.md)). "Genuinely better" must therefore be defined before any training, and it must be hard to fake. This section defines it, then ranks the work most likely to achieve it on one Colab GPU.

### 1.1 Definition: what V4 must show, all at once

| # | Requirement | How it is measured | Why it matters |
|---|---|---|---|
| G1 | **Better than base on one pre-registered primary endpoint** | V4 − base pass@1 on the frozen 300–400-task Docker-free repo-task test split. k = 3 runs per task. Task-clustered paired-bootstrap 95% CI must exclude 0. Test split run **once**, after all selection. | One fixed question, answered with enough power to see ~5 pp **[Inf: McNemar sizing]** |
| G2 | **No meaningful regression anywhere else** | Non-inferiority margin −2 pp, Holm-corrected, on BFCL-V4, tau2, LiveCodeBench (post-cutoff window), MMLU-Pro bio+chem sample, and the dev agent suites | V3 "won" one suite while losing three. That must not happen again. |
| G3 | **Better in the user's actual setting** | The same comparison on the **shipped Q4_K_M GGUF vs. base Q4_K_M**, in llama.cpp with the shipped template, in opencode- and codex-style harnesses | People run quants in CLIs, not BF16 in a lab harness |
| G4 | **Both modes** | Report thinking and non-thinking separately; neither may regress beyond the G2 margin | V3 measured only non-thinking |
| G5 | **Fair sampling** | Base and V4 both use Qwen's recommended sampling (and V4's own recommended settings if they differ, reported as such) | A T = 0 handicap on base would make V4 look better for free |
| G6 | **Behavioural wins are counted, not claimed** | Loop rate (consecutive + A-B-A-B), duplicate calls per task, tool-call parse-failure rate, share of tool calls inside `<think>`, tasks hitting max turns, median tokens and turns to solve | "Fewer loops" must be a measured number |
| G7 | **Reproducible** | Every number traces to a manifest (git SHA, hashes, sampler, seed) and per-task JSONL committed to the repo. Gate verdicts are produced by code, never typed. | Prevents a repeat of V3's record conflicts |

**Kill criterion:** if no candidate meets G1 + G2, **do not release V4 as "better"**. Publish the null result and the improved evaluation suite instead. The suite alone is a real contribution.

**Attribution rule:** if most of V4's gain comes from sampling, the template or the harness guard rather than the weights, the card must say so, by reporting **base + Threnody runtime profile** as its own row.

### 1.2 Levers ranked by expected gain per Colab GPU-hour

Expected-effect figures are **[Inf]** estimates for one-GPU scale unless marked. Order matters: each step is cheap insurance for the next.

**Lever 1: Stop damaging the base (prerequisite, ~0 extra GPU-h).**
V3's biggest problem was probably what it broke, not what it failed to add.
- Start every run from **base Qwen3.5-9B**.
- Use **on-policy data only**: RL rollouts or base's own verified samples.
- Track **KL(V4 ‖ base)** on a fixed general prompt set during training. Forgetting follows this KL distance **[Lit: RL's Razor]**.
- Mix in base-generated general replay if any SFT is used.
- Merge at strength 1.0 only.
Expected: removes most of the −7 to −8 pp agent regressions seen in V3 **[Inf; Lit: TMax SFT −6 to −10 on this base]**.

**Lever 2: Find out where base actually fails (Stage 0 by-product, ~5 GPU-h).**
You cannot beat a strong base by training on what it already does well.
- Build a **failure taxonomy** from base's Stage 0 trajectories, roughly these buckets:
  - wrong edit or logic
  - tool-call format or parse error
  - tool call inside `<think>`
  - consecutive loop or A-B-A-B oscillation
  - gives up early or never runs tests
  - context exhaustion
  - environment or setup error
  - harness or schema mismatch
- Put **every later lever on the two or three largest buckets**. This decides whether V4's leverage lies in behaviour (cheap) or capability (expensive).

**Lever 3: Behavioural reliability in the target CLIs (~10–25 GPU-h). This is the most achievable genuine win.**
These are the failures local users hit most and that a small run can actually fix:
- **Tool calls after `</think>`.** llama.cpp issue #20837 is open, and Qwen3.5/3.6 often put the call inside `<think>`, so it is dropped **[Lit]**. Train against it (reward penalty or DPO pairs) and ship a tested template.
- **Loops and oscillation.** DeLoop-style DPO on base's own loop/continuation pairs cut the loop rate from 57.3% to 6.1% on Qwen3-8B, with success flat. Qwen3-14B gained +7.2 pp **[Lit, not peer-reviewed]**.
- **Recovery after errors.** After a failed command or test, change approach instead of repeating. Use RL reward plus masked error→recovery steps; V3's filter threw these examples away.
- **Verify before finishing.** Reward runs that execute tests after the last edit. The harness already records `verified`.
- **Harness/template robustness.** Train across the native `qwen3_coder` format plus the opencode, codex and claudecode-style schemas. Template diversity raised SWE-bench Verified at equal data volume in Qwen3-Coder-Next **[Lit]**.
Expected: large measured improvements on G6 metrics, plus a few pp on G1/G3 wherever those failures were costing tasks **[Inf]**.

**Lever 4: On-policy RL on verified tasks at the edge of competence (~20–60 GPU-h). This is the capability lever.**
- Run the Stage 4 setup on **pure-Python, venv-installable repo tasks graded by hidden tests**.
- **Keep only tasks where base passes 1–7 of 8 samples.** Tasks base always passes teach nothing; tasks it never passes give no gradient **[Lit: TMax zero-variance filtering; easy data wasted 7× rollouts]**.
- Use outcome reward plus small penalties for loops and calls inside `<think>`.
- Use the stability guards (FP32 LM head, ≤300 steps).
Expected: +2–5 pp in-distribution; transfer smaller **[Inf]**. For reference, cluster-scale TMax got +9 on SWE-bench Verified on this exact base **[Lit]**, so the direction is proven and the size is limited by compute.

**Lever 5: Specialize where the target users are (cheap multiplier).**
A 9B model will not beat the base across all of coding on one GPU, but it can be **measurably better at the job Threnody is for**: multi-turn CLI agent work in opencode/codex-style harnesses on Python repos, served by llama.cpp.
- Put RL tasks, schemas and evaluation there.
- Use the G2 margin to guard everything else.
Saying "better at X, not worse at Y" honestly is stronger than a vague "frontier" claim.

**Lever 6: Make the shipped quant the thing that is better (~10 GPU-h).**
- Run the final comparisons on **Q4_K_M vs. base Q4_K_M** as well as Q8_0 (G3).
- Measure KLD on agent transcripts kept apart from the imatrix data, and log parse failures per quant.
- If Q4_K_M loses the G1 gain, recommend Q5_K_M/Q6_K and say why.
- Optionally build the imatrix from Threnody's own agent transcripts, and verify that it helps on tasks, not just on KLD.

**Lever 7: Optional, if budget remains: on-policy distillation (~15–40 GPU-h, G4 preferred).**
- A quantized stronger teacher (e.g., a 27B Qwen) scores the student's own rollouts token by token.
- It matched RL quality at about 1/10 the compute in Thinking Machines' setting **[Lit]**.
- Memory is the constraint. Feasible on a G4 (~96 GB) with a 4-bit teacher; tight on an A100 **[Inf]**.

### 1.3 What will *not* make V4 genuinely better

| Tempting move | Why it fails |
|---|---|
| 10× more off-policy teacher SFT | Lowered this exact base in TMax even with a stronger teacher **[Lit]** |
| Picking the best of many checkpoints on the same suite you report | Selection leakage inflates the reported number **[Repo: V3 opencode]** |
| Lower merge strength until loops stop | Dilutes whatever was learned; hides the cause **[Inf]** |
| Abliteration for "more intelligence" | Measured side effects: GSM8K up to −18.8 pp, TruthfulQA down, insecure code 88–97% on coder models **[Lit]**; no capability gain shown |
| Comparing V4 at tuned sampling vs. base at T = 0 | Inflates the gain; violates G5 |
| Claiming SWE-bench or Terminal-Bench numbers from Colab | Their Docker runners can't run there; SWE-bench Verified is contamination-flagged **[Lit]** |

### 1.4 Release scoreboard (fill from data, not prose)

| Check | Base | V4 | Δ (pp) / 95% CI | Pass? |
|---|---|---|---|---|
| G1 Repo-task test pass@1 (k = 3), non-thinking | | | | |
| G1 Repo-task test pass@1 (k = 3), thinking | | | | |
| G2 BFCL-V4 | | | | |
| G2 tau2 | | | | |
| G2 LiveCodeBench (window) | | | | |
| G2 MMLU-Pro bio+chem sample | | | | |
| G2 Dev agent suites (AgentLoop × 3 schemas, OmniAgent, Hard) | | | | |
| G3 Same, Q4_K_M vs. base Q4_K_M | | | | |
| G6 Loop rate / A-B-A-B / tool-call-inside-`<think>` / parse failures | | | | |

V4 may be called "better than base" only if every row passes. The model card is generated from this table.

---

## 2. Why V3 regressed: ranked root causes

| # | Cause | Evidence | Confidence |
|---|---|---|---|
| 1 | Off-policy SFT overwrote a strong base policy | TMax: SFT lowered Qwen3.5-9B even from a stronger teacher; the same data *raised* Qwen3-8B. **[Lit]** | High that this matters; its exact share of V3's regression is unknown |
| 2 | The evaluation could not resolve differences of under ~20 pp | 14/14/16/30-task suites, single run **[Repo]** | Certain |
| 3 | Greedy decoding (T = 0) caused loops | `harness.py` uses T = 0.0 in non-thinking mode **[Repo]**; Qwen card: "DO NOT use greedy decoding" **[Lit]** | High |
| 4 | The anti-loop filter removed recovery behaviour | `build_v3.py` drops `max_repeat ≥ 2` and sorts by fewest turns **[Repo]**, which biases toward short first-try successes. Lit: keeping error→recovery steps (masked) helps. **[Lit]** | Medium |
| 5 | Too little data for the SFT that was done | 662 trajectories against 12k–66k in published 8B recipes **[Lit]** | Medium (secondary, given cause 1) |
| 6 | Reduced adapter strength (0.75) used as a stabiliser | It shrinks the learned change after the fact; it does not fix the cause **[Inf]** | Medium |
| 7 | Abliteration was unmeasured and possibly mis-applied | See §2.1 **[Repo]** | Unknown effect |
| 8 | Selection leakage | `fast_score` adds opencode passes into checkpoint and candidate choice **[Repo]** | Certain (inflates the opencode result) |

### 2.1 New code findings in `pipeline/abliterate.py` [Repo]

- The refusal direction is estimated from **8 harmful + 8 harmless prompts**, at the last token, per layer.
- It loads with `AutoModelForCausalLM` and looks for layers at `model.model.layers`. For the multimodal Qwen3.5 checkpoint, which `train_sft.py` loads with `AutoModelForImageTextToText`, the language layers may be nested elsewhere. If loading or lookup fails, the script exits with an error. `run_v3.sh` then prints "Standard model weights retained" and **ships the un-abliterated model silently** (`... && M_DIR="$ABL_DIR" || echo ...`). **Whether the released V3 weights were actually abliterated cannot be confirmed without `v3_export.log`.** [Repo/Inf]
- Layer range: ratios 0.35–0.85 of 32 layers give `range(11, 27)`, which is 0-indexed layers 11–26 (16 layers). The docs say "layers 12–28". [Repo]
- It edits only `mlp.down_proj` and `self_attn.o_proj`. In Qwen3.5, only 1 layer in 4 has `self_attn`; the Gated DeltaNet layers' `linear_attn.out_proj` is never orthogonalised. So most writes into the residual stream are only partly edited. [Repo/Lit]
- There is no KL check, no refusal-rate check and no capability check afterwards. [Repo]

---

## 3. Compute envelope (one Colab GPU)

| Resource | A100 80 GB | G4 ~96 GB | Notes |
|---|---|---|---|
| bf16 Qwen3.5-9B weights | ~18 GB | ~18 GB | [Inf] |
| Policy + LoRA training (r = 32, grad-ckpt, 16K context) | fits | fits more comfortably | [Inf] |
| Co-resident vLLM / llama.cpp rollout server | tight; use 0.35–0.45 GPU memory | comfortable | Verify that vLLM LoRA supports the hybrid DeltaNet model **[Inf, unverified]** |
| Co-resident 27B teacher (on-policy distillation) | only quantized, likely too tight | maybe at Q4 | Optional stage only **[Inf]** |
| Docker | **not available** | **not available** | Sandboxes are Python subprocess + venv + timeouts **[Inf]** |

Session discipline (required for every stage):
- Write resumable state to Drive every 20–25 steps: LoRA, optimizer state, RNG, rollout buffer, eval cursor.
- Evaluation appends one JSONL line per task, so a disconnect costs minutes, not a session.
- Every run writes a `manifest.json`: git SHA, adapter/GGUF hashes, llama.cpp build and flags, template hash, `enable_thinking`, full sampler, seed.

---

## 4. Staged plan with go/no-go gates

Hour figures are **Colab GPU-hour estimates [Inf]**. Each stage ends in a decision.

### Stage 0: Rebuild the measurement system (15–30 GPU-h; no training)

Deliverables:
1. **`eval/repotasks/`**: a Docker-free pool of pure-Python repo bug-fix tasks, SWE-smith-style. Each task is pip-installable into a venv and graded by hidden pytest tests run in a subprocess with a timeout.
   - Split: **dev ≈ 100** tasks and **frozen test 300–400** tasks.
   - Repos must be disjoint from training repos and from SWE-bench's.
   - Prefer issues created after the base model's cutoff.
   - Test files are read-only and hash-checked.
2. **Fix the in-house suites** and relabel them as dev or regression suites:
   - Run AgentLoop under all three non-training schemas, including `codex`, which V3 never ran.
   - Relabel `opencode`/`codex` as **"schema transfer, same 14 tasks"**.
   - Commit `hard_eval.jsonl` to the repo (it currently lives only on Drive).
3. **`max_repeat` → `loop_metrics`**: keep consecutive repeats, and also report the **A-B-A-B oscillation rate**, duplicate calls per task, and the share of tasks hitting `max_turns`.
4. **BioChem-Hard**: retire it in favour of `sci_eval.py`. If you keep it, fix `max_tokens = 128` (use ≥ 2048 non-thinking, ≥ 8192 thinking), store full responses, use stable hashed IDs, add a fallback extractor, and report the extraction-failure rate. [Repo]
5. **Public checks with no Docker:** BFCL-V4, tau2-bench (pinned user simulator), LiveCodeBench (window after the cutoff).
6. **`pipeline/gates.py`**: one gate function, called by `run_*.sh`. Its output JSON includes the gate code's git SHA. Gate records are generated, never edited by hand.
7. **Re-baseline Base Qwen3.5-9B and V3** with identical harness, quant (Q8_0) and sampling, k = 3.
8. **Failure taxonomy of base** (§1.2 Lever 2): label every failed base trajectory into buckets; the two or three largest buckets decide what Stages 3–4 target.
9. **Pre-register** the G1–G7 definition (§1.1) in `eval/PREREGISTRATION.md`, committed before any V4 training run.

**Gate:** base re-runs reproduce within their CI, and per-task trajectories are logged. Until this passes, no training.

### Stage 1: Find out what broke V3 (10–20 GPU-h)

- Score V3's merged adapter at strengths **{0, 0.25, 0.5, 0.75, 1.0}**, with and without abliteration, on the dev set.
- **Decision:**
  - If some strength beats base under fair evaluation, light SFT on this base is still viable. Keep it as a fallback.
  - If none does, SFT is the problem. Go RL-first.
  - If abliteration explains the drop, drop abliteration from the main release.

### Stage 2: Free wins from decoding and serving (5–10 GPU-h)

- Evaluate base and V3 at **Qwen's recommended sampling** (see the [base-model notes](../research_notes/Threnody%20V4%20improvement%20plan/base_model_and_peers.md)):
  - thinking + precise coding: T 0.6 / top_p 0.95 / top_k 20 / presence_penalty 0
  - non-thinking: T 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5
- Turn DRY sampling off. Add a **harness loop guard**: on the third identical call, inject "you already tried this; change approach".
- **Tool calls after `</think>`:** llama.cpp issue #20837 (open) drops tool calls emitted inside `<think>`. Ship and test a template that places them correctly, and measure how often calls land inside `<think>`.
- **Decision:** if vendor sampling closes most of the gap or removes most loops, Stage 3 becomes small and Stage 4 becomes the main lever.

### Stage 3: Loop DPO, only if base still loops (5–15 GPU-h)

- Run only if base loops on **≥ ~10% of long tasks** under recommended sampling.
- DeLoop-style method: mine pairs from **base's own** rollouts. *Rejected* = the step that starts a loop; *chosen* = the next non-looping step in the same context.
- Size: ~5–8K pairs, LoRA DPO, β ≈ 0.1, 1 epoch. Settings start from DeLoop's and need tuning. [Lit/Inf]
- Expected effect: a large cut in loop rate; success +0 to a few pp. [Lit: Qwen3-8B success flat 25.1 → 25.9; 14B and 32B rose 7 pp]

### Stage 4: Single-GPU LoRA RL on verified tasks, the main lever (20–60 GPU-h)

Start with **one session that tests the infrastructure only**: TRL/Unsloth GRPO plus a vLLM or llama.cpp rollout server running on Qwen3.5's hybrid layers with LoRA. If that fails, go to Stage 4b.

| Setting | Value | Basis |
|---|---|---|
| Start point | **Base Qwen3.5-9B** (not a V3 merge) | TMax RL-from-base [Lit] |
| Algorithm | GRPO / DPPO-style, token-level loss, **zero-variance group filtering** | TMax [Lit] |
| LoRA | r = 32, α = 32, all decoder linear layers. Ablate `in_proj_a`/`in_proj_b` (Axolotl excludes them). | Thinking Machines: RL works at low rank [Lit]; [Inf] |
| LR | ~1e-5 | TMax full-FT 1e-6 × ~10 for LoRA [Lit → Inf] |
| Group size / prompts per step | 8 (4 if slow) / 8 | scaled down from 32 [Inf] |
| Rollout | T = 1.0, ≤ 16K tokens, ≤ 20 tool calls, **thinking kept across turns** | TMax [Lit], scaled [Inf] |
| Stability | **FP32 LM head** ("especially crucial" for Qwen3.5), checkpoint every 20 steps, stop by 200–300 steps | TMax runs collapsed after ~300 steps [Lit] |
| Reward | Fraction of hidden tests passed + small parse-validity term − exact-repeat penalty − tool-call-inside-`<think>` penalty − mild turn cost | [Inf, built from cited parts] |
| Anti-hacking | Read-only, hash-checked tests; hidden tests; reject edits to test files; log diffs | TMax-9B hacked rewards [Lit] |
| Task curriculum | Pilot each training task with 8 base samples; **keep only tasks with pass rate strictly between 0 and 1** | TMax: easy data wasted 7× the rollouts [Lit] |
| Model selection | On **dev only**; the frozen test split is run **once**, at the end | [Inf] |

Expected effect: +2–5 pp in-distribution on the repo-task split, smaller transfer. [Inf]

### Stage 4b: Fallback, rejection-sampling fine-tuning on base's own successes (10–25 GPU-h)

- Sample base at T = 1.0 on the training tasks and keep verified successes, plus error→recovery steps with the failed step masked from the loss.
- LoRA SFT at a low LR on this on-policy data, mixed with base-generated general replay (sweep 10–30%).
- This stays close to the base distribution, so forgetting is limited (RL's Razor: forgetting follows the KL distance from base). [Lit]

### Stage 5: Ship only what was measured (15–30 GPU-h)

- **Merge at strength 1.0 only.** If 1.0 is unstable, the run failed; do not dilute it.
- **Quants:** Q8_0, Q6_K and Q4_K_M each go through the dev agent suite, BFCL-V4 and a 100-task test slice at k ≥ 3, paired against BF16. Log parse failures and loop metrics. Measure KLD on transcripts that are **disjoint from the imatrix data**. IQ4_XS/IQ3_M are labelled **experimental** unless they pass.
- **Abliteration:** off the main release. If you ship a variant:
  - estimate one direction from hundreds of prompt pairs;
  - select it under KL < 0.1;
  - apply it to all residual-writing matrices, including GDN `out_proj`;
  - score refusal with a classifier, not keywords;
  - run harm and insecure-code checks, since abliterated coder models wrote SQL-injectable code 88–97% of the time [Lit].
- **Model card:** generated from the run manifest and gate JSON. Report thinking and non-thinking separately, with k, CIs and the sampler. No "frontier", "dominance" or "lossless" language unless measured.

---

## 5. Repo change list (concrete)

| File | Change |
|---|---|
| `eval/agentloop/harness.py` | Sampler flags (default: Qwen recommended), loop guard, k-run support, per-task JSONL streaming, oscillation metric |
| `eval/agentloop/schemas.py` | `loop_metrics()` (consecutive + A-B-A-B + duplicates) |
| `eval/repotasks/` (new) | Task builder, venv sandbox runner, hidden-test grader, dev/test split files with hashes |
| `eval/biochem_hard/eval_biochem_hard.py` | Retire, or fix max_tokens / IDs / storage / extractor |
| `pipeline/gates.py` | Single pre-registered gate; output includes code SHA; paired bootstrap + McNemar + Holm |
| `pipeline/ckpt_gate.py` | Select on dev split only; remove opencode from `fast_score` |
| `pipeline/rl_grpo.py` (new) | Single-GPU LoRA GRPO loop with the Stage 4 settings, Drive-resumable |
| `pipeline/mine_loop_pairs.py` (new) | DeLoop-style pair mining from base rollouts |
| `pipeline/abliterate.py` | If kept: correct layer lookup for the multimodal checkpoint, GDN `out_proj`, more prompts, KL/refusal checks; **fail loudly** |
| `pipeline/run_v4.sh` (new) | Stages 0–5 with markers; **no silent fallbacks** (`|| echo` on critical stages is banned) |
| `eval/kl_monitor.py` (new) | KL(V4 ‖ base) on a fixed general prompt set, logged during training (§1.2 Lever 1) |
| `eval/failure_taxonomy.py` (new) | Bucket failed trajectories (§1.2 Lever 2) |
| `eval/scoreboard.py` (new) | Builds the §1.4 release table from per-task JSONL with paired bootstrap / McNemar / Holm |
| `pipeline/publish_hf.py` | Card generated from measured JSON only; remove hard-coded claims |
| `pipeline/templates/` | Template tested for tool calls after `</think>` under llama.cpp `--jinja` |

---

## 6. What not to do

- Do not scale off-policy SFT data 10× and expect gains on this base. [Lit]
- Do not use fractional merge strength as a stabilizer. [Inf]
- Do not select checkpoints on any split you report as held-out. [Repo]
- Do not evaluate at T = 0 only. [Lit]
- Do not claim SWE-bench Verified or Terminal-Bench scores. Colab cannot run their Docker runners, and SWE-bench Verified was flagged as contaminated in Feb 2026. [Lit/Inf]
- Do not let any pipeline stage fail silently. [Repo]

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hybrid DeltaNet + LoRA unsupported in the vLLM/TRL/Unsloth RL stack | Medium–High | Stage 4 infrastructure test first; fallback to Stage 4b; llama.cpp server for rollouts |
| RL collapse or reward hacking | Medium | FP32 head, zero-variance filter, early stop at ≤ 300 steps, hidden read-only tests, diff audits |
| Colab disconnects or compute-unit exhaustion | High | Resumable state every 20 steps; per-task JSONL; stage markers |
| Gain smaller than what the test split can detect | Medium | 300–400 tasks × k = 3; pre-registered endpoint; report the null honestly |
| Base already near ceiling on in-house suites | Medium | Primary endpoint is the new repo-task split, not AgentLoop |

---

## 8. Two cheap experiments that could reorder everything

Run both before any training; each takes 1–2 Colab sessions.

1. **Recommended sampling vs. T = 0** for base and V3 on the dev set. If loops vanish, Stage 3 shrinks and the regressions may partly disappear.
2. **V3 merge-strength sweep, with and without abliteration.** This shows whether SFT or abliteration caused the regressions.

V4 is a success if it is no worse than base on every check, measurably better on one endpoint fixed in advance, and free of the loop and tool-call-inside-`<think>` failures that hurt local users. That would be the first Threnody release whose claim holds up to scrutiny.
