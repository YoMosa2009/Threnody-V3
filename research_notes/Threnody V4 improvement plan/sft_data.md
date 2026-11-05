# Agentic-Coding SFT Data Strategy for Small (7–14B) Open Models (Threnody V4)

Scope note: about 16 search and fetch calls. Most numbers come from arXiv abstracts or HF dataset/model cards. Full-paper tables were not read unless stated. "(older)" marks pre-2025 work. Today is 2026-11-05.

Name check:
- **nvidia/Open-SWE-Traces exists.** HF card: ~1.14M rows over v1.0 to v2.0. The v1.2 split was distilled from **Qwen3.8-27B (xhigh reasoning)**, and the paper is arXiv 2606.16038.
- **Qwen3.5 exists.** Qwen3.5-122B is named as a teacher in the Open-SWE-Traces abstract. Qwen3.5-9B itself was not checked.
- **Not verified:** "OmniAgent-Bench", "AgentLoop" and "AgentLoop-Hard" were not searched. Treat them as internal or unverified.

## 1. Trajectory counts, token budgets and data-scaling evidence

### Takeaway
Every published small or mid-size agentic-SWE SFT success uses **thousands to tens of thousands of trajectories**. Two independent studies (Skywork-SWE, Kwai-Klear) show **log-linear gains with trajectory count and no saturation**. V3 used 662 trajectories / 2.9M tokens, about 8% of its own 35M-token target, which sits at the very bottom of these curves. Most published gains come from data an order of magnitude larger.

### Cited Findings
- **SWE-smith (2025).** 5,016 Claude-3.7-Sonnet trajectories (SWE-agent), each task capped at ≤3 trajectories, fine-tuning Qwen2.5-Coder-32B-Instruct. Result: **40.2% SWE-bench Verified** pass@1. The scaling figure runs from 100 to ~6.2k trajectories, with labelled points 14.3 → 27.8 → 35.4 → 40.2. The pool was 50k instances from 128 repos, and expert trajectories were attempted on 8,686 instances. — [SWE-smith arXiv 2504.21798](https://arxiv.org/html/2504.21798v2)
- **Kwai-Klear (2025).** 65,994 mini-swe-agent-plus trajectories on SWE-smith tasks. The solve rate "improves approximately linearly with the logarithm of the data scale" from 1k to 66k trajectories. **Qwen3-8B → 39.0% Verified** (Klear-Agent-8B-SFT). The teacher is not stated. — [HF dataset card](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k)
- **Skywork-SWE (Jun 2025).** 10,169 instances, 8,209 used for training, on Qwen2.5-Coder-32B with OpenHands (100-turn cap). Results: **38.0% Verified**, and 47.0% with test-time scaling. Log-linear trend with R²≈0.992 up to ~8k trajectories and "no sign of saturation". — [arXiv 2506.19290](https://arxiv.org/html/2506.19290v1)
- **SWE-Lego (Jan 2026).** 32k task instances and 18k validated trajectories, real plus synthetic. Results: **Qwen3-8B 42.2%** and Qwen3-32B 52.6% Verified, SFT only. With a verifier at TTS@16: 8B goes to 49.6% and 32B to 58.8%. — [arXiv 2601.01426](https://arxiv.org/abs/2601.01426)
- **SWE-Mirror (Sep 2025).** 6,431 Claude-3.7/4-Sonnet successful trajectories plus 6,025 SWE-rebench trajectories, 12,456 in total. **Qwen2.5-Coder-7B → 22.8% Verified**. The 22.8 comes from secondary tables; the abstract states "+21.8% resolve rate" for 7B. — [arXiv 2509.08724](https://arxiv.org/pdf/2509.08724); [Klear card table](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k)
- **SWE-Dev (2025, THUDM).** Scaled agent trajectories; **7B 23.4%** and 32B 36.6% Verified. — [Kimi-Dev/SWE-Dev search summary; SWE-Dev repo THUDM/SWE-Dev](https://arxiv.org/abs/2509.23045?context=cs) (exact trajectory count not verified)
- **Kimi-Dev (2025).** About 5k public trajectories as an SFT adaptation on top of an Agentless-trained 72B prior gives 48.6% as an SWE-agent. Shows that a strong skill prior lowers the number of trajectories needed. — [Kimi-Dev arXiv 2509.23045](https://arxiv.org/pdf/2509.23045)
- **SWE-Gym (Dec 2024, older).** Only **491** trajectories on Qwen2.5-Coder-32B gave 20.6% Verified, and performance was still scaling at 491. — [arXiv 2412.21139](https://arxiv.org/html/2412.21139v2)
- **NVIDIA corpora.** Nemotron-SWE-v1 has **59k** OpenHands trajectories from Qwen3-Coder-480B. Nemotron-SFT-SWE-v2 has **~46k** OpenHands trajectories plus an agentless subset (localization, repair, test generation). — [Nemotron-SWE-v1](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1); [Nemotron-SFT-SWE-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2)
- **Open-SWE-Traces (2026).** The paper version covers 207,489 trajectories in 9 languages from 20k PRs, using OpenHands and SWE-agent. Teachers are MiniMax-M2.5 (thinking) and Qwen3.5-122B (non-thinking). Student: Qwen3-30B-A3B Thinking/Instruct/Coder. Best result: **61.7% Verified, 57.1% Multilingual, 36.8% SWE-bench Pro**. The abstract gives no baseline deltas. — [arXiv 2606.16038](https://arxiv.org/abs/2606.16038)
- **Open-SWE-Traces HF card.** Splits: v1.0 151k (cleaned from ~207k by removing "git hacking"), v1.1 263k (DeepSeek-V4-Flash-0731 and Qwen3.6-27B), v1.2 235k (**Qwen3.8-27B xhigh**), v2.0 489k (Qwen3.8-Flash-Next at xhigh/medium/low effort). v2.0 is split across 5 harnesses: claudecode 99.4k, opencode 101k, openhands 82.3k, sweagent 104k, minisweagent 103k. Has a `resolved` field (1/0/-1). License is CC BY 4.0. No token counts or benchmark results on the card. — [HF card](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
- **LoRA curation study (2026).** Qwen2.5-Coder-7B with LoRA on SWE-trajectory (67,074 trajectories, 32,161 resolved). Going from 500 to 1,000 trajectories cut held-out CE by ~12.7%. Quality filtering vs random selection gave <1% difference at that scale (p>0.10), widening to 3.6% only at 2,000 trajectories. The authors used CE because **7B models reach near-zero SWE-bench resolve rates** at these scales. — [arXiv 2607.17205](https://arxiv.org/abs/2607.17205)

### Inferences
- Published 7–9B successes need about **12k–66k trajectories**: SWE-Mirror-7B 22.8%, SWE-Dev-7B 23.4%, Klear-8B 39%, SWE-Lego-8B 42.2%. On log-linear curves, 662 trajectories sits in the region where SWE-smith's curve gives ~14% even at 32B. A +1 task gain is consistent with being at the bottom of the curve.
- The 35M-token target is plausible but probably still small. Assuming ~20–50k tokens per SWE trajectory (not verified from a source), 35M tokens is roughly 1–2k trajectories. Matching Klear or SWE-Lego scale would mean several hundred million tokens. **Priority #1 for V4 is volume of agentic trajectories**, which public corpora (Open-SWE-Traces, Nemotron-SWE, SWE-smith-66k, SWE-Lego) make cheap.
- Open-SWE-Traces v1.2 uses the same teacher as V3 (Qwen3.8-27B), so V3 likely drew a tiny slice of an available ~235k pool.

### Gaps
- Per-trajectory token counts for these corpora were not found, so token budgets are estimates.
- No found paper sweeps data scale specifically for 9B-class Qwen3.5 students, and none with LoRA rank 64 vs full fine-tuning.
- The Open-SWE-Traces full-paper ablations (scale, harness, mode) were not read.

## 2. Verified-only vs failures; length; resolved filtering; dedup; decontamination

### Takeaway
The default recipe is rejection sampling: keep only resolved trajectories and cap trajectories per task (SWE-smith ≤3). The 2026 consensus is that **step-level handling beats whole-trajectory dropping**: mask the loss on bad steps but keep them in context. Failed trajectories add value when they are converted or annotated, not when they are naively imitated.

### Cited Findings
- **APIGen-MT (2025).** Rejection sampling keeps only trajectories that achieve the goal. The release is 5k τ-bench-domain trajectories (APIGen-MT-5k). — [arXiv 2504.03601](https://arxiv.org/abs/2504.03601)
- **SWE-smith.** At most 3 trajectories per task instance, to avoid over-representation (a form of deduplication). — [arXiv 2504.21798](https://arxiv.org/html/2504.21798v2)
- **Step Rejection Fine-Tuning (ICML 2026).** A critic LLM labels erroneous steps. Their loss is masked, but they **stay in context**, so the model "learns to recover from errors without reproducing them". Trajectory-level RFT (dropping unresolved trajectories) gave +2.4% on SWE-bench Verified; SRFT gave **+3.7%, reaching 32.2%**. — [arXiv 2605.10674](https://arxiv.org/abs/2605.10674)
- **SWE-Lego.** Uses step-level error masking: steps with failed tool calls are excluded from the loss rather than whole trajectories being dropped. It also uses a difficulty curriculum. — [arXiv 2601.01426](https://arxiv.org/abs/2601.01426); description via [curation search results](https://arxiv.org/pdf/2608.27449)
- **Microsoft debug-gym (Aug 2026 blog).** Failed MiniMax-M2.5 trajectories on SWE-rebench are summarized, and only facts checked as "Wrong" against the gold patch are kept. These are added as privileged info in the prompt (removed at eval) and mixed with positive trajectories. Gains over positive-only SFT: **+4.1 on Qwen3-32B (to 58.05 Verified, <4.5k trajectories)** and **+2.8 on Qwen3-4B (<37k trajectories)**. This is a blog post and is not peer-reviewed. — [debug-gym blog](https://microsoft.github.io/debug-gym/blog/2026/08/negative-pi/)
- **Open-SWE-Traces.** Ships a `resolved` flag rather than pre-filtering. v1.0 removed ~56k "git hacking" trajectories (reward hacking via git), going from 207k to 151k. — [HF card](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
- **Licensing hygiene.** Open-SWE-Traces filtered SWE-rebench-V2 to permissive licenses (MIT, Apache, BSD). — [arXiv 2606.16038](https://arxiv.org/abs/2606.16038)
- **Length.** Skywork-SWE training and eval used OpenHands with a 100-round cap. — [arXiv 2506.19290](https://arxiv.org/html/2506.19290v1). SWE-Next reports **87.4% of failures end by hitting the step budget**, not by crashing. — [arXiv 2603.20691](https://arxiv.org/pdf/2603.20691)

### Inferences
- For V4:
  - Use `resolved==1` as the primary filter.
  - Cap trajectories per task at 2–3.
  - Drop reward-hacking patterns: git checkout or reset of tests, editing test files.
  - Prefer **loss-masking bad steps** over discarding whole trajectories.
- Unresolved trajectories should go in only with masking or annotation. Do not feed them in as plain imitation targets.
- Decontamination: the task sources here (SWE-smith, SWE-rebench-V2, Scale-SWE) are separate from SWE-bench repos and instances by design. V4 should still drop any instance whose repo and issue matches SWE-bench Verified, Multilingual or Pro. **V4 should also hash-check against the 14-task held-out harness and the AgentLoop / OmniAgent tasks.**

### Gaps
- No found decontamination statistics, such as overlap rates with SWE-bench, for these corpora.
- No direct ablation of trajectory-length truncation vs dropping long trajectories for small students.

## 3. Does filtering out repeated actions hurt recovery? Error-recovery data

### Takeaway
No source tested V3's exact rule ("drop any trajectory with an immediately repeated action"). However, the best-supported practice is to **drop only true loops**: repeated actions with no change of strategy that never recover. Keep trajectories where the agent hits an error and then changes approach, and mask the erroneous steps. A blanket repeated-action filter likely removed exactly the error→retry→recover demonstrations that teach recovery. It also skews the data toward easy, short, first-try successes, which is a plausible contributor to the AgentLoop regressions (inference).

### Cited Findings
- **LiteCoder-Terminal (2026).** An LLM judge drops looping trajectories and trajectories with only tiny syntax tweaks and no change of approach. It **explicitly keeps trajectories where the agent switches tool or strategy after an error**. Teachers: MiniMax M2/M2.1. Scaffolds include Claude Code and OpenHands. — [arXiv 2605.29559](https://arxiv.org/pdf/2605.29559)
- **SRFT.** Keeping error steps in context, with masked loss, is designed to teach recovery and beats dropping whole trajectories (+3.7 vs +2.4 on Verified). — [arXiv 2605.10674](https://arxiv.org/abs/2605.10674)
- **LoRA 7B curation.** Error-retry rate (which penalizes *repeated errors*) was the dominant quality sub-signal; using it alone matched the full composite (Δ<0.2%). This mattered only at ≥2k trajectories. Below that, quantity beat filtering. — [arXiv 2607.17205](https://arxiv.org/abs/2607.17205)
- **SWE-agent (NeurIPS 2024, older).** Failed-edit recovery was 23.4% of unresolved GPT-4-Turbo runs, and the odds of recovery fall as failed edits accumulate. — [SWE-agent paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf)
- **Guided-OPD (2026).** Small-student errors compound across turns and push rollouts off the teacher's state distribution. Gains over vanilla OPD are larger for smaller students. — [arXiv 2606.15912](https://papers.cool/arxiv/2606.15912)

### Inferences
- Replace V3's filter with a graded rule:
  - (a) Drop trajectories with ≥N identical consecutive actions *and* identical observations (true loops), or with no recovery.
  - (b) Keep "action repeated after an environment change" cases, such as re-running tests after an edit. These are legitimate and were likely over-filtered in V3.
  - (c) Loss-mask failed tool calls and erroneous steps.
- Re-running `pytest` after an edit is an "immediately repeated action" by naive string match. V3's filter may have systematically removed the test→edit→retest loop. Check how many V3 rejects fit this pattern.
- Error-retry rate is a better quality signal than a binary repeat flag, but it pays off only once there are thousands of trajectories.

### Gaps
- No direct experiment measuring loop rate or recovery rate in a small model as a function of repetition filtering.
- No found study isolating "recovery trajectories" as a data slice with a measured benchmark delta for a 7–14B model.

## 4. Mixing general / instruction / reasoning data to prevent forgetting; thinking vs non-thinking

### Takeaway
Narrow agentic or tool SFT measurably erodes general ability in small models. The standard mitigation is rehearsal (replay of general or original-distribution data) plus forgetting-aware losses. No validated universal replay ratio was found. Small, high-quality agentic sets can also *transfer* to general function calling, so V3's regressions on agent benchmarks point to distribution narrowness (one harness, a small set, filtered behaviours) rather than agentic data per se.

### Cited Findings
- **EAFT (2026).** Standard SFT of Qwen3-4B-Instruct on a Nemotron-Agentic-Tool-Use-v1 subset dropped general capability from **81.1 to 74.8**. The entropy-adaptive loss held it at 77.5 with near-equal target performance. — [arXiv 2601.02151](https://arxiv.org/pdf/2601.02151)
- **Rehearsal.** Mixing general data is common practice, but its effectiveness "depends entirely on the quality and similarity" to the model's original instruction-tuning data. SelfAug used a 1:9 MAGPIE:task mix (a method-specific setting). — [SelfAug, EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.763.pdf)
- **"SFT or RL for Tool-Calling Agents?"** Reports that dataset mixing gives consistently strong transfer while staying close to in-distribution specialist performance. Seen only as an abstract-level summary; ratios not checked. — [search summary, HF papers](https://huggingface.co/papers?q=SFT)
- **Goedel Prover.** Just 100 Lean-specific agentic traces improved general BFCL function calling, i.e. positive transfer from small agentic sets. — [arXiv 2604.08388](https://arxiv.org/pdf/2604.08388)
- **Dual-mode distillation.** Open-SWE-Traces trains on both thinking traces (MiniMax-M2.5) and non-thinking traces (Qwen3.5-122B) across Thinking, Instruct and Coder students. v2.0 also varies the teacher's reasoning effort (xhigh/medium/low). — [arXiv 2606.16038](https://arxiv.org/abs/2606.16038); [HF card](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)

### Inferences
- For V4, replay should be **self-distilled from the base Qwen3.5-9B**: the base model's own responses on general, tool-calling and agent prompts. Per the rehearsal finding, this matches the original distribution best. Also add BFCL-style function-calling data.
- A starting ratio of ~10–30% replay by tokens is a reasonable sweep range, **not an evidence-backed optimum**.
- Bio/chem data gave +2.5 pp, so keep it, but cap it so it does not dilute agentic tokens.
- Match the thinking mode in training data to the mode used at eval for each benchmark. Include both modes if the harness toggles `enable_thinking`. Varying reasoning effort (as in Open-SWE-Traces v2.0) may help keep traces short enough for a 9B context.
- With LoRA r=64 on a tiny set, the forgetting could reflect format or behaviour overwrite rather than knowledge loss. A held-out eval on the base model's own agent harness after each data change would localize this.

### Gaps
- No found ablation of replay ratios specific to agentic-coding SFT at 7–14B.
- No head-to-head result on mixing thinking and non-thinking traces at small scale (Open-SWE-Traces ablations not read).

## 5. Multi-harness / multi-schema tool-call diversity and unseen-schema generalization

### Takeaway
2026 corpora deliberately span many harnesses. Open-SWE-Traces v2.0 has roughly 100k trajectories each across Claude Code, OpenCode, OpenHands, SWE-agent and mini-swe-agent. Function-calling datasets (APIGen-MT) train on verified multi-turn trajectories across varied APIs. Direct quantitative evidence that harness diversity improves unseen-harness performance for 7–14B models was not found within budget.

### Cited Findings
- **Open-SWE-Traces v2.0.** 5 harness splits: claudecode 99.4k, opencode 101k, openhands 82.3k, sweagent 104k, minisweagent 103k. — [HF card](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
- **APIGen-MT / xLAM-2.**
  - Multi-turn trajectories are generated via simulated agent-human interplay, with APIs modelled as dependency graphs and rejection sampling applied.
  - Released data: 5k trajectories with ~12 turns on average for long tasks (GPT-4o).
  - Models are 8B–70B (Llama-xLAM-2-8b-fc-r). The authors claim the small models beat larger ones in multi-turn settings; xLAM-2-70b-fc-r scores 78.19% on BFCL v3.
  - — [arXiv 2504.03601](https://arxiv.org/abs/2504.03601); [HF model card](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r)
- **LiteCoder-Terminal.** SFT data comes from multiple scaffolds, including Claude Code and OpenHands. — [arXiv 2605.29559](https://arxiv.org/pdf/2605.29559)
- **Nemotron-SFT-SWE-v2.** Pairs agentic OpenHands trajectories with agentless subtasks (localization, repair, test generation). — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2)

### Inferences
- V3 used a single "Open-SWE-Traces-style" harness plus a tool-schema/identity slice, then regressed on *other* agent harnesses (OmniAgent, AgentLoop). That pattern is consistent with schema and harness overfitting.
- V4 should draw from several harness splits.
- V4 should also randomize tool names, argument orders and system prompts on part of the data (schema augmentation). This is a recommendation; no measured effect was found.
- Include APIGen-MT-style or BFCL-style function-calling data, plus "no tool needed / ask for clarification" negatives (When2Call-type), to protect tool-call precision.

### Gaps
- ToolACE, When2Call and Hermes function-calling dataset sizes and effects were not verified in this pass.
- No ablation found of harness diversity → unseen-harness score at 7–14B.

## 6. Teacher choice and on-policy vs off-policy distillation

### Takeaway
Strong frontier or large-open teachers produce the best SFT data. Examples: Claude-3.7/4-Sonnet for SWE-smith and SWE-Mirror, Qwen3-Coder-480B for Nemotron-SWE-v1, MiniMax-M2.5 and Qwen3.5-122B for Open-SWE-Traces. The 2026 literature treats off-policy SFT as a warm start. On-policy distillation (OPD) or RL then fixes compounding multi-turn errors, and the gains are larger for smaller students.

### Cited Findings
- **Teachers by project.**
  - SWE-smith: Claude-3.7-Sonnet. — [arXiv 2504.21798](https://arxiv.org/html/2504.21798v2)
  - SWE-Mirror: Claude-3.7/4-Sonnet. — [arXiv 2509.08724](https://arxiv.org/pdf/2509.08724)
  - Nemotron-SWE-v1: Qwen3-Coder-480B-A35B. — [HF](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1)
  - Open-SWE-Traces: MiniMax-M2.5 (thinking) and Qwen3.5-122B (non-thinking) per the paper; Qwen3.8-27B for HF v1.2; Qwen3.8-Flash-Next for v2.0. — [arXiv 2606.16038](https://arxiv.org/abs/2606.16038); [HF card](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
- **Guided-OPD.** Small-student errors compound across turns. Mixing teacher and student turns, with decaying teacher intervention, beats vanilla OPD on ALFWorld, ScienceWorld and WebShop (Qwen3 students), with larger gains for smaller students. — [arXiv 2606.15912](https://papers.cool/arxiv/2606.15912)
- **SOD.** Erroneous tool calls cascade and amplify student-teacher divergence, so it uses step-wise reweighting. — [arXiv 2605.07725](https://arxiv.org/html/2605.07725v1)
- **ATOD.** OPD gains saturate as the student approaches the teacher, so it combines OPD with RL. — [arXiv 2606.27814](https://arxiv.org/abs/2606.27814)
- **ReOPD.** An offline variant that reuses teacher trajectories and needs no tool calls during training, ≥4× faster per step than OPD. Flags a "prefix trap". Listed under two titles across sources. — [arXiv 2607.04763](https://arxiv.org/abs/2607.04763)
- **DGPO (ACL 2026).** For 0.5–1B students, both RL and plain distillation are weak. It uses cold-start SFT followed by teacher-guided policy optimization. — [ACL 2026](https://aclanthology.org/2026.acl-long.1751.pdf)
- **No head-to-head.** No direct OPD vs SFT number on the same multi-turn SWE benchmark was found. — (search summary, above sources)

### Inferences
- Qwen3.8-27B is a reasonable teacher, and its traces are already public at scale (Open-SWE-Traces v1.2, 235k). Adding a second, stronger or differently-styled teacher may improve coverage. Examples: the MiniMax-M2.5 thinking traces, or Qwen3.8-Flash-Next at low/medium effort, which gives shorter traces suited to a 9B context.
- V4 plan: (1) large off-policy SFT with resolved filtering, step masking and replay; then (2) an on-policy distillation or RL pass in the target harnesses to address the AgentLoop-type regressions caused by compounding errors.

### Gaps
- No quantitative study found of teacher-size effects on 7–14B SWE students (e.g. 27B vs 480B teacher at a fixed trajectory count).
- No OPD result found specifically on SWE-bench-style coding for 7–14B students.
