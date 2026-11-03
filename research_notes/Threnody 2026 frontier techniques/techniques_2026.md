# 2026 Frontier Post-Training Techniques for a Small (~9B) Agentic Coding Model on 1x A100 80GB (~22 GPU-h)

Scope: Qwen/Qwen3.5-9B student, closed-loop agent eval baseline 12/14 (failures: "declares done without verifying", "fixes symptom not root cause"). Research date 2026-11-02. All claims below carry a source; items marked "Inference" are my reasoning, not reported results. Many 2026 papers were seen only via abstract/search snippet; this is flagged where relevant.

## 1. On-policy distillation (OPD): recipe, efficiency, tooling, single-A100 feasibility

### Takeaway
OPD (student samples, teacher scores each token, per-token reverse KL as the advantage) is the best-documented, cheapest way to get large gains on a small model, and 2026 work has specialised it for multi-turn agents. TRL now ships it as a top-level `DistillationTrainer` with vLLM student generation and an external vLLM teacher server. On one A100 the bottleneck is fitting student training, student sampling and a 27B teacher at once. The workable pattern is phased: sample, then score, then train.

### Cited Findings
- **Core recipe (Thinking Machines, Oct 2025).** The loss is per-token reverse KL(student‖teacher) on student-sampled tokens. It is implemented as RL with per-token advantage = −reverse KL, an importance-sampling loss, and discount 0 (discounts >0 did not help). Partial rollouts are fine because no episode needs to finish. Defaults were 64 prompts × 4 samples per batch. The math run took about 150 steps to reach 70% AIME'24 from an SFT checkpoint. — [Thinking Machines: On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **Reported efficiency.**
  - Versus continuing SFT: about 9–30x cheaper.
  - Versus RL: 7–10x fewer gradient steps and about 50–100x less compute overall.
  - Qwen3-8B-Base student with a Qwen3-32B teacher: OPD reached 74.4% AIME'24 / 63.3% GPQA-D in 1,800 GPU-h. RL in the Qwen3 report reached 67.6% / 61.3% in 17,920 GPU-h.
  — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **LoRA.** At rank 32, LoRA trailed full fine-tuning by 13% after SFT but only by 6% after OPD, so OPD is relatively LoRA-friendly. The blog notes that the Tinker cookbook now uses Qwen3.5-9B-Base as student and Qwen3.5-9B as teacher. — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- **Tokenizer.** The method requires the teacher to give logprobs on the student's tokens, so tokenizers must match. The Thinking Machines blog does not state this explicitly. TRL GOLD supports cross-tokenizer distillation via `use_uld_loss=True` plus `teacher_tokenizer_name_or_path`, with byte-offset alignment in later releases. — [TRL distillation docs](https://huggingface.co/docs/trl/en/distillation_trainer); [TRL GOLD docs mirror](https://huggingface-gw-kz.just-ai.com/docs/trl/gold_trainer)
- **TRL implementation.**
  - `DistillationTrainer` moved from `trl.experimental.distillation` to top-level `trl`. GOLD stays experimental and inherits on/off-policy mixing from `experimental.gkd.GKDTrainer`.
  - vLLM server mode works as follows: `VLLM_SERVER_DEV_MODE=1 vllm serve ... --logprobs-mode processed_logprobs --max-logprobs -1`, with `use_vllm=True, vllm_mode="server"`.
  — [TRL docs](https://huggingface.co/docs/trl/en/distillation_trainer); [releases.sh TRL](https://releases.sh/hugging-face/trl?page=3)
- **TRL blog, Apr 12 2026 ("Distilling 100B+ models 40x faster").**
  - Config: `loss_top_k=1`, `use_vllm=True`, `use_teacher_server=True`, `teacher_model_server_url`.
  - Speedups: a generation buffer gives 41.7x at 64 grad-accum steps; teacher-server request batching cuts p99 latency about 10x; binary logprob encoding makes payloads about 5x smaller.
  - Qwen3-4B no-think student gained more than 39 points on AIME25.
  - A 235B teacher was about 10 points stronger than a 30B teacher, but the students ended similar, so a bigger teacher is not necessarily better.
  - GPQA-D degraded about 10% after step 40 with the 235B teacher.
  — [HF TRL distillation post](https://huggingfacetb-trl-distillation-trainer.hf.space/)
- **2026 multi-turn agentic OPD follow-ups.**
  - **STRIDE** (arXiv 2609.14636, Sep 2026). On τ²-bench, teacher signal concentrates in each turn's prefix, and teacher endorsement collapses at the student's first erroneous action. The method has two parts: early-stop a rollout when cumulative teacher logprob drops below an OOD threshold, and keep a prefix buffer that restarts at the weakest correct turn. Reported speedups over full OPD are 3.73x on τ²-retail (matching full OPD and beating the 30B teacher), 4.51x cross-domain multi-teacher, and 5.10x on AIME25. — [arXiv 2609.14636](https://arxiv.org/abs/2609.14636)
  - **SOD (step-wise OPD for small agents)** (arXiv 2605.07725). Scales distillation strength per step by step-level divergence. Its 0.6B student reaches 26.13% AIME25 with tool-integrated reasoning. — [arXiv 2605.07725](https://arxiv.org/abs/2605.07725)
  - **ATOD** (arXiv 2606.27814). Annealed, turn-aware hybrid OPD for multi-turn agentic tasks, aimed at the plateau that appears once the student nears the teacher. — [arXiv 2606.27814](https://arxiv.org/abs/2606.27814)
  - Related (snippet-level only): "When Teacher Guidance Misleads: Reward-Aligned OPD" (2608.27960); "Look Ahead Before You Distill" (2608.01953); a survey of OPD (2604.00626); and "Distilling Agentic Systems" roadmap (2609.36630). — [2608.27960](https://arxiv.org/pdf/2608.27960); [2608.01953](https://arxiv.org/pdf/2608.01953); [2604.00626](https://arxiv.org/html/2604.00626v3); [2609.36630](https://arxiv.org/html/2609.36630v1)
- **Community evidence.** A Hugging Face checkpoint "Qwen3.5-9B-Coding-MixedOPD-TMAX27B" applies mixed OPD from a 27B teacher to Qwen3.5-9B on SWE-rebench/SWE-bench tasks. Results are on a 150-task custom harness subset, not official. — [HF CharlieLLL](https://huggingface.co/CharlieLLL/Qwen3.5-9B-Coding-MixedOPD-TMAX27B-u109)
- **Teacher.** Third-party write-ups describe Qwen3.8-27B as follows: released Aug 14 2026, Apache-2.0, dense 27B hybrid Gated DeltaNet + Gated Attention, 262K context, FP8 variant `Qwen/Qwen3.8-27B-FP8`, and a reported 61.7% SWE-bench Pro (vendor number). I found no primary HF card in search, so these are secondhand. — [morphllm](https://www.morphllm.com/qwen-3-8); [datanorth](https://datanorth.ai/news/alibaba-releases-qwen3-8-27b)

### Inferences
- **Memory on one A100 80GB (estimate, not measured).**
  - Student 9B bf16 weights: about 18 GB, plus LoRA, optimizer state, and activations with gradient checkpointing at 16K.
  - Teacher 27B FP8: about 27–28 GB of weights (bf16 would be about 54 GB).
  - All three components co-resident (student vLLM, student trainer, teacher vLLM) do not fit comfortably at long context. A100 (Ampere) also lacks FP8 tensor cores, so vLLM would run FP8 as weight-only (Marlin-style) kernels, slower than on H100. This is from general knowledge, not verified here.
- **Phased loop to use instead (time-multiplex the GPU), per round:**
  1. vLLM serves the student plus LoRA to generate N multi-turn rollouts in our harness.
  2. Unload the student, load the 27B teacher in vLLM, and compute prompt logprobs on those exact token sequences (prefill only, cheap, no decoding).
  3. Unload the teacher and train LoRA on per-token reverse KL with masked tool outputs.

  This is "slightly off-policy" OPD with stale-by-one-round samples, similar to what the TRL generation buffer already does. Teacher scoring is prefill-only, so it should cost far less than teacher decoding.
- **Check tokenizer compatibility first.** Before committing, verify the Qwen3.5 and Qwen3.8 tokenizers and chat templates are byte-identical, especially tool-call and `<think>` tokens. If they differ, use GOLD/ULD or fall back to sequence-level distillation (SFT on teacher trajectories).
- **Target the failure modes with STRIDE-style signal.** Because teacher endorsement collapses at the first erroneous action, per-token OPD should directly penalise tokens like a premature "done" or a symptom-only edit, provided the teacher itself verifies. This makes OPD a better fit for our failure modes than plain SFT.

### Gaps
- No found paper reports OPD wall-clock on a single A100 with a 27B teacher. The budget above is my estimate.
- I did not read the full text of SOD, ATOD, or STRIDE, so hyperparameters (LR, rollouts per step) are unknown.
- Qwen3.5 → Qwen3.8 tokenizer identity is unverified, and the Qwen3.8-27B primary model card was not retrieved.

## 2. Agentic SFT data scaling and quality for small models (2026)

### Takeaway
At 8B, SFT on agent trajectories keeps scaling with no plateau seen up to about 100K trajectories. Task source and diversity, and how task descriptions are generated, matter more than raw token count. Error-recovery and expert-augmented trajectories (RFT-filtered) give outsized gains for small models.

### Cited Findings
- **OpenThoughts-Agent** (arXiv 2606.24855, Jun 2026).
  - More than 100 controlled ablations; task sources and diversity are the key levers.
  - Qwen3-32B trained on 100K examples reaches a 44.8% average on 7 agentic benchmarks, +3.9 points over Nemotron-Terminal-32B.
  - The data beats other open datasets at every compute-controlled size.
  — [arXiv 2606.24855](https://arxiv.org/abs/2606.24855)
- **OpenThoughts-Agent with Qwen3-8B (from search snippet, full text not read).**
  - SWE-bench Verified-100 rises from 9.3% to 24.3% at 10K rows, 26.3% at 31.6K, and 39.7% at 100K.
  - The task-description generation strategy changes accuracy by up to 30 points.
  — [arXiv 2606.24855 HTML](https://arxiv.org/html/2606.24855v1)
- **Nemotron-Terminal** ("On Data Engineering for Scaling LLM Terminal Capabilities", arXiv 2602.21193, Feb 2026).
  - Qwen3 8B/14B/32B trained on synthetic Terminal-Task-Gen / Terminal-Corpus data.
  - Terminal-Bench 2.0 scores: 8B 13.0, 14B 20.2, 32B 27.4. The 32B beats Qwen3-Coder-480B (23.9).
  - Performance increases with data volume at 8B and 14B. Data and models are on HF (`nvidia/nemotron-terminal`).
  — [arXiv 2602.21193](https://arxiv.org/pdf/2602.21193)
- **SWE-Protégé** (arXiv 2602.22124).
  - Setup: a 7B SLM (Qwen2.5-Coder-7B) gets an `ask_expert` tool. SFT uses expert-augmented trajectories (Claude Sonnet with ground truth) filtered by RFT. RL adds stall penalties and expert-as-judge rewards.
  - Result: 42.4% SWE-bench Verified, versus the prior 7B SOTA of 17%.
  - Looping: trajectories with more than 10 repeated actions fell from 31.0% to 0.8%, and mean steps dropped from about 60 to about 20.
  — [alphaXiv 2602.22124](https://www.alphaxiv.org/abs/2602.22124)
- **SWE-Fuse** (arXiv 2603.07927). The Klear-Agent-8B-SFT baseline scores 39.4. SWE-Fuse-8B with SFT plus entropy-aware RLVR scores 43.0 with mini-SWE-agent-plus, so RL adds a few points on top of strong SFT at 8B. — [arXiv 2603.07927](https://arxiv.org/html/2603.07927v1)
- **SWE-Next** (arXiv 2603.20691). Higher pass@1 with fewer or comparable trajectories at 7B/14B, i.e. quality over volume. — [arXiv 2603.20691](https://arxiv.org/html/2603.20691v1)
- **Kimi-Dev** (arXiv 2509.23045). Agentless training (localize, edit, self-test) used as a skill prior before SWE-agent SFT and RL. — [arXiv 2509.23045](https://arxiv.org/pdf/2509.23045)
- **Nanbeige4.2-3B** (arXiv 2607.22083). A compact-model agentic-capabilities report (not read in detail). — [arXiv 2607.22083](https://arxiv.org/pdf/2607.22083)
- **Community reference point.** The fine-tune "Qwopus3.5-9B-coder" reports 53.33% SWE-bench Verified on a Qwen3.5-9B base. This is self-reported on the model card and unverified. — [HF Jackrong/Qwopus3.5-9B-coder](https://huggingface.co/Jackrong/Qwopus3.5-9B-coder)

### Inferences
- Our ~64M-token plan is in the right range: about 100K trajectories × ~600 tokens is lower than typical agent trajectories (often 5–25K tokens). Prioritise task diversity and verified-success filtering over more tokens.
- For our two failure modes, oversample trajectories with these properties:
  1. A reproduction or test is run after the edit and its output is inspected before finishing.
  2. A failed first fix is followed by root-cause investigation (error-recovery segments).
- Keep the 16K length unless the eval needs more. Truncating trajectories that end in verification removes exactly the behavior we want, so drop rather than truncate.

### Gaps
- I found no 2026 controlled ablation isolating "thinking before tool calls" or turn count per token for 8–9B models (the OpenThoughts-Agent full text likely has some of this, but was not read).
- I found no direct per-token comparison between error-recovery and clean trajectories.

## 3. Rejection sampling, expert iteration, and self-improvement without Docker

### Takeaway
RFT on verified trajectories (SWE-smith style) is the standard, proven step. Self-play bug-injection RL (SSR) adds about 10 points but uses containers. A Docker-free pytest-in-checkout sandbox has been shown to work on Qwen3.5-9B on a single GPU.

### Cited Findings
- **SWE-smith** (arXiv 2504.21798). Synthetic bugs come from LLM rewrites, AST mutation, PR reversal, and patch combination. RFT on this data gave SWE-agent-LM-32B 40.2% on SWE-bench Verified. The official pipeline builds Docker images. — [arXiv 2504.21798](https://arxiv.org/pdf/2504.21798)
- **Self-play SWE-RL (SSR)** (arXiv 2512.18552, ICML 2026). One model both injects and fixes bugs, trained with RL. It reports +10.4 on SWE-bench Verified and +7.8 on SWE-bench Pro, using containerized environments. — [arXiv 2512.18552](https://arxiv.org/abs/2512.18552v1); [PMLR](https://proceedings.mlr.press/v306/wei26x.html)
- **Docker-free single-GPU harness (community, Qwen3.5-9B)** — [GitHub agentic-rl-swe](https://github.com/d4wnchen-jpg/agentic-rl-swe)
  - Sandbox: check out the defective branch, apply the patch, and remove `.git` and anti-cheat files.
  - Grading: pytest with fail-to-pass (F2P) and pass-to-pass (P2P) tests, binary reward.
  - **Pitfall 1, chat template.** The native Qwen3.5 template breaks the token prefix chain across turns and leaves an empty think block. After training, 90.3% of rollouts repeated that block. A fixed template cut this to 0%.
  - **Pitfall 2, visible tests.** Visible test files lead to huge pytest output that gets truncated, and 99% of runs exhausted their turns. The fix is to hide the F2P tests and prompt the model to write its own minimal repro script.

### Inferences
- **Docker-free self-improvement loop for our setup:**
  1. Generate tasks by mutating functions in small pure-Python repos (SWE-smith-style AST/LLM mutation) with F2P tests in a venv.
  2. Sample k=4–8 student rollouts per task with vLLM.
  3. Keep only passing rollouts in which the agent itself ran a check before finishing.
  4. SFT on these (RFT/STaR) for 1–2 rounds.
- The "write your own repro script" prompt is a direct fix for "declares done without verifying". Bake it into the RFT filter.

### Gaps
- I found no 2026 paper that reports RFT gains specifically for Docker-free sandboxes. The evidence is community-level only.

## 4. RL for agents at small scale (GRPO-family) on one A100

### Takeaway
Multi-turn GRPO with LoRA on Qwen3.5-9B runs on one 96 GB GPU, but the only public attempt saw within-noise held-out gains after 20 steps / 640 rollouts and estimated it needed about 24x more training. In 22 GPU-hours, full agentic RL is the lowest-yield option. Shaped rewards (stall penalties, verification) and loop penalties are the parts worth borrowing.

### Cited Findings
- **Single-GPU GRPO on Qwen3.5-9B (community)** — [GitHub agentic-rl-swe](https://github.com/d4wnchen-jpg/agentic-rl-swe)
  - Setup: 96 GB RTX PRO 6000 (about 43 GB vLLM + 45 GB training). LoRA, group size 8, up to 40 turns.
  - Each rollout is packed into one ~25K-token sequence, about 18x cheaper than one row per turn.
  - Training: 20 steps / 640 rollouts over "tens of hours".
  - Results: held-out pass 23.9% → 25.7% (noise level); mid-difficulty tier 5/17 → 9/17. The author estimates about 24x more training is needed to detect a 12-point change.
- **Multi-turn GRPO with LoRA on TextCraft (community).** Qwen2.5-3B reached 82% pass@1 on a single GPU. LoRA RL drifted from the reference and collapsed without periodic adapter resets (ReLoRA). — [GitHub rl-gym-workout](https://github.com/Vadim-Lagresle/rl-gym-workout)
- **SWE-Protégé RL.** A composite reward covering correctness, stall penalty, and expert-judge, with a curriculum that first targets stalling. It virtually eliminated loops (31.0% → 0.8%). — [alphaXiv 2602.22124](https://www.alphaxiv.org/abs/2602.22124)
- **SWE-Fuse.** Entropy-aware RLVR added about 3.6 points over 8B SFT. — [arXiv 2603.07927](https://arxiv.org/html/2603.07927v1)
- **Reward hacking.** "The Verification Horizon" (arXiv 2606.26300) argues reward hacking emerges once the policy outpaces the verifier. SWE-Marathon (arXiv 2606.07682) observed reward-hacking in 13.8% of frontier rollouts. — [2606.26300](https://arxiv.org/pdf/2606.26300); [2606.07682](https://arxiv.org/abs/2606.07682)
- **Infrastructure noted.** verl GRPO+LoRA handbook (multi-GPU); NeMo RL supports multi-turn tool-use RL. — [HF blog](https://huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl); [NeMo RL](https://github.com/Leonyuan1/NeMo-RL)

### Inferences
- On an A100 80 GB (less than the 96 GB card above), GRPO at 9B with 16K+ context and 8 rollouts per group is feasible only in a phased mode. A few hours buys tens of steps, which per the above is likely noise-level.
- If GRPO is used at all, restrict it to a short, targeted run on mid-difficulty tasks, with these rewards:
  - an F2P/P2P pass reward
  - a −penalty for finishing without a test/repro command after the last edit
  - a repetition penalty
- OPD gives a dense per-token signal at far lower cost (section 1) and is the better use of the budget.

### Gaps
- No found 2026 paper benchmarks DAPO/GSPO/CISPO specifically at 9B on a single GPU. These variants were not researched in depth within this budget.
- Turn-level credit assignment papers were not retrieved.

## 5. Long-horizon: context management, loops, verify-before-finish

### Takeaway
Trained context management (summarization, folding, memory) lets a model with a small window beat the same model with a large window on SWE tasks. Premature "false finish" is a recognized 2026 failure mode with dedicated metrics. An evidence-gated finish (inference-time) gave +4.8 to +11.8 points on SWE-bench Verified with no training.

### Cited Findings
- **Context management**
  - **SWE-MeM** (Jun 2026, snippet). Trained on-demand memory management. A 30B model with a 32K window reaches 60.2% on SWE-bench Verified, versus 51.6% for the same model with a 256K window, while using fewer tokens. — [alphaXiv 2606.28434](https://www.alphaxiv.org/abs/2606.28434v1)
  - **Context-Folding / FoldGRPO** (Oct 2025). 0.580 on SWE-bench Verified, about +8.8% over the untrained folding agent. — [liner review](https://liner.com/review/scaling-longhorizon-llm-agent-via-contextfolding)
  - **SUPO** (arXiv 2510.06727, ACL 2026). Summarization-based context in RL; gains on function-calling and search, not SWE. — [2510.06727](https://tldr.takara.ai/p/2510.06727)
  - **CompactionRL** (snippet only). GLM-4.5-Air reaches 66.8% on SWE-bench Verified. I could not verify a primary URL.
- **Premature finish**
  - **ECLoop** (arXiv 2607.28815). An evidence-conditioned execution layer gates commit and finish actions until supporting evidence exists. +4.8 to +11.8 Pass@1 on SWE-bench Verified across 2 models × 2 scaffolds, with no training. — [arXiv 2607.28815](https://arxiv.org/pdf/2607.28815)
  - **Long-Horizon-Terminal-Bench** (arXiv 2607.08964). Defines a "false finish" metric. Agents overestimate completion and under-invest in final verification. — [arXiv 2607.08964](https://arxiv.org/html/2607.08964)
  - **SWE-Marathon** (arXiv 2606.07682). Frontier agents solve under 30% of tasks; failures are attributed to weak self-verification and early exits. — [arXiv 2606.07682](https://arxiv.org/abs/2606.07682)
  - **Agent loops and self-evaluation** (arXiv 2607.25152). Self-evaluation bias in long-running loops; argues for externally grounded verification. — [arXiv 2607.25152](https://arxiv.org/pdf/2607.25152)
- **Loops.** SWE-Protégé's stall-penalty RL cut long action loops from 31.0% to 0.8% (section 4). A chat-template mismatch alone caused 90% repeated think-blocks in Qwen3.5-9B RL (section 3). — [alphaXiv 2602.22124](https://www.alphaxiv.org/abs/2602.22124); [agentic-rl-swe](https://github.com/d4wnchen-jpg/agentic-rl-swe)

### Inferences
- **Training-free first.** Add an ECLoop-style finish gate to our harness, rejecting "done" unless a test or repro ran after the last edit. Then distill that behavior: RFT/SFT on gated trajectories, which removes the need for the gate at inference.
- **Audit the Qwen3.5 chat template** for prefix-chain consistency and empty `<think>` blocks before any SFT, RFT, or OPD run.
- Context-management training is high-value for very long tasks but probably second-order for a 14-task eval at 16K. Defer it.

### Gaps
- I found no paper training a reward specifically for "verify before finishing" at small scale.
- "Root cause vs symptom" has no dedicated 2026 training paper in my search.

## 6. Model merging and weight-space methods (2026)

### Takeaway
2026 evidence says SFT-style merging (task arithmetic, TIES) underperforms on RL-trained agents. RL-trained models merge better than SFT models. There is no 2026 coding-agent-specific evidence for ExPO-style extrapolation, so the planned merge-strength sweep is cheap and fine but should be judged empirically.

### Cited Findings
- **Reinforced Agent Merging (RAM)** (arXiv 2601.13572, ACL 2026). RL updates are sparse and heterogeneous, so SFT-oriented merging underperforms. RAM separates shared from task-specific regions, and RAM+ beats the specialist coding agent on LiveBench and LiveCodeBench. — [arXiv 2601.13572](https://arxiv.org/abs/2601.13572v1)
- **"Enough is as good as a feast"** (arXiv 2607.22039, Jul 2026). RL-trained models retain more performance after merging than SFT models. The authors attribute this to on-policy data, conflict-shrinking objectives, and positive plus negative examples. — [alphaXiv 2607.22039](https://www.alphaxiv.org/abs/2607.22039)

### Inferences
- A LoRA merge-strength sweep (alpha scaling 0.5–1.5) costs only eval time and is worth doing. Extrapolation above 1.0 is ExPO-like and unproven for agents.
- Because OPD is on-policy, OPD adapters should merge more cleanly than SFT adapters (extrapolated from 2607.22039).

### Gaps
- No 2026 ExPO and coding-agent result was found. The original ExPO paper was not re-verified in this pass.

## 7. Other 2026 items with outsized gains; ranked top-5 for our setup

### Takeaway
The repeated 2026 pattern for small coding agents is: strong curated SFT, then on-policy teacher signal (OPD or expert-augmented RFT), then a light shaped-reward RL or harness gating. Official post-training details for Qwen3.5/3.8, Devstral 2, GLM-4.7-Flash and Nemotron 3.5 were not retrieved within this budget.

### Cited Findings
- **KAT-Coder-V2** (per the OPD survey). Consolidates five domain experts into one agentic coder via OPD; 79.6% on SWE-bench Verified (large model). — [OPD survey 2604.00626](https://arxiv.org/html/2604.00626v3)
- **MAD-OPD** (per survey). Multi-teacher debate OPD with Qwen3/3.5 students from 1.7B to 14B on agentic and code tasks. — [OPD survey 2604.00626](https://arxiv.org/html/2604.00626v3)
- **Teacher size.** In the TRL experiments, the much larger teacher did not yield a better student than a 30B teacher. — [HF TRL post](https://huggingfacetb-trl-distillation-trainer.hf.space/)

### Inferences: ranked top 5 by expected gain per GPU-hour for our setup (judgment, not measured)
1. **Harness and template hygiene, plus a verify-before-finish gate, distilled via RFT** (about 1–3 GPU-h).
   - Fix the Qwen3.5 template prefix chain and empty-think issue.
   - Add an ECLoop-style finish gate and a "write a minimal repro" instruction.
   - Collect gated, verified rollouts and SFT on them.
   - Evidence: ECLoop +4.8 to +11.8 points with no training; the template fix took repetition from 90% to 0%.
2. **Multi-turn OPD from Qwen3.8-27B on student rollouts** (about 8–12 GPU-h, phased sample → teacher prefill-score → LoRA train).
   - Use STRIDE-style early stop and prefix restart to cut cost 2–5x.
   - This gives dense per-token correction exactly at the first wrong action, such as a premature "done".
   - Evidence: 9–30x cheaper than SFT and 50–100x cheaper than RL for reasoning; gains survive LoRA better than SFT.
   - Precondition: confirm tokenizer and chat template identity.
3. **Curated SFT** (the existing plan; about 5–8 GPU-h).
   - Weight data toward diverse task sources, error-recovery segments, and trajectories that verify before finishing.
   - Drop rather than truncate long trajectories.
   - Evidence: OpenThoughts-Agent, Nemotron-Terminal, SWE-Next.
4. **One to two rounds of Docker-free RFT / expert iteration** on SWE-smith-style mutated-function tasks with hidden F2P tests (about 3–5 GPU-h, mostly vLLM sampling). Evidence: SWE-smith, SWE-Protégé RFT.
5. **LoRA merge-strength sweep** (eval-only, under 1 GPU-h). Optionally follow with a short shaped-reward GRPO (pass + verification + repetition penalty) only if budget remains. GRPO evidence on one GPU is noise-level at a few hours (agentic-rl-swe).

### Gaps
- I did not retrieve Qwen3.5/3.6/3.8 technical-report post-training sections, Devstral 2, GLM-4.7-Flash, Nemotron 3.5, or gpt-oss post-training details. Verify these separately.
- I did not verify GPU-hour estimates for our exact pipeline; they are inferences.
