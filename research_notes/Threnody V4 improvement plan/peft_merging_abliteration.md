# PEFT choices, adapter scaling/merging, forgetting, and abliteration side effects (for Threnody V4)

Context recap (from assignment, not externally verified): V3 = LoRA r=64, alpha=32 (so PEFT scale alpha/r = 0.5) on all decoder linear layers incl. Gated-DeltaNet projections of Qwen/Qwen3.5-9B, lr 1e-4, ~56 steps, merged at 0.75 (effective LoRA scale 0.375), then refusal-direction abliteration on layers 12–28. V3 regressed vs base on 3/4 agent benchmarks. Research date 2026-11-05.

## 1. LoRA vs full FT; rank / alpha / lr / steps / modules / batch size; what merge-strength instability indicates

### Takeaway
Properly configured LoRA (all layers incl. MLP, lr ~10x FullFT, small batch, rank big enough for the dataset) matches FullFT on post-training-sized data (Thinking Machines 2025), and LoRA forgets less than FullFT mostly because it learns less (Biderman et al. 2024). Needing to scale a merged adapter down to 0.3–0.75 to avoid looping is a symptom of an over-strong / over-fit delta (too-high effective lr or too narrow data relative to steps), not something to fix at merge time.

### Cited Findings
- Thinking Machines "LoRA Without Regret" (2025): with a few details right, LoRA reaches the same sample efficiency and final performance as FullFT on small-to-medium post-training datasets; it underperforms when the dataset exceeds adapter capacity (pretraining-scale) — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Optimal LoRA LR ≈ 10x FullFT LR; fitted multiplier across 14 Llama and Qwen models = 9.8; for short runs (< ~100 steps) a preliminary multiplier of ~15x looked better; optimal LR varies < 2x between rank 4 and rank 512 (rank 1 somewhat lower) — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- They used alpha = 32 throughout with standard init (A uniform scaled 1/√d_in, B zero, same LR for A and B) and "could not improve on these defaults"; with 1/r scaling learning curves are identical across ranks early in training — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Attention-only LoRA significantly underperforms MLP-only; adding attention on top of MLP gave no further benefit; not just param count (attention-only r=256, 0.25B params, < MLP-only r=128, 0.24B params); held for Llama-3.1-8B and Qwen3-30B-A3B — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Batch size: LoRA's penalty vs FullFT grows with batch size regardless of rank (10k-example OpenThoughts3 subset); at batch 32 the gap is small and shrinks; both methods best at smaller batch — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- RL: LoRA matches FullFT for policy-gradient RL even at rank 1 (info argument: RL gives O(1) bits/episode; ~320k bits for their MATH run vs ~3M params in rank-1 LoRA on Llama-3.1-8B); on DeepMath with Qwen3-8B-base LoRA and FullFT trajectories were nearly identical, LoRA had a wider band of near-optimal LRs — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Capacity rule of thumb: ~2 bits/param memorization capacity; SFT datasets ~1 bit (0.69 nats) per token of loss — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Biderman et al. "LoRA Learns Less and Forgets Less" (TMLR 2024; Llama-2-7B, code & math, ~100K IFT pairs and 20B-token CPT): in standard low-rank settings LoRA substantially underperforms FullFT but better maintains out-of-domain performance; FullFT perturbations have rank 10–100x greater than typical LoRA configs — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- Biderman best practices: use LoRA for instruction FT not CPT; target all modules at r=256 if memory allows ("ranks 16 to 64 tend not to suffice for code tasks"); attention-only underperforms, most gains from MLP; set alpha = 2r (at r=256, alpha=512 beat alpha=256 or 32); sweep LR 1e-5..5e-4 and take the highest stable; best LoRA LR 5e-4 (code), 2e-4 (math) vs FullFT 5e-5 (code), 1e-5 (math); LoRA needs more epochs (Tülu-v2: FullFT peaked at epoch 2, LoRA ~4) — [arXiv 2405.09673v2](https://arxiv.org/html/2405.09673v2)
- Biderman numbers: Code IFT HumanEval FullFT 0.497 (ep 8) vs LoRA r=256 0.498 (ep 4); Code IFT ep16 forgetting-avg (HellaSwag/ARC-C/WinoGrande) FullFT 0.414 vs LoRA r=64 0.509; Math IFT GSM8K FullFT 0.641 vs LoRA 0.634; Math IFT ep16 forgetting-avg 0.559 vs 0.567 — [arXiv 2405.09673v2](https://arxiv.org/html/2405.09673v2)
- LoRA as regularizer: dropout (0.05/0.1) and weight decay (5e-5/1e-4) learned and forgot about as much as FullFT; LoRA r=16 learned less & forgot less; r=256 learned as much as the regularized models while forgetting less; LoRA preserved more generation diversity than FullFT — [arXiv 2405.09673v2](https://arxiv.org/html/2405.09673v2)
- Note conflicting alpha guidance: Thinking Machines keep alpha=32 fixed (scale 32/r, LR re-tuned), Biderman recommends alpha=2r. These are reconcilable because alpha and LR are partially redundant (TM: two of four nominal hyperparameters redundant) — [Thinking Machines](https://thinkingmachines.ai/blog/lora/); [arXiv 2405.09673v2](https://arxiv.org/html/2405.09673v2)
- Unsloth's Qwen3.5 example: r=16, alpha=16, dropout 0, batch 1 x grad-accum 4, 10 warmup, 100 max steps; targets q/k/v/o + gate/up/down (does not mention linear-attention modules); 4-bit QLoRA "not recommended" for any Qwen3.5 model; Qwen3.5-9B bf16 LoRA ≈ 22 GB, FullFT ≈ 4x VRAM; to preserve reasoning keep ≥75% reasoning-style examples in the mix — [Unsloth Qwen3.5 docs](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Axolotl lists a Qwen3.5-9B FFT config (vision+text, single GPU) at ~61 GiB peak — [Axolotl Qwen3.5 docs](https://docs.axolotl.ai/docs/models/qwen3.5.html)

### Inferences
- V3's alpha=32 at r=64 gives scale 0.5; under TM's convention (alpha fixed, LR ~10–15x FullFT) lr 1e-4 is a plausible-to-low value for an 8–9B model, not obviously too high. So the instability at merge strength 1.0 after only ~56 steps more likely reflects: (a) a narrow/small dataset driving the model far off distribution in few steps (looping = mode collapse onto repetitive training patterns), (b) data/template issues (missing EOS, chat-template or thinking-tag mismatch — Unsloth names template/EOS mismatch as the most common cause of degraded exported models), or (c) adapter applied to sensitive GDN gate projections (see §4). These are hypotheses; V3 logs/data should be checked.
- Merging at 0.75 / 0.3 is equivalent to shrinking the learned delta (≈ early stopping / lower lr post hoc). If 0.3 was needed in V2, the delta was ~3x "too large" — fix upstream with lower lr, more diverse data, replay, and early stopping by held-out eval, then validate that strength 1.0 is stable.
- Because agent tasks are multi-step and code-heavy, Biderman's finding that r=16–64 is insufficient for code suggests either higher rank (128–256) for skill learning, or accept LoRA's "learns less" and use RL (where TM show rank barely matters).
- 56 steps is very short; TM's 15x multiplier hint for <100-step runs is preliminary. A small LR sweep (3–5 values) with per-checkpoint agent eval is cheap at 9B and more informative than merge-strength sweeps.
- Use small effective batch (≤32 sequences) for LoRA per TM.

### Gaps
- No Qwen3.5-9B-specific published LoRA LR / rank sweeps found; Unsloth page gives no LR. Closest verifiable: TM results on Qwen3-8B / Qwen3-30B-A3B (dense+MoE, not hybrid linear attention).
- No primary source found that directly diagnoses "looping at merge strength 1.0"; the diagnosis above is inference.
- rsLoRA (alpha/√r scaling, Kalajdzievski 2023, arXiv 2312.03732) not fetched this session — cite cautiously; TM's results suggest the standard 1/r scaling is fine if LR is tuned.

## 2. Merging methods: linear/task arithmetic, TIES, DARE, soups, checkpoint averaging/EMA, WiSE-FT

### Takeaway
Interpolating toward base (WiSE-FT style) reliably trades target gains for retained general ability, but recent LLM evidence says merge operators are not a "free lunch" against capability forgetting; operator choice (linear vs TIES vs DARE) matters little for a single adapter. Merging is best used to combine several independently good checkpoints, not to rescue an unstable one.

### Cited Findings
- WiSE-FT (weight-space ensembling of zero-shot and fine-tuned weights, CLIP/ImageNet): +4–6 pp accuracy under distribution shift over prior work on ImageNet shifts, +2–23 pp on six more shifts, +1.6 pp in-distribution on ImageNet, +0.8–3.3 pp on seven transfer datasets, no extra compute — [arXiv 2109.01903](https://arxiv.org/abs/2109.01903)
- DARE: randomly drop a fraction p of delta parameters and rescale survivors by 1/(1−p); SFT deltas typically within 0.002 and highly redundant; can drop 90% (even 99%) of deltas while preserving abilities; used to merge multiple SFT models sharing a base; benefits more pronounced in larger models — [arXiv 2311.03099](https://arxiv.org/abs/2311.03099)
- CapTrack (Thede et al., Feb/May 2026, models up to 80B): forgetting extends beyond parametric knowledge with pronounced drift in robustness and default behaviors; instruction fine-tuning induces the strongest relative drift; preference optimization is more conservative and can partially recover lost capabilities; "no universal mitigation emerges" — [arXiv 2603.06610](https://arxiv.org/abs/2603.06610)
- Per a search summary of CapTrack (not verified in the full text this session): comparing linear interpolation, TIES, DARE-linear and DARE-TIES between base and instruction-tuned models, general-purpose merging operators are unlikely to provide a "free lunch" against capability-level forgetting without limiting learning — [arXiv 2603.06610 PDF](https://arxiv.org/pdf/2603.06610)
- A continual-learning paper merging LoRA-adapted weights into the base (search-summary, unverified): higher base weight (α=0.75) gave more stable performance / less forgetting, equal weighting (α=0.5) higher initial performance but more degradation — [arXiv 2503.05683](https://arxiv.org/pdf/2503.05683)
- A 2026 study merging per-document LoRA adapters found four merge strategies within ~1 point of each other on every dataset (search summary) — [arXiv 2609.17346](https://arxiv.org/pdf/2609.17346)
- Optimal scaling coefficients for Task Arithmetic and TIES decrease as the number of merged LoRA models increases (thesis, search summary) — [Georgia Tech repository](https://repository.gatech.edu/entities/publication/ae6fc506-f79a-438d-ab3e-24e2daca969b)
- Blog posts promoting DARE+TIES as "best practice" for LoRA merging provide no controlled benchmarks — treat as unverified — [machinelearningatscale](https://www.machinelearningatscale.com/blog/model-merging-dare-ties-guide)

### Inferences
- V3's 0.75 merge is exactly WiSE-FT-style interpolation; it can only move along the line base↔adapter, so if the adapter direction itself hurts agent skills (3/4 regressions), no α recovers gains. Evaluate an α-curve (0, 0.25, 0.5, 0.75, 1.0) on agent benchmarks before abliteration to see whether any α beats base; if none does, the training data/objective is the problem.
- Better uses of merging for V4: (a) average several checkpoints / seeds of the same run (soup / EMA-like) to reduce variance; (b) TIES/DARE only when combining multiple specialist adapters (e.g., coding + tool-use) on the same base.
- Given CapTrack's finding that preference optimization drifts less and can recover capabilities, a DPO/RL stage is more promising than merge tricks for preserving base ability.

### Gaps
- No found evidence specifically on merging agent/coding fine-tunes of hybrid linear-attention models.
- TIES (Yadav et al. 2023, arXiv 2306.01708) and model soups (Wortsman et al. 2022) not fetched in this session; no numbers quoted.
- No primary data found on EMA / checkpoint averaging for LoRA SFT on LLM agent tasks.

## 3. Forgetting mitigation: replay, KL regularization, lower lr, LoRA variants, freezing, sequential vs mixed

### Takeaway
Forgetting tracks how far the fine-tuned policy moves from the base on the new-task distribution (KL); methods that keep that shift small (on-policy RL, preference optimization, replay/mixed data, lower LR, LoRA) forget less. No universal fix exists; measure forgetting explicitly.

### Cited Findings
- RL's Razor (Shenfeld, Pari, Agrawal, Sep 2025; NeurIPS 2025 workshop): RL preserves prior knowledge better than SFT at similar new-task performance; forgetting is governed by KL between fine-tuned and base policy on the new-task distribution, not the algorithm per se; on-policy RL is implicitly biased toward KL-minimal solutions, SFT can converge arbitrarily far — [arXiv 2509.04259](https://arxiv.org/abs/2509.04259)
- LoRA mitigates forgetting more than dropout or weight decay (r=16 learned less/forgot less; r=256 learned as much as regularized FullFT but forgot less) — [arXiv 2405.09673v2](https://arxiv.org/html/2405.09673v2)
- Instruction FT causes strongest drift; preference optimization more conservative and partially recovers capabilities; drift pronounced in robustness and default behaviors; no universal mitigation — [arXiv 2603.06610](https://arxiv.org/abs/2603.06610)
- Unsloth: for Qwen3.5 keep ≥75% reasoning-style examples to preserve reasoning ability when mixing in direct-answer data — [Unsloth Qwen3.5 docs](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Replay ratio claims: practitioner sources suggest 1–5% old data or 20–30% general data; these figures are unsourced in the blogs — [DEV Community](https://dev.to/javieraguilarai/the-forgetting-you-dont-measure-3h11); [genai-systems-lab](https://genai-systems-lab-ivory.vercel.app/gt/catastrophic-forgetting)
- I-LoRA (2024) proposes dual-memory replay via LoRA parameter interpolation for continual instruction tuning — [arXiv 2402.18865](https://arxiv.org/abs/2402.18865v1)
- A 2025 paper constrains domain FT with parameter-importance regularization computed on general data — [arXiv 2501.13669](https://arxiv.org/html/2501.13669v2)

### Inferences
- Practical V4 recipe candidates: (1) mixed rather than sequential training — interleave agent trajectories with general/reasoning data, ideally self-generated by base Qwen3.5-9B (on-policy-ish distillation keeps KL low); (2) track KL(fine-tuned || base) on held-out task prompts as a forgetting proxy per RL's Razor; (3) follow SFT with on-policy RL or DPO rather than more SFT; (4) lower LR + early stopping by agent eval rather than post-hoc merge scaling.
- Self-distilled (base-generated, filtered-for-success) agent trajectories are a direct way to reduce SFT distribution shift — inference from RL's Razor, not a tested result here.

### Gaps
- No head-to-head DoRA vs PiSSA vs OLoRA vs LoRA forgetting comparison found; can't rank them. (DoRA arXiv 2402.09353, PiSSA arXiv 2404.02948 not fetched.)
- No verified primary-source replay ratio for LLM SFT found this session.
- No evidence found on freezing specific layer types (e.g., GDN layers) to reduce forgetting.

## 4. Fine-tuning hybrid linear-attention models (Gated DeltaNet / Qwen3.5)

### Takeaway
Qwen3.5 interleaves full attention with Gated DeltaNet layers; tooling works (Axolotl/Unsloth, transformers v5, flash-linear-attention) but target-module naming is a real pitfall: q/k/v/o patterns miss the GDN layers, "all" may miss them in some setups, and fused names differ between training and serving.

### Cited Findings
- Qwen3.5 is a hybrid model interleaving standard attention with Gated DeltaNet linear attention; all Qwen3.5 models are early-fusion VLMs (dense: `Qwen3_5ForConditionalGeneration`) — [Axolotl Qwen3.5 docs](https://docs.axolotl.ai/docs/models/qwen3.5.html)
- Axolotl LoRA targets for GDN: `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.out_proj` (plus q/k/v/o for full attention) — omits `in_proj_a`/`in_proj_b` — [Axolotl Qwen3.5 docs](https://docs.axolotl.ai/docs/models/qwen3.5.html)
- Sample packing with GDN requires flash-linear-attention (pinned `flash-linear-attention==0.5.2`, uninstall `causal-conv1d`); without FLA packed sequences raise a RuntimeError — [Axolotl Qwen3.5 docs](https://docs.axolotl.ai/docs/models/qwen3.5.html)
- Unsloth: transformers v5 required; Qwen3.5 uses custom Mamba-style Triton kernels (slow compile); vLLM 0.16.0 does not support Qwen3.5; exported-model degradation most often from mismatched chat template/EOS; QLoRA not recommended for Qwen3.5 — [Unsloth Qwen3.5 docs](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Unfused HF names: `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`; vLLM fuses into `in_proj_qkvz` and `in_proj_ba`; SGLang needed a PR to map adapters trained on unfused layout (incl. `in_proj_ba`) — [vLLM llm-compressor PR #3408](https://github.com/vllm-project/llm-compressor/pull/3408); [SGLang PR #31522](https://github.com/sgl-project/sglang/pull/31522)
- Pitfalls reported: standard q/k/v/o targets hit only the full-attention layers (Qwen3.6 activation-oracle paper) — [arXiv 2605.26045](https://arxiv.org/pdf/2605.26045); a PEFT export bug wrote GDN modules under `self_attn` names that readers can't map back — [lattice issue #2406](https://github.com/ohdearquant/lattice/issues/2406); LoRA support on GDN mixer projections added as a feature in another trainer — [surogate PR #218](https://github.com/invergent-ai/surogate/pull/218)

### Inferences
- LoRA on GDN projections is supported and probably needed for full capacity (≈3/4 of Qwen3.5 layers are GDN in the Qwen3-Next style — layer ratio not verified here for 9B). But `in_proj_a`/`in_proj_b` produce the decay/beta gates of the recurrence; perturbing them can change memory retention over long contexts. Axolotl's omission of a/b is a conservative default worth copying for V4 (inference, not tested).
- Check that V3's merged weights actually contained GDN deltas under correct `linear_attn.*` names, and that "all-linear" did not adapt the vision tower.
- If V3 trained with packing but without FLA varlen support, cross-sample state leakage in the recurrence could corrupt training — a hypothesis to check in logs.
- Abliteration tooling built on TransformerLens may not support Qwen3.5 (FailSpy had no TL support even for Qwen3-8B per §5); verify which matrices V3's abliteration orthogonalized in GDN layers (out_proj) vs attention o_proj.

### Gaps
- No official Qwen technical guidance found on LoRA for GDN layers or recommended Qwen3.5-9B fine-tuning hyperparameters.
- No published study comparing LoRA with vs without GDN projections on Qwen3.5.
- Exact Qwen3.5-9B layer count / GDN:attention ratio not verified in this session.

## 5. Abliteration side effects, layer choices, alternatives, evaluation, responsible release

### Takeaway
Single-direction orthogonalization usually costs ≤~1–2 pp on MMLU/ARC/GSM8K but consistently lowers TruthfulQA, and effects are strongly model- and tool-dependent (up to −18.8 pp GSM8K). No published measurement of abliteration's effect on agent/tool-use benchmarks was found, so V4 must measure it; applying abliteration across 17 layers (12–28) after a fine-tune is a plausible contributor to V3's regressions.

### Cited Findings
- Arditi et al. 2024 orthogonalized vs baseline (largest model per family): Gemma 7B MMLU 51.8 vs 51.7, GSM8K 31.3 vs 32.0, TruthfulQA 44.7 vs 47.1; Yi 34B MMLU 73.5 vs 74.9, TruthfulQA 51.9 vs 55.4; Llama-3 70B GSM8K 90.8 vs 91.2, TruthfulQA 59.5 vs 61.8; Qwen 72B MMLU 76.5 vs 77.2, TruthfulQA 55.0 vs 56.4. Except Qwen 7B and Yi 34B all metrics within 99% CIs; TruthfulQA consistently drops — [arXiv 2406.11717](https://arxiv.org/html/2406.11717)
- Arditi direction selection: choose direction minimizing bypass score subject to induce score > 0, KL on harmless prompts < 0.1, and layer index < 0.8·L; authors note the method "relies on several heuristics" — [arXiv 2406.11717](https://arxiv.org/html/2406.11717)
- Young (arXiv 2512.13655v2, Jan 2026), 16 models 7–14B, four tools: average deltas — Heretic MMLU −0.78, GSM8K −7.81, HellaSwag −0.56; DECCP −0.61/−0.13/−0.25; ErisForge −0.12/−0.28/−0.08; worst case Heretic on Yi-1.5-9B GSM8K −18.81 pp (−26.5% rel.); math reasoning most sensitive; KL–refusal correlation r = 0.87 (n=8); Qwen3-8B Heretic KL 0.210, 25/100 refusals; refusal-marker heuristic underestimates ASR (classifier 95.7% non-refusal vs 72.2% markers over 900 responses); single runs only; GSM8K SE ~1.2–1.4 pp — [arXiv 2512.13655](https://arxiv.org/html/2512.13655)
- mlabonne abliteration blog: direction = normalized mean(harmful) − mean(harmless) last-token residual activations per layer; candidates ranked and picked manually; orthogonalizes embedding, attention-output and MLP-output weights; abliterated Daredevil-8B scored lower on every Open LLM Leaderboard and Nous benchmark; healed by DPO (QLoRA r=64, alpha=32, lr 5e-6, 1 epoch, orpo-dpo-mix-40k) recovering most of the drop except GSM8K — [HF blog](https://huggingface.co/blog/mlabonne/abliteration)
- Closest Qwen3.5-9B datapoint (self-reported model card, uncensored "HauhauCS-Aggressive" variant, lm-eval-harness on vLLM): MMLU 78.64→78.34, GSM8K strict 87.64→84.99, ARC-C 54.52→53.75, WinoGrande 72.77→71.35, TruthfulQA MC2 53.76→45.77 (−7.99), Lambada ppl 3.88→4.05; method of uncensoring not confirmed as abliteration; ~3–4% GSM8K truncation — [Friendli model page](https://friendli.ai/models/DreamFast/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark). A search snippet also cited a Qwen3.5-9B variant dropping GSM8K 87.64→81.96 on the same comparison page; not verified.
- Refusal direction is polysemantic (entangled with syntax/formatting/capability); degradation attributed to imprecise interventions, motivating "surgical" spectral-cleaning variants — [arXiv 2601.08489](https://arxiv.org/html/2601.08489v1); refusal geometry involves multi-dimensional concept cones; in Qwen2.5-7B only one candidate direction met both refusal and low-side-effect criteria — [arXiv 2502.17420](https://arxiv.org/pdf/2502.17420)
- Code LLMs (Qwen2.5-Coder 3B/7B/14B, Jun 2026): abliteration drove refusal to ~0 while syntactic validity stayed >93%, and SQL-injection-vulnerable code generation reached 88–97% (14B) — concrete harm uplift; no HumanEval/MBPP reported in abstract — [arXiv 2606.05396](https://arxiv.org/abs/2606.05396)
- Defenses noted: extended-refusal training and representation rerouting increase resistance to abliteration — [arXiv 2512.13655](https://www.arxiv.org/pdf/2512.13655)
- No study measuring abliteration effects on BFCL / agentic tool-calling was found — [search result listing incl. arXiv 2512.13655](https://www.arxiv.org/pdf/2512.13655)

### Inferences
- V3 ablated across layers 12–28 (a wide band; Arditi-style is a single direction projected out of all write matrices, layer only chooses where the direction is extracted). If V3 extracted per-layer directions and projected each out, that is a larger intervention than the published method; check implementation.
- Agent benchmarks rely on calibrated abstention (e.g., BFCL tests not calling a tool when inappropriate) and on following system-prompt constraints; removing a "refusal/decline" direction plausibly harms these — an untested hypothesis. Also TruthfulQA drops consistently, suggesting hedging/calibration behaviour is affected.
- Order matters: abliterating after fine-tuning means the direction is computed on the fine-tuned model; V4 should evaluate each stage separately (base, SFT, merged, abliterated) to attribute regressions.
- Evaluation protocol for V4: capability (agent benchmarks + MMLU/GSM8K/HumanEval-like + TruthfulQA) before/after; KL on harmless prompts (Arditi threshold < 0.1); refusal on harmful sets scored by a classifier, not string markers (markers undercount, per Young); over-refusal on benign borderline prompts; harmfulness via HarmBench-style judge; multiple seeds given GSM8K SE ~1.3 pp.
- Alternative to abliteration if the goal is reducing over-refusal: train it in (DPO on benign-but-borderline prompts with helpful responses), which CapTrack suggests drifts less, and which can be scoped to over-refusal without removing refusal of genuinely harmful requests.
- Responsible release: abliteration demonstrably raises harmful compliance (e.g., vulnerable-code generation 88–97%); a release should document the intervention, report harm evals, avoid presenting as safety-aligned, and consider whether full refusal removal is needed for the agent use case at all.

### Gaps
- No measured effect of abliteration on agent/tool-use benchmarks (BFCL, SWE-bench, tau-bench etc.) found.
- No Qwen3.5 (hybrid GDN) abliteration study with capability numbers beyond the self-reported model card; whether refusal is mediated similarly in GDN layers is unknown.
- Arditi per-model chosen layers (Table 5) not retrieved.
