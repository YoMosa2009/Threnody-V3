# Qwen3.5-9B base model facts and recipes behind small open agentic-coding peers (Threnody V4 planning)

Research date: 2026-11-05. "Vendor" = number published by the model's maker; "Independent" = third party / paper using its own harness. Unverified items are flagged.

## Q1. Verified facts about Qwen3.5-9B (architecture, context, template, thinking, sampling, benchmarks, license)

### Takeaway
Qwen3.5-9B (Feb/Mar 2026, Apache-2.0) is a 32-layer hybrid model (3 Gated DeltaNet : 1 Gated Attention), natively multimodal, 262K context, thinking-on by default, with the `qwen3_coder` XML tool-call format. Qwen publishes **no SWE-bench or Terminal-Bench numbers for the 9B**. The official agent numbers are BFCL-V4 66.1, TAU2 79.1 and LiveCodeBench v6 65.6. The best independent harness numbers are about 20–21% on Terminal-Bench 2.x and 46.2% on SWE-bench Verified (Ai2 TMax harness). That is the realistic floor V4 has to beat.

### Cited Findings
**Release / family / license**
- The Qwen3.5 series launched in February 2026 (flagship Qwen3.5-397B-A17B; the blog is titled "Qwen3.5: Towards native multimodal agents"). The model card citation says "February 2026" — [HF Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B); [DeNovoSWE ref list citing qwen.ai/blog?id=qwen3.5](https://arxiv.org/pdf/2606.10728)
- The small series (0.8B, 2B, 4B, 9B, plus Base checkpoints) was reported released on March 2, 2026. This comes from secondary sources; the official blog post was not retrieved — [Product Hunt](https://www.producthunt.com/posts/1088853); [thedailypac blog](https://thedailypac.blogspot.com/2026/03/alibaba-just-released-qwen-35-small.html)
- License: Apache-2.0 — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- Medium Qwen3.5 sizes include 27B (dense) and 35B-A3B and 122B-A10B (MoE) — [Qwen3.6-27B card table](https://huggingface.co/Qwen/Qwen3.6-27B); [llama.cpp #20837](https://github.com/ggml-org/llama.cpp/issues/20837)

**Architecture (vendor, from model card)**
- Causal LM with vision encoder, 9B params, 32 layers, hidden size 4096, FFN intermediate 12288, vocab 248,320 (padded) — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- Layout: `8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`. That gives 24 linear-attention (DeltaNet) layers and 8 full-attention layers — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- Gated DeltaNet: 32 V heads, 16 QK heads, head dim 128. Gated Attention: 16 Q heads, 4 KV heads, head dim 256, RoPE dim 64 — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- MTP: the card says only "trained with multi-steps", with no further detail — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- Context: 262,144 native. It extends to about 1,010,000 with YaRN, and the card warns that static YaRN scaling may hurt short texts — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)

**Thinking modes / template**
- Thinking is on by default (`<think>...</think>`). Disable it with `chat_template_kwargs: {"enable_thinking": false}`. Qwen3.5 does **not** officially support Qwen3's `/think` and `/nothink` soft switches — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- Tool calling: vLLM `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`; SGLang `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` (XML `<tool_call><function=...><parameter=...>` format) — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B); format illustrated in [llama.cpp #20837](https://github.com/ggml-org/llama.cpp/issues/20837)
- Successor Qwen3.6 added a `preserve_thinking` chat-template kwarg, which keeps reasoning from earlier turns. Qwen says Qwen3.6 was "trained to preserve and leverage thinking traces from historical messages." This is a Qwen3.6 feature; it is not documented for 3.5 — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)

**Official recommended sampling (vendor, model card "Best Practices")**
| Mode | temp | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|---|
| Thinking, general | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 |
| Thinking, precise coding (e.g. WebDev) | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Non-thinking (instruct), general | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |
| Non-thinking, reasoning tasks | 1.0 | 1.0 | 40 | 0.0 | 2.0 | 1.0 |
- The card is internally inconsistent. The Quickstart lists the non-thinking reasoning row as top_p 0.95, top_k 20, presence_penalty 1.5 — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- Qwen's own agentic evals (Qwen3.6 card, SWE-bench and Terminal-Bench) use temp 1.0 and top_p 0.95 (TB2 also uses top_k 20), with 200K–256K context — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)

**Official 9B benchmark scores (vendor, thinking mode)**
- LiveCodeBench v6 65.6; OJBench 29.2; BFCL-V4 66.1; TAU2-Bench 79.1; VITA-Bench 29.8; DeepPlanning 18.0; IFEval 91.5; IFBench 64.5; MMLU-Pro 82.5; GPQA-D 81.7; AA-LCR 63.0; LongBench v2 55.2; OSWorld-Verified 41.8; ScreenSpot Pro 65.2; AndroidWorld 57.8 — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- The card has **no SWE-bench Verified or Terminal-Bench entry for the 9B** — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)

**Independent 9B agent numbers**
- Ai2 TMax used a mini-SWE-agent-style harness with persistent shell and interleaved thinking. Qwen3.5-9B base scored: Terminal-Bench Lite 49.4±1.9, TB 2.1 20.5±1.7, **TB 2.0 21.0** (Daytona, 3 trials), and **SWE-bench Verified 46.2±0.8**. AIME'24/25 was 67.5 with no harness and 72.2 with the harness — [TMax, arXiv 2606.23321v2](https://arxiv.org/abs/2606.23321)
- Harness sensitivity for Qwen3.5-9B on TB-Lite: own harness 49.4, mini-SWE-agent 44.9, Terminus-2 37.9, OpenHands 29.0. Scores differ by 20 points depending on harness alone — [TMax](https://arxiv.org/abs/2606.23321)
- Third-party aggregator benchmarklist.com lists Qwen3.5-9B at **60.0% SWE-bench Verified** and **29.2% Terminal-Bench 2.1**, with no harness given. This conflicts with TMax's 46.2 / 20.5 and is **unverified** — [benchmarklist](https://benchmarklist.com/models/qwen-qwen3.5-9b/)
- Reference points for larger siblings (vendor, internal scaffold): Qwen3.5-27B scored SWE-V 75.0, SWE-Pro 51.2, TB2.0 41.6. Qwen3.5-397B-A17B scored SWE-V 76.2 and TB2.0 52.5 — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)

### Inferences
- Only 8 of 32 layers keep a growing KV cache; the 24 DeltaNet layers keep a fixed recurrent state. Long agent contexts should therefore be cheap in memory relative to a full-attention 9B. However, prefix/prompt caching for hybrid recurrent models is harder in llama.cpp (the recurrent state cannot be truncated to an arbitrary prefix), so multi-turn agents may re-process prompts. This was not verified with a source in this pass; test it.
- Qwen's recommended presence_penalty of 1.5 is unusual. Harnesses that don't send presence_penalty (many OpenAI-compatible clients) run the model off-spec, which may contribute to looping. Evaluate V3, V4 and base under identical sampling.
- Because Qwen gives no official SWE or TB numbers for the 9B, any "beats base" claim must come from Threnody's own harness runs of base vs. fine-tune.

### Gaps
- The official Qwen3.5 blog (qwen.ai/blog?id=qwen3.5) and small-model announcement were not fetched directly. The March 2 date is from secondary sources.
- No official 9B SWE-bench, Terminal-Bench or Aider numbers. The 60.0% SWE-V figure is unverifiable.
- MTP head details (number of steps) and whether llama.cpp uses the Qwen3.5 MTP head for speculative decoding were not verified. An unsloth "Qwen3.5-122B-A10B-MTP-GGUF" repo exists, which suggests some MTP GGUF support — [unsloth](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-MTP-GGUF).
- Whether a Qwen3.6 small (≤9B) model exists was not found. The Qwen3.6 open weights found are 35B-A3B and 27B only.

## Q2. Strongest small open agentic-coding models (≈2B–35B) in 2025–2026 and their recipes

### Takeaway
The frontier has moved. In late 2026, sub-10B open models reach about 30–33% on TB2.0 (TMax-9B, RL-only on Qwen3.5-9B). Mid-size open models reach 68–77% SWE-V (Devstral Small 2 24B; Qwen3.6-27B). 2025-era 32B fine-tunes (SWE-smith, Skywork-SWE, DeepSWE, SERA) reached only 40–54% SWE-V on Qwen2.5/Qwen3 bases, which are weaker than Qwen3.5-9B. The 2026 recipes that worked on strong hybrid bases were **outcome-reward RL in executable environments**, not SFT.

### Cited Findings
**Small (≤10B) — directly relevant to a 9B base**
- **TMax-9B (Ai2/UW, Jun–Nov 2026)**: RL-only on Qwen3.5-9B with no SFT warm-start.
  - Data: TMax-13K synthetic terminal environments, built from a taxonomy plus difficulty control, personas and diverse verifiers.
  - Algorithm: DPPO (a GRPO variant with TV-divergence token masking), token-level loss, fully async. Zero-std groups are filtered and active sampling is used. Group size 32, 8 prompts/batch, 65,536 max sequence length, 64 max tool calls, temp 1.0, 500 steps. The LM head is kept in FP32 to reduce train–inference mismatch.
  - Templates: edited so that thinking is preserved on intermediate turns (interleaved thinking).
  - Compute: 2 H100 nodes for training plus 6 for inference, 1–2 days. About $3,150 of Daytona sandbox cost per run.
  - Results: TB-Lite 49.4→65.6, TB2.1 20.5→29.6, **TB2.0 21.0→33.0**, SWE-V 46.2→55.3, AIME 72.2→93.3.
  - Gains carried over to other harnesses: OpenHands 29.0→42.7, mini-SWE-agent 44.9→57.8, Terminus-2 37.9→46.1.
  - Source: [TMax, arXiv 2606.23321](https://arxiv.org/abs/2606.23321)
- TMax at other sizes: Qwen3.5-4B scored TB2.0 16.5→25.1 and 2B scored 1.5→9.4. Qwen3.6-27B went 42.7→44.9, within noise on TB2.1 (43.6→43.9). Gains shrink as the base gets stronger — [TMax](https://arxiv.org/abs/2606.23321)
- Data-source comparison (same RL recipe on Qwen3.5-9B, TB2.1): base 20.5; TerminalTraj 15.9 (worse than base); TermiGen 21.2; OpenThinker-Agent 22.7; SWE-Smith 27.3; Endless Terminals 28.4; TMax-13K 29.6 — [TMax](https://arxiv.org/abs/2606.23321)
- Other ~8B terminal agents (TB2.0, self-reported or leaderboard): Nemotron-Terminal-8B 13.0; OT-Agent-ColdSFT+RL-8B 13.5; Endless-Terminals-8B 6.7; SETA-8B 10.7; Terminal-Lego-Qwen3-8B 11.8; base Qwen3-8B 0.9. Nemotron-Terminal-14B/32B scored 20.2 and 27.4 — [TMax Table 15](https://arxiv.org/abs/2606.23321)
- River (Salesforce et al., Aug–Sep 2026) filters low-quality synthetic environments and adds process-level behavior regularization to outcome RL. It reports +106% / +30% larger RL gains on TB-Lite / TB2.1 using under 30% of TMax environments, and its 8B is the best among open RL-trained 8B models. It argues RL mainly shapes high-level decisions over skills learned in pretraining and SFT. The base models were not visible in the abstract — [River, arXiv 2608.22631](https://arxiv.org/abs/2608.22631)
- SWE-agent-LM-7B (SWE-smith, on Qwen2.5-Coder-7B) scored 15.2% SWE-V — [SWE-smith, arXiv 2504.21798](https://arxiv.org/pdf/2504.21798) (figure via search summary)

**Mid-size (24–35B) — reference targets**
- Qwen3.6-27B (dense, Apr 2026, vendor, internal scaffold): SWE-V 77.2, SWE-Pro 53.5, SWE-Multilingual 71.3, TB2.0 59.3, SkillsBench 48.2. Qwen3.6-35B-A3B: SWE-V 73.4, TB2.0 51.5 — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)
- Devstral Small 2 (24B, Mistral, Dec 9 2025, Apache-2.0, 256K context): **68.0% SWE-V** per most aggregators. One site lists 65.8%, and the official card was not fetched — [Vercel AI Gateway](https://vercel.com/ai-gateway/models/devstral-small-2); [llmreference](https://www.llmreference.com/model/devstral-small-2); conflicting [canirun.ai](https://www.canirun.ai/model/devstral-small-2-24b)
- Qwen3-Coder-Next (80B-A3B on Qwen3-Next hybrid, early 2026, vendor):
  - Scores: SWE-V 70.6/71.1/71.3 (SWE-Agent / mini-SWE-agent / OpenHands); SWE-Pro 42.7; TB2.0 34.2 (Terminus2-xml) and 25.8 (QwenCode).
  - Mid-training: "trillions" of tokens at 262K context, including about 600B repo-level tokens.
  - Environments: about 808K PR-derived plus about 852K synthetic bug-injection executable tasks.
  - RL: multi-turn SWE RL with penalties for unfinished trajectories and malformed tool calls.
  - Distillation: four experts (WebDev, UX/tool-calling, QA, SWE) distilled into one model.
  - Source: [Qwen3-Coder-Next tech report, arXiv 2603.00729](https://arxiv.org/html/2603.00729v1)
- OpenThoughts-Agent (Qwen3-32B SFT on 100K examples, Jun 2026): SWE-V 54.0, TB2.0 26.2. Nemotron-Terminal-32B scored 41.9 / 25.1 — [Data Recipes for Agentic Models, arXiv 2606.24855](https://arxiv.org/abs/2606.24855)
- SERA-32B (Ai2, Jan 2026, Qwen3-32B, SFT only via "Soft Verified Generation"): SWE-V 49.5 / 54.2 at 32K / 64K context. Total cost about $2,000 (40 GPU-days); about $400 reproduces the main result. Training on 8K Django-only samples gave 52.23% vs. the teacher's 51.20% — [SERA, arXiv 2601.20789](https://arxiv.org/html/2601.20789v1); [devops.com](https://devops.com/open-source-coding-agents-just-got-accessible/)
- DeepSWE-Preview (Agentica/Together, mid-2025): RL-only on Qwen3-32B with 4.5K R2E-Gym tasks. SWE-V Pass@1 rose 23→42.2 in about 200 RL steps; Pass@16 71.0; about 59% with test-time scaling — [Together AI blog](https://together.ai/blog/deepswe)
- SWE-smith-LM-32B (Qwen2.5-Coder-32B, SFT): 40.2% SWE-V — [SWE-smith](https://arxiv.org/pdf/2504.21798). Skywork-SWE-32B: data-scaling study — [Skywork-SWE, arXiv 2506.19290](https://arxiv.org/pdf/2506.19290) (exact score not re-verified in this pass)
- DeNovoSWE (2026): about 11K long-horizon trajectories, SFT at LR 1e-5, batch 128, 131K context. Qwen3.5-35B-A3B improved BeyondSWE-Doc2Repo 43.8→50.0 and NL2RepoBench 23.5→27.1. A weak base (Qwen3-30B-A3B-Instruct) improved 5.8→47.2 — [DeNovoSWE, arXiv 2606.10728](https://arxiv.org/pdf/2606.10728)

**Benchmark caveat**
- OpenAI published "Why SWE-bench Verified no longer measures frontier coding capabilities" in Feb 2026, citing contamination. Newer work uses SWE-bench Pro and Terminal-Bench 2.x — [DeNovoSWE ref list](https://arxiv.org/pdf/2606.10728); [search summary](https://arxiv.org/pdf/2609.08149)

### Inferences
- A realistic V4 target on Qwen3.5-9B is the TMax envelope: roughly +5 to +12 points on TB2.x and +9 points on SWE-V **in the same harness**, from RL on about 13K verified environments. Expecting 27B-class numbers (60%+ TB2.0, 75% SWE-V) from a 9B fine-tune is unrealistic.
- For a hobby budget, the cheapest proven lever that lifted this exact base is outcome-only RL. TMax-scale compute is about 8 H100 nodes for 1–2 days plus sandbox cost; scaled-down variants are untested. The cheapest SFT lever (SERA-style, ≤$2K) was demonstrated only on the weaker Qwen3-32B.
- Repo-specialization (SERA Django result) and long-horizon data (DeNovoSWE) suggest a narrow, well-targeted dataset can beat broad data. Threnody's CLI-agent focus could exploit this if evals match the training harness distribution.

### Gaps
- No official Mistral card fetched for Devstral Small 2 (68.0 vs. 65.8 conflict). The Devstral Small 1.1 (53.6) and Kimi-Dev-72B (60.4) figures could not be verified in this pass.
- NVIDIA Nemotron agentic recipes (Nemotron-Terminal, Nemotron 3 Nano) are known only via TMax's table; their model cards were not fetched.
- No fully-open fine-tune of Qwen3.5-9B reporting SWE-bench Pro was found.

## Q3. Do small-model fine-tuners beat official instruct/agent models? Conditions and negative results

### Takeaway
Fine-tunes beat the official model only when (a) the official base is weak at agentic use (Qwen3-8B, Qwen3-30B-A3B-Instruct), or (b) training is RL against executable verifiers in a harness close to the eval. On the strong, heavily agent-post-trained Qwen3.5/3.6 bases, **SFT on distilled trajectories frequently hurts**, which is consistent with Threnody V3's failure.

### Cited Findings
- **Directly on Qwen3.5-9B, SFT degraded performance.** TMax tested its own SFT set (16.5K trajectories, 8K successful, teacher Qwen3.6-27B) and a larger mix including Nemotron-Terminal, TerminalTraj, OpenThoughts-Agent and SETA data. Both lowered Qwen3.5-9B's TB-Lite from 41.9 to 35.5 (TMax SFT) and 31.3 (large SFT); TB2.1 moved within noise (16.1 / 15.0 / 16.9). The same data helped Qwen3-8B (TB-Lite 7.3→11.5→16.4). The authors attribute this to weak teachers in older data, "while TMax SFT uses Qwen 3.6 27B as a teacher model, we still find it results in degraded performance for Qwen 3.5 9B". They "opted to go straight to RL training for Qwen 3.5" — [TMax §G.2, Table 16](https://arxiv.org/abs/2606.23321)
- TMax filtered SFT traces containing Qwen3 XML-style tool calls because that format "can confuse small models with different tool call formats" — [TMax §G.1 fn. 3](https://arxiv.org/abs/2606.23321)
- RL with a bad dataset can also regress: TerminalTraj RL took Qwen3.5-9B's TB2.1 from 20.5 to 15.9 — [TMax Table 2](https://arxiv.org/abs/2606.23321)
- Gains shrink as base quality rises: TMax-27B on Qwen3.6-27B gained only 43.6→43.9 on TB2.1, which the authors attribute to Qwen3.6's already-strong post-training — [TMax](https://arxiv.org/abs/2606.23321)
- The search summary reported a 2026 paper whose Qwen3.5-27B fine-tune scored 70.6 SWE-V vs. 72.4 for the untouched model. The specific paper could not be pinned down, so treat this as **unverified** — [search result set incl. arXiv 2608.13681](https://arxiv.org/pdf/2608.13681)
- Fine-tuning works on strong bases when data matches a specific long-horizon gap: DeNovoSWE lifted Qwen3.5-35B-A3B by +6.2 on Doc2Repo — [DeNovoSWE](https://arxiv.org/pdf/2606.10728)
- Teacher choice matters non-obviously. "The strongest model by benchmark performance does not necessarily make the best teacher": GPT-5.3-Codex was the weakest teacher and GLM-4.7 the best, with about a 5% TB2.0 difference. Instruction/task-source choice is among the most important factors. Keeping traces with more turns helps. Diversity of sources beats repeating top sources — [OT-Agent, arXiv 2606.24855](https://arxiv.org/abs/2606.24855)
- Tool-call template diversity in training improved SWE-V at fixed data volume. Single-template training overfits to one output structure. Trajectories from one scaffold transferred poorly to others — [Qwen3-Coder-Next report](https://arxiv.org/html/2603.00729v1)
- Harness choice alone moved Qwen3.5-9B by up to 20 points (TB-Lite 29.0–49.4). Terminus-2 is "more brittle with small models" because it requires raw keystrokes — [TMax](https://arxiv.org/abs/2606.23321)

### Inferences
- V3's failure to beat base is the expected outcome for SFT on Qwen3.5-9B. V4 should either (1) go RL-first (GRPO/DPPO-style, outcome reward, interleaved thinking preserved, no SFT warm-start, or only a tiny on-policy self-distilled warm-start), or (2) do strictly on-policy or rejection-sampled self-distillation from Qwen3.5-9B's own successful traces, keeping the native `qwen3_coder` format and thinking style. TMax also lists a "+ Qwen 3.5 9B-Generated RL (3.1k samples)" ablation, but its numbers were not captured.
- Never mix tool-call formats or chat templates different from Qwen3.5's native ones into SFT data for this base, unless deliberately training format diversity as Qwen did at much larger scale.
- Use a fixed eval harness and sampling for base vs. V4 comparisons, ideally the same harnesses the users run (opencode, codex-style), since harness variance exceeds typical fine-tune gains.

### Gaps
- No public example found of a hobbyist or LoRA-scale fine-tune of Qwen3.5-9B beating base on TB2.x or SWE-V with independently reproduced numbers.
- The TMax self-generated RL ablation values and the "pre-repair" ablation were not extracted.

## Q4. Practical llama.cpp / Ollama serving lessons for agentic Qwen3.5

### Takeaway
The biggest known llama.cpp problem for Qwen3.5-9B agents is **tool calls emitted inside the `<think>` block**. Grammar-triggered termination then swallows them, and as of Sep 2026 this is still open with no merged fix. Practical mitigations: disable thinking for tool-heavy harnesses (or use `--reasoning-format none`), use `--jinja` with a corrected or refreshed template, use post-March-5-2026 GGUFs, and avoid DRY sampling.

### Cited Findings
- **llama.cpp issue #20837** (opened Mar 21 2026; open as of the last comment Sep 7 2026; labels bug and chat-parser; assigned to pwilkin).
  - Symptom: with thinking enabled, Qwen3.5-9B writes `<tool_call><function=...>` XML inside its reasoning and generation stops, so the client gets no structured call. Sometimes one call works and the model then degrades to raw XML. Started around build b8255; reproduced on b8461 with unsloth Qwen3.5-9B-UD-Q4_K_XL. Also reported on 27B, 35B-A3B, 122B-A10B and Qwen3.6.
  - Root cause per maintainer: the grammar trigger is independent of the reasoning parser and forces end-of-generation once a tool call is complete. vLLM emits tool calls found in reasoning as real calls; SGLang's parser consumes them.
  - Candidate fix PR #24202 (tagged thinking tool parser) was not merged.
  - Workarounds: `enable_thinking:false` (reliable); `--reasoning-format none` (partial); removing DRY sampling fixed degeneration and path errors; custom templates only partial; lowering temperature reduced but did not stop loops.
  - Source: [llama.cpp #20837](https://github.com/ggml-org/llama.cpp/issues/20837)
- Reporter's config for reference: `ctx-size 131072`, `flash-attn`, `cache-type-k/v q8_0`, `kv-unified`, `temp 0.6`, `top-p 0.95`, `top-k 20`, `min-p 0` — [#20837](https://github.com/ggml-org/llama.cpp/issues/20837)
- Bundled template problems for agentic harnesses:
  - A community "corrected Qwen3.5 chat template for llama.cpp" (Mar 28 2026) fixes system-message ordering for OpenCode and Codex: tool instructions come first, user system content is appended, reasoning is rendered only after the last real user query — [gist](https://gist.github.com/garg-aayush/c0211a5fdca3e237d248d52806ff8d96)
  - QwenLM/Qwen3 issue #1831 proposes a "21-fix" template covering tool-calling crashes, parallel calls and thinking bleed, verified on Qwen3.5-27B in llama.cpp — [QwenLM #1831](https://github.com/QwenLM/Qwen3/issues/1831)
  - Mainline llama.cpp parses historical tool-call arguments into objects before template rendering — [Moore2877 fixed templates](https://huggingface.co/Moore2877/Qwen-Fixed-Chat-Templates-llamacpp)
- Unsloth refreshed Qwen3.5 GGUFs with tool-calling fixes on Feb 27 2026 and again with a fixed chat template on Mar 5 2026. Users reported raw-XML tool-call failures on older GGUFs that were fixed by re-downloading. A Jinja error was reported after build b8227. Thinking off: `--chat-template-kwargs '{"enable_thinking":false}'` — [unsloth Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF); [unsloth 122B MTP GGUF](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-MTP-GGUF)
- A general llama.cpp guide says repetition loops usually trace back to chat-template mismatch in server parsing. Fixes: run with `--jinja` and check the build for regressions. Setups can work in short sessions and break with longer history — [netclaw troubleshooting](https://netclaw.dev/troubleshooting/llama-cpp/)
- The Ai2 TMax team edited chat templates to preserve thinking on intermediate turns, which is not Qwen3.5's default behavior, and found it helpful for agentic RL. Qwen3.6 made this official via `preserve_thinking` — [TMax](https://arxiv.org/abs/2606.23321); [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)

### Inferences
- For V4, ship a GGUF with an embedded, tested template: Qwen3.5-native `qwen3_coder` XML tool calls, correct system ordering, and an explicit choice on historical thinking. Train and serve with the same rendering, and test via llama-server `--jinja` with opencode and codex-style clients.
- Consider training V4 so that tool calls are always emitted **after** `</think>`, for example with an RL penalty for tool calls inside reasoning, as Qwen3-Coder-Next penalized malformed calls. This directly targets the llama.cpp failure mode and would be a distinctive, measurable V4 improvement for llama.cpp users.
- Sampling for the model card should follow Qwen's table (or document deviations). Explicitly warn against DRY and note that presence_penalty support varies by client.

### Gaps
- No verified data on llama.cpp prompt-cache or context-checkpoint behavior for Qwen3.5's recurrent DeltaNet layers in multi-turn agents, or on KV-quantization quality impact for this model.
- Ollama-specific Qwen3.5 template and tool-call status was not researched.
- No authoritative source found on whether llama.cpp currently uses Qwen3.5's MTP head for speculative decoding.
