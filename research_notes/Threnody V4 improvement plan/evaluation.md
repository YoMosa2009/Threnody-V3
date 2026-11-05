# Evaluation Methodology for Threnody V4 (small agentic-coding LoRA fine-tune, single A100)

Research date: 2026-11-05. All numbers carry an inline source. Numbers marked "my arithmetic" are standard textbook formulas I applied, not values taken from a source. Some sources are aggregators or blogs; they are flagged.

## Q1. Which public benchmarks are standard and feasible for 7–14B agentic coding models, and what are the baselines?

### Takeaway
The benchmarks most often reported for this model class in 2025–2026 are SWE-bench Verified (500 tasks, usually through mini-SWE-agent or OpenHands), Terminal-Bench 2.x (89 tasks, run through Harbor), BFCL-V4, tau2-bench and LiveCodeBench v6. The base model's own card already reports BFCL-V4 66.1, TAU2 79.1 and LiveCodeBench v6 65.6. V4 must at least match these under the same settings, or it has regressed the base. Scaffolds differ from model card to model card, so the only fair baselines are ones V4 re-runs itself, with the same harness and the same quant.

### Cited Findings
**Base-model baselines (Qwen3.5-9B official card).** Most of these are probably thinking-mode results, because thinking is on by default.
- LiveCodeBench v6 65.6; OJBench 29.2; BFCL-V4 66.1; TAU2-Bench 79.1; VITA-Bench 29.8; DeepPlanning 18.0. For comparison, Qwen3.5-4B scores 55.8 / 24.1 / 50.3 / 79.9 / 22.0 / 17.6 on the same benchmarks — [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- The card's TAU2 airline domain was evaluated "with fixes proposed in the Claude Opus 4.5 system card". A V4 run must say whether it applies the same fixes — [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- Thinking is on by default. Turning it off requires `enable_thinking: False` in the chat template kwargs — [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- The card recommends 32,768 output tokens for most queries and 81,920 for hard math and programming benchmarks — [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- I could not confirm SWE-bench Verified or Terminal-Bench figures for Qwen3.5-9B on the official card. One aggregator lists SWE-bench Verified 60.0% and Terminal-Bench **2.1** 29.2%, but these are unverified — [benchmarklist.com](https://benchmarklist.com/models/qwen-qwen3.5-9b/)

**SWE-bench Verified baselines for comparable open models.** All are vendor-reported, and the scaffolds differ.
- Devstral Small 1.1 (24B): 53.6% with OpenHands. Devstral Small 1.0: 46.8% (OpenHands). SWE-smith-LM 32B: 40.2% (SWE-agent). Skywork-SWE: 38.0% (OpenHands). DeepSWE: 42.2% (R2E-Gym). The run count is not stated — [Devstral-Small-2507 card](https://huggingface.co/mistralai/Devstral-Small-2507)
- SWE-agent-LM-32B scores 40.2% pass@1 on Verified. It was fine-tuned from Qwen2.5-Coder-32B-Instruct on 5,016 Claude 3.7 Sonnet trajectories — [SWE-smith paper](https://arxiv.org/abs/2504.21798)
- Skywork-SWE-32B scores 38.0%, or 47.0% with best-of-8 test-time scaling — [emergentmind summary](https://www.emergentmind.com/topics/skywork-swe-model) (secondary source)
- Devstral Small 2: most sources say 68.0% on Verified (vendor figure), but other sources give 65.8% or 46.8%, so the reports conflict. Terminal-Bench 2 22.5% comes from one comparison table only — [jangwook.net](https://jangwook.net/en/blog/en/devstral-qwen3-coder-small-models/), [Artificial Analysis comparison](https://artificialanalysis.ai/models/comparisons/devstral-small-2-vs-qwen3-coder-30b-a3b-instruct) (secondary sources; unverified)
- On Artificial Analysis "TerminalBench Hard", Devstral Small 2 scores 16.7% and Qwen3-Coder-30B-A3B-Instruct 15.2% — [Artificial Analysis](https://artificialanalysis.ai/models/comparisons/devstral-small-2-vs-qwen3-coder-30b-a3b-instruct) (aggregator)
- Qwen3-32B with an older harness: 15.2% on a 100-sample Verified subset — [SambaNova blog](https://sambanova.ai/blog/are-llms-truly-solving-software-problems) (shows how much the harness matters)
- In the KTH variance study, Qwen3-32B and DeepSWE-preview (a Qwen3-32B fine-tune) on Verified range from about 20% to 34% pass@1 depending on scaffold and temperature — [On Randomness in Agentic Evals](https://arxiv.org/html/2602.07150)

**SWE-bench family and harnesses**
- swebench.com offers Verified, Lite, Multilingual, Multimodal and Full variants. Its "Bash Only" leaderboard runs Verified with mini-SWE-agent for every model, using bash only and not the model's tool-calling interface — [swebench.com](https://www.swebench.com/), [Bash Only page](https://www.swebench.com/bash-only.html)
- Bash-only cost per task for open-weight models on Vals.ai ranges from about $0.02 to $1.13 per test. GPT-OSS-120B scores 33.6% at about $0.17 per test — [Vals.ai](https://www.vals.ai/benchmarks/swe_bench-verified) (third-party; the figures were inconsistent between two archived tables)
- SWE-bench Live started with 1,319 tasks from GitHub issues created since 2024, across 93 repos. Curation is automated, and one third-party summary says about 50 new verified instances are added each month (unverified) — [SWE-bench Goes Live!](https://arxiv.org/pdf/2505.23419), [alphaxiv](https://alphaxiv.org/benchmarks/shanghai-artificial-intelligence-laboratory/swe-bench-live)
- SWE-rebench mines tasks from GitHub continuously and publishes them in monthly YYYY_MM splits. The authors explicitly say some models' scores "might be inflated due to contamination". A V2 reports 32,000+ tasks in 20 languages — [SWE-rebench](https://fugumt.com/fugumt/paper_check/2505.20411v1_enmode), [Harbor hub listing](https://hub.harborframework.com/datasets/swe-rebench/swe-rebench-leaderboard/latest?page=107)
- On the SWE-rebench leaderboard (BenchLM, Oct 2026), Qwen3.5-27B scores 58.9% and Qwen3.5-35B-A3B 53.7%. No 9B entry is listed — [BenchLM](https://benchlm.ai/benchmarks/swe-rebench) (aggregator)

**Terminal-Bench**
- Terminal-Bench 2.0 (arXiv 2601.11868, Jan 2026) has 89 tasks. Three reviewers verified each, and they were chosen from 229 crowd-sourced tasks. Frontier agents solve under 65% of tasks and smaller models about 15%. Figures show 95% CIs. The benchmark is distributed through the Harbor harness and registry, and the authors' configs are at github.com/laude-institute/terminal-bench-experiments — [Terminal-Bench 2.0 paper](https://arxiv.org/html/2601.11868v1)
- Secondary summaries mention ">32,000 trials across 16 models and 6 agents" and at least 5 trials per task. I did not confirm the per-pair minimum in the primary text — [emergentmind](https://www.emergentmind.com/topics/terminalbench-2-0)
- A Terminal-Bench 2.1 now exists, and the Quesma study used it (89 tasks, 3-hour timeout, 98k reserved context) — [Quesma blog](https://quesma.com/blog/qwen38-27b-quantizations-benchmarked/)

**BFCL-V4 / tau2-bench**
- BFCL V4 adds agentic categories: web search, memory (KV, vector, recursive-summary; 5 domains) and format sensitivity — [Gorilla BFCL v4 memory blog](https://gorilla.cs.berkeley.edu/blogs/16_bfcl_v4_memory.html). One third-party guide says format sensitivity is unscored and gives a weighting, but this is unverified — [bfcl guide](https://jonathanding.github.io/llm-learning/en/articles/bfcl-guide)
- tau2-bench subsets are airline, retail and telecom, aggregated as mean plus pass^k (consistency across repeated runs) — [EvalScope docs](https://evalscope.readthedocs.io/en/v1.4.1/get_started/supported_dataset/agent.html)

### Inferences
- **Feasibility on one A100 (my estimate; no source gives a wall-clock time for a 9B model).** BFCL-V4, LiveCodeBench and tau2-bench are cheap: no Docker, and hours of runtime. tau2 needs a user-simulator LLM, which is an API cost and a source of variance, so pin the simulator model. Terminal-Bench (89 tasks × ≥3–5 trials, Docker) and SWE-bench Verified (500 tasks, Docker images, tens of GB of disk) take about a day per full pass at local throughput. That is still feasible, but plan the trial counts.
- Recommended V4 suite:
  - Tier A, primary claims: SWE-bench Verified via mini-SWE-agent bash-only, which tests the model and not the scaffold; Terminal-Bench 2.0 or 2.1 via Harbor/Terminus-2; BFCL-V4; tau2-bench.
  - Tier B, contamination-resistant: a SWE-rebench or SWE-bench Live monthly split created after the base model's and V4 data's cutoff; a LiveCodeBench window after the cutoff.
  - Tier C: the existing in-house suites, used only as dev and regression checks.
- Re-run the base Qwen3.5-9B in the same harness, quant and sampling as V4. Published card numbers use unknown harnesses and probably thinking mode, so they are context and not paired baselines.
- V3's custom suites (14 / 14 / 30 / 16 tasks) cannot support any public-facing claim (see Q2).

### Gaps
- No primary-source SWE-bench Verified or Terminal-Bench numbers for Qwen3.5-9B, Qwen3-8B/14B or Qwen3-Coder-30B-A3B. Read them off the swebench.com bash-only and tbench.ai leaderboards directly; the leaderboard tables did not render here.
- Aider Polyglot and Multi-SWE-bench: I did not gather sizes or small-model scores.
- LiveCodeBench release-date window mechanics: not verified in this pass. It is known to filter problems by contest date, but I have no fresh citation.
- No measured wall-clock or GPU-hour figures for running these benchmarks on a 9B model on one A100.

## Q2. Statistical power: how many tasks, which CIs/tests, how many seeds; how much do agent benchmarks vary?

### Takeaway
At 14–30 tasks, a 95% CI is roughly ±18–26 pp, so V3's results could not resolve differences under about 20 pp. Detecting 5 pp needs hundreds of paired tasks with McNemar or a paired bootstrap, plus several runs per task. Even at temperature 0, agent runs on SWE-bench Verified vary with an SD of about 0.7–1.8 pp, and single-run ranges span 2–6 pp.

### Cited Findings
- **Variance at scale.** "On Randomness in Agentic Evals" (Bjarnason, Silva, Monperrus, KTH) ran 10 runs × 12 configurations on SWE-bench Verified (500 tasks) with Qwen3-32B, DeepSWE-preview and Devstral-2: 60,000 trajectories in total. Single-run pass@1 ranges spanned 2.2–6.0 pp and SDs ran 0.7–1.8 pp — [arXiv 2602.07150](https://arxiv.org/html/2602.07150)
- **Temperature 0 is not deterministic.** Qwen3-32B/nano-agent had SD 1.2% at T=0 versus 0.7% at T=0.6, because inference engines are non-deterministic (floating point, batching). Trajectories diverge within about the first 1% of tokens; the median first divergence is token 56 at T=0 — [arXiv 2602.07150](https://arxiv.org/html/2602.07150)
- **Temperature can matter a lot for fine-tunes.** DeepSWE-preview scored 31.4% at T=1.0 versus 20.4% at T=0, a significant difference. Devstral-2 scored 63.5% at T=0.2 versus 63.8% at T=0, not significant — [arXiv 2602.07150](https://arxiv.org/html/2602.07150). This bears directly on V3's T=0-only evaluation.
- **pass@1 vs pass@k vs pass^k.** For DeepSWE on r2e-gym: pass@1 34.4%, pass@5 52.9%, pass^5 15.5%. The authors recommend reporting all three — [arXiv 2602.07150](https://arxiv.org/html/2602.07150)
- **Number of runs.** With σ = 1.5 pp, α = 0.05 and 80% power: about 9 runs to detect a 2 pp gain, about 36 runs for 1 pp, and about 15 runs for 2 pp at 95% power. A gain of about 10 pp may be detectable in a single run — [arXiv 2602.07150](https://arxiv.org/html/2602.07150)
- **Older reference point.** GPT-4 on SWE-bench Lite over 6 runs: mean 17.94%, SD 0.49, but per-instance outcomes changed considerably between runs — [SWE-agent paper](https://arxiv.org/pdf/2405.15793)
- **Small subsets are noisier.** On a 50-issue SWE-bench Pro sample over 5 runs, SDs were 2.0 to 3.4 pp (2.2 for mini-SWE-agent) — [SWE-Doctor](https://arxiv.org/pdf/2607.00990)
- **Miller, "Adding Error Bars to Evals".** Report a CLT standard error next to every mean; use clustered SEs when items are related (the adjustment can be up to 3×); compare models with paired differences on the same questions, which gives tighter intervals than comparing the two means — [arXiv 2411.00640](https://arxiv.org/abs/2411.00640). Implemented in the `errorbars` package — [PyPI](https://pypi.org/project/errorbars/)
- **Small-n warning (ICML 2025 position).** CLT-based error bars "dramatically underestimate" uncertainty below a few hundred datapoints. The authors recommend Bayesian or non-CLT frequentist intervals such as Wilson or Clopper-Pearson, and provide a library at github.com/sambowyer/bayes_evals — [Bowyer, Aitchison, Ivanova, arXiv 2503.01747](https://arxiv.org/pdf/2503.01747)
- **Worked example of a rigorous small-model protocol.** 3 seeds; exact two-sided McNemar on paired outcomes with Holm correction across 8 comparisons; a cluster bootstrap over problems (20,000 resamples); a within-problem permutation test — [Beyond Accuracy: How Quantization Affects Coding Agents, arXiv 2611.02368](https://arxiv.org/html/2611.02368)

### Inferences (my arithmetic, standard formulas)
- **95% CI half-width at p = 0.5, n = tasks.** n=14: ±26 pp; n=30: ±18 pp; n=89 (Terminal-Bench): ±10 pp; n=500 (Verified): ±4.4 pp. Wilson or Bayesian intervals will be wider still at small n.
- **Unpaired two-proportion test, 30% vs 35%, α = .05, 80% power.** About 1,370 tasks per arm. Unpaired designs are hopeless for 5 pp.
- **Paired McNemar, 5 pp difference.**
  - If 10% of tasks are discordant between models, about 310 tasks are needed.
  - If 20% are discordant, about 625.
  - Formula: n ≈ (z_α·√p_d + z_β·√(p_d − Δ²))² / Δ².
- **Consequences for V4.** A 5 pp claim is only realistic on SWE-bench Verified (500), combined with SWE-rebench/Live splits, using paired per-task comparison and k ≥ 3 runs per task, where per-task mean success reduces variance. Terminal-Bench (89 tasks) alone can resolve only about 10–15 pp differences, even with 5 trials.
- **Recommended analysis:**
  - Per-task success rate averaged over k runs.
  - Paired bootstrap over tasks (cluster = task) for the CI of the V4 − base difference.
  - Exact McNemar on the majority-vote or per-run outcomes.
  - Holm correction across benchmarks × quant variants.
  - Report pass@1 (mean ± SE), pass@k and pass^k.
  - Run at the vendor-recommended temperature as well as T=0.

### Gaps
- No published discordance rates between a base model and its LoRA fine-tune on agent benchmarks. Estimate p_d from a pilot run.
- I did not verify the "≥5 trials" Terminal-Bench protocol in the primary paper.

## Q3. Held-out discipline: dev/selection split vs final test, contamination, time-based held-out sets

### Takeaway
V3 selected checkpoints on its own "held-out" opencode suite, so that suite is a dev set and V3's held-out numbers are optimistically biased. V4 needs three things: a dev split used for checkpoint and quant selection; a frozen test split touched once; and a time-based split that starts after both the base model's and the training data's cutoffs.

### Cited Findings
- SWE-rebench was built because of contamination concerns. Its authors say performance of some models "might be inflated due to contamination issues", and they publish new tasks in monthly splits — [SWE-rebench paper](https://fugumt.com/fugumt/paper_check/2505.20411v1_enmode), [Harbor hub](https://hub.harborframework.com/datasets/swe-rebench/swe-rebench-leaderboard/latest?page=107)
- SWE-bench Live is "continuously updated" from issues created since 2024, with automated curation — [SWE-bench Goes Live!](https://arxiv.org/pdf/2505.23419)
- SWE-agent-LM / SWE-smith trains on synthetic tasks from 128 repos. Such pipelines can overlap with SWE-bench repos, so check repo overlap — [SWE-smith](https://arxiv.org/abs/2504.21798); [Diyi Yang post](https://x.com/Diyi_Yang/status/1920146923333517478)

### Inferences
- **Protocol:**
  1. Write the selection metric and gate thresholds into code, version-controlled, before training.
  2. Select checkpoints and quants on dev only: the custom suites plus, for example, a random 100-task slice of SWE-bench Verified marked as dev.
  3. Freeze the config by commit hash and run the test split once: the remaining Verified tasks, Terminal-Bench and BFCL.
  4. Additionally report a post-cutoff SWE-rebench or Live month as the contamination-resistant headline.
- **Decontamination checks against V4 training data:**
  - Repo-level exclusion of every SWE-bench, SWE-rebench and Terminal-Bench source repo.
  - n-gram (for example 13-gram) or MinHash overlap between training trajectories and benchmark problem statements and gold patches.
  - Report the check and its thresholds in the model card.
- If V4 training data is distilled from a teacher, the teacher may itself be contaminated on Verified. Time-based splits are the main defense.
- **Gate integrity (V3's records conflicted with its code).**
  - Generate gate decisions from the logged eval JSON by script.
  - Store the gate-code commit hash next to each decision.
  - Never hand-edit published gate records.

### Gaps
- No source here quantifies how much selecting on the test set inflates scores for agent benchmarks specifically. Standard ML practice says it does, but I found no number for this setting.
- Exact Qwen3.5 training cutoff date, which is needed to choose a "safe" SWE-rebench month: not found.

## Q4. Evaluating quantized models beyond KLD/perplexity

### Takeaway
Task-level evals on each shipped quant are now the norm. The best 2026 evidence suggests 4-bit (Q4_K_M, AWQ/GPTQ W4A16) usually keeps final accuracy on coding and agent benchmarks within noise for ≥7B models. But quantization can change agent behavior (tool use, malformed calls) significantly, and degradation becomes visible at 2-bit and collapses at 1-bit. KLD measured on the imatrix calibration text is in-sample: it overstates fidelity and says nothing about agent behavior.

### Cited Findings
- **Quesma, Aug 26 2026, Qwen3.8-27B GGUF (Unsloth), llama.cpp, F16 KV cache.**
  - Q4_K_M matches BF16 and Qwen's reported score on Terminal-Bench 2.1 (89 tasks).
  - UD-Q2_K_XL shows a "noticeable fall".
  - 1-bit (UD-IQ1_M/S) is near or below random on GPQA Diamond.
  - IFBench shows no measurable change down to Q2_K_XL.
  - Q8_0 was not run on Terminal-Bench; its value was interpolated.
  - Error bars were Wilson 95% on GPQA only; the trial count was not stated.
  - The author deliberately did not use KLD, arguing token-level metrics don't reliably indicate task performance.
  - Caveat: Unsloth replaced the quant files on Aug 19 2026, so the exact files can no longer be reproduced.
  - [Quesma blog](https://quesma.com/blog/qwen38-27b-quantizations-benchmarked/)
- **"Beyond Accuracy: How Quantization Affects Coding Agents" (Mehboob et al., 2026 workshop, arXiv 2611.02368).**
  - Setup: Qwen2.5-Coder 7/14/32B and Devstral-Small-2507; W4A16 AWQ/GPTQ against BF16; OJBench (115 problems), 3 seeds, T=0.6.
  - Final accuracy: BF16 / AWQ / GPTQ was 7.83 / 7.25 / 6.67 (7B), 11.30 / 9.28 / 9.86 (14B) and 13.91 / 10.72 / 11.01 (32B). After Holm correction no difference is significant (32B AWQ raw p = 0.027, adjusted 0.21).
  - Behavior shifts:
    - Qwen-7B one-shot solves fell from 17 to 0 under AWQ, while loop solves rose from 10 to 25.
    - Devstral's valid tool use fell from 8.7% to 1.2% (GPTQ), with up to 96.5% malformed calls.
    - Devstral's failure-category shift was significant (adjusted p ≈ 0.01).
  - The authors' conclusion: similar final accuracy "can hide large behavioral shifts".
  - Calibration: 256 samples of 1,024 tokens.
  - [arXiv 2611.02368](https://arxiv.org/html/2611.02368)
- An older blog study found instruction following degrades fastest under low-bit quantization, especially AWQ and Q4_K_M — [ionio](https://www.ionio.ai/blog/llm-quantize-analysis) (blog; older; unverified)
- Practitioner guidance says not to go below Q4 for agent and tool-use workloads — [runlocalmodel](https://runlocalmodel.com/choosing-quantization-2026.html) (blog; opinion)
- Low-bit reasoning models can inflate their token use: longer reasoning exhausts the budget and returns empty answers. Quesma observed this at 1-bit — [Quantization Inflates Reasoning, arXiv 2606.25519](https://arxiv.org/pdf/2606.25519) (title only seen); [Quesma](https://quesma.com/blog/qwen38-27b-quantizations-benchmarked/)
- "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models" (arXiv 2504.04823) exists, but I did not verify its findings here — [arXiv 2504.04823](https://arxiv.org/pdf/2504.04823)

### Inferences
- For each shipped quant (Q8_0, Q6_K, Q4_K_M, and any IQ3/IQ4), run the dev agentic suite plus at least one public test benchmark (BFCL-V4 and a Terminal-Bench or SWE-bench subset) with k ≥ 3 runs. Compare against BF16 or Q8_0 with paired McNemar or bootstrap and Holm correction.
- Log behavioral metrics per quant:
  - tool-call parse failure rate;
  - tool calls and turns per task;
  - tokens per task;
  - timeout or empty-answer rate;
  - failure-category distribution.
- Measure KLD and same-top-token on a held-out text distinct from the imatrix calibration set: agent trajectories, code, and chat-templated tool-call transcripts. In-sample KLD is optimistic.
- Keep the imatrix calibration data disjoint from every eval set. If calibration text contains eval prompts, the quant is tuned toward the test and both KLD and task scores become suspect.
- At 9B, expect 4-bit to be riskier than at 27B. In arXiv 2611.02368 the smaller Qwen-7B showed the largest behavioral shift, but that is one study. Treat IQ3 and below as requiring a full agent eval before release.

### Gaps
- No peer-reviewed measurement of llama.cpp K-quants (Q4_K_M, IQ3_XXS) on SWE-bench- or Terminal-Bench-style agents at about 9B. The available evidence is either 27B (Quesma) or AWQ/GPTQ (arXiv 2611.02368).
- No quantitative study found on imatrix calibration-set leakage into evals. The concern rests on reasoning, not measurement.

## Q5. Reproducibility and reporting standards

### Takeaway
Credible 2026 reports publish four things for each benchmark: the harness and version; the scaffold; the full sampling configuration, including thinking mode; and the number of runs with error bars. Ideally they also release per-task trajectories. The base model's own card shows that sampling settings differ by mode, so V4 must report which mode and settings were used.

### Cited Findings
- **Qwen3.5-9B recommended sampling settings by mode:**
  - Thinking, general: T=1.0, top_p=0.95, top_k=20, presence_penalty=1.5.
  - Thinking, precise coding: T=0.6, top_p=0.95, top_k=20, presence_penalty=0.
  - Non-thinking, general: T=0.7, top_p=0.8, top_k=20, presence_penalty=1.5.
  - The card contradicts itself on non-thinking reasoning settings (T=1.0/top_p 1.0/top_k 40/pp 2.0 versus top_p 0.95/top_k 20/pp 1.5).
  - [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B)
- Benchmark sampling and length settings: 32,768 output tokens in general, 81,920 for hard coding and math — [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B)
- The Terminal-Bench 2.0 authors publish their experiment configs (terminal-bench-experiments repo) and distribute the benchmark through the Harbor registry for exact reruns — [Terminal-Bench 2.0](https://arxiv.org/html/2601.11868v1)
- The KTH study recommends multiple independent runs, power analysis to choose the run count, and reporting pass@k and pass^k alongside pass@1. It also reports token and tool-call totals (25.58B tokens, 1.88M tool calls) — [arXiv 2602.07150](https://arxiv.org/html/2602.07150)
- Devstral's model card names a scaffold per row but does not state run counts, which shows a common reporting gap — [Devstral-Small-2507 card](https://huggingface.co/mistralai/Devstral-Small-2507)
- Miller recommends reporting the SE in parentheses beside each mean and noting clustering and pairing — [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)

### Inferences
- **V4 eval manifest.** Store one JSON per run, generated automatically:
  - model SHA and adapter SHA;
  - GGUF file SHA256 and the quant tool commit;
  - imatrix dataset hash;
  - llama.cpp build hash and server flags (ctx, KV-cache type, batch size, `-np` parallel slots, which affect non-determinism);
  - chat template hash and `enable_thinking`;
  - full sampler settings and seed;
  - harness name and version or commit;
  - benchmark split and version (for example Terminal-Bench 2.0 vs 2.1; tau2 with or without the airline fixes);
  - max turns, max tokens and timeouts;
  - per-task outcome, trajectory file, token and turn counts, and wall-clock time.
- **Model card "Evaluation" table.** Columns: benchmark (version, n tasks), harness and scaffold, mode (thinking or non-thinking), sampling, runs k, pass@1 mean ± 95% CI, pass@k, and the base-model score under identical settings. Also state contamination checks and which splits were used for selection.
- Report both thinking and non-thinking modes. V3 was non-thinking only, and the official base numbers are likely thinking mode, so mixing them misrepresents the delta.
- Release the trajectories, or at least per-task pass/fail JSONL, so third parties can run the paired tests themselves.

### Gaps
- I did not fetch Hugging Face's model-card guidance text (the model-index/`eval_results` metadata spec) this pass. The writer should cite it directly if needed.
- No single community standard for agent-eval reporting was found. The recommendations above synthesize the papers cited.
