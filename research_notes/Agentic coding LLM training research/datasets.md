# Open SFT Datasets for a Small Multi-Turn Agentic Coding / Tool-Use / Defensive-Security Model (as of Nov 2026)

Method note: Every repo id below was checked against the live Hugging Face API (`https://huggingface.co/api/datasets/<id>`) and/or its README on 2026-11-02. "Verified" means the API returned the repo. License tags come from the HF API unless stated otherwise. A 401 "Invalid username or password" from the API means the repo does not exist publicly (or is private). Download counts and last-modified dates are from the same API call.

## 1. Multi-turn agentic SWE trajectories with real execution feedback

### Takeaway
There is now a lot of open, permissively licensed, execution-grounded multi-turn SWE trajectory data. Most of it comes from OpenHands, SWE-agent, or mini-swe-agent driven by Qwen3-Coder-480B or Claude, and much of it is filtered to resolved, test-passing runs. For a small model on a Colab budget, the binding constraint is trajectory length (median about 24K-41K tokens), not availability. Pick resolved-only subsets and filter by token length. Prefer sets whose tool format matches the target CLI harness (bash plus a str_replace editor).

### Cited Findings

**Top-tier (resolved/test-verified, permissive, verified to exist)**

- **`nebius/SWE-rebench-openhands-trajectories`**: CC-BY-4.0, 10K-100K rows. Made with Qwen3-Coder-480B-A35B-Instruct on OpenHands v0.54.0 over real GitHub issues from 1,823 repos. 67,074 trajectories, of which 32,161 are successful (3,792 resolved issues). Average 64.3 turns, max 100. Uses native function calling. The format is OpenAI-style `trajectory` messages (system/assistant with `tool_calls`/user/tool). Each row has a `resolved` flag (0/1) plus `gen_tests_correct` and `pred_passes_gen_test`. Note: tool-call `arguments` are serialized to strings and should be `json.loads`-ed before applying a chat template — [HF card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
- The same Nebius card has a comparison table covering several sets. SWE-smith-trajectories: 49,897 trajectories, 21,513 successful, 30.2 average turns, function calling. Kwai-Klear mini-swe-agent-plus: 65,994 trajectories, all successful, 34.3 average turns, no function calling. nebius/SWE-agent-trajectories: 80,036 trajectories, 13,389 successful, 26.4 average turns. SWE-Gym OpenHands sampled: 6,055 trajectories, 491 successful, 18.9 average turns. R2E-Gym SFT: 3,231 trajectories, all successful, 16.1 average turns — [HF card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
- **`SWE-bench/SWE-smith-trajectories`**: MIT. Three configs by tool format: `tool` (24,100 rows), `xml` (26,076) and `ticks` (25,826), about 10.7 GB total. Made with SWE-agent plus Claude 3.7 Sonnet (and Claude 3.5 Sonnet and GPT-4o per the Nebius table) on SWE-smith synthetic tasks. The card says the 5,017 trajectories used to train SWE-agent-LM-32B (40.2% on SWE-bench Verified, pass@1) come from this set. A boolean column marks resolution — [HF card](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
- **`Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k`**: MIT. 65,994 trajectories, all successful, about 4.6 GB, made with mini-swe-agent-plus on SWE-smith tasks. Klear-Agent-8B-SFT, trained on this set, reports 39.0% on SWE-bench Verified. Solve rate grows roughly linearly with log(data size) from 1K to 66K. The bash-only scaffold makes it a close match for CLI harnesses — [HF card](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k)
- **`togethercomputer/CoderForge-Preview`**: 100K-1M rows, about 50 GB download. The HF repo has no dataset-level license tag, but each row has a `license` column. 258,134 trajectories over 51,201 tasks and 1,655 repos (R2E-Gym 4,216 tasks, SWE-Smith 37,221, SWE-Rebench 9,764). 155,144 trajectories pass all repo tests (reward 1.0). Scaffold is OpenHands v0.52.1 with execute_bash, str_replace_editor, think and finish. Tool calls use Qwen-Coder XML. Teacher is Qwen3-Coder-480B. Median length is 38,052 tokens, mean 41,398, P99 91,150; successful runs average 48-56 steps. Tasks sharing a repo+base_commit or problem statement with SWE-bench Verified were removed. Only permissively licensed repos (MIT, Apache, BSD, ISC; checked with scancode) were kept. Qwen3-32B trained on it goes from 23.0% to 59.4% on SWE-bench Verified, and Qwen3-4B reaches 43.0% — [Together blog](https://www.together.ai/blog/coderforge-preview); [HF card](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview). The card lists these limitations: one scaffold and tool set (may not transfer to other harnesses), mostly bug-fixing tasks, and no user messages mid-trajectory — [HF card](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview)
- **`Lego-X/SWE-Lego-Synthetic-Data`**: Apache-2.0. 11.5K synthetic Python issues, split into `resolved` (9.1K) and `unresolved` (2.4K) parquet files, with a `messages` column. Made with Qwen3-Coder-480B on OpenHands v0.53.0. SWE-Lego-Qwen3-8B, trained with SFT only, reaches 42.2% pass@1 on SWE-bench Verified. Built on SWE-rebench and SWE-smith. Note that the card's own code loads `SWE-Lego/SWE-Lego-Synthetic-Data`, while the repo id verified in search is `Lego-X/...` — [HF card](https://huggingface.co/datasets/Lego-X/SWE-Lego-Synthetic-Data)
- **`nvidia/Nemotron-SWE-v1`**: CC-BY-4.0 (some subsets BSD-3). 51,029 OpenHands trajectories in an `r2e_gym` subset (the card text also says "59k"), about 11 GB, JSONL. Teacher is Qwen3-Coder-480B. Commercial use OK — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1)
- **`nvidia/Nemotron-SFT-SWE-v2`**: CC-BY-4.0 plus Apache, MIT and BSD notices. `openhands_swe` has 46,278 trajectories (Qwen3-Coder-480B; issues from SWE-Gym and R2E-Gym). `agentless_swe` has 209,976 single-turn localization, repair and test-generation samples from DeepSeek-R1-0528. About 17 GB, commercial use OK — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2)
- **`nvidia/Nemotron-SFT-SWE-v3`**: CC-BY-4.0 plus Apache, MIT and BSD. 237,970 samples, about 11.7 GB. Uses three harnesses (OpenHands, SWE-agent and mini-SWE-agent), so it adds scaffold diversity. Created 2026-06-04, commercial use OK — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3)
- **`nvidia/Nemotron-SFT-SWE-v3.5`**: CC-BY-4.0. 5,115 records, 1.9 GiB, so trajectories are very long. Multi-file and multi-language tasks (Python, TS, JS, Java, Go, Rust) covering code, tests, docs and config. Lists OpenCode as a reference harness. The card says "for commercial uses only", which is probably a typo — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3.5)
- **`nvidia/SWE-Zero-openhands-trajectories`**: CC-BY-4.0. 318,115 OpenHands trajectories over 118,092 issues (SWE-Gym, SWE-Gym-Raw, R2E-Gym-Subset, SWE-Fixer-Train-110K, SWE-rebench), all from MIT, Apache or BSD repos. Teacher is Qwen3-Coder-480B. Execution-free: per the title and report (arXiv 2604.01496), these are not test-verified. The rows have no `resolved` field — [HF card](https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories)
- **`allenai/SERA-4.6-Lite-Best-Subset`**: 47,498 samples taken from Sera-4.6-Lite-T1 and T2, filtered to model-submitted samples and a truncation ratio of 0.88 at 32K tokens. Reports 50.67% on SWE-bench Verified at 32K context. The card says the samples are not test-verified (`target_patch` is provided for optional checking). License conflict: the HF tag says Apache-2.0, but the card body says ODC-By, research and education use — [HF card](https://huggingface.co/datasets/allenai/SERA-4.6-Lite-Best-Subset)

**Second-tier / older / smaller**

- **`nebius/SWE-agent-trajectories`**: CC-BY-4.0. 80,036 SWE-agent trajectories from several open models (Qwen2.5-72B, Llama3-70B). Only 13,389 resolved. Resolved runs average 31.3 steps and 8,352 tokens of context (Llama-3 tokenizer); unresolved runs average 58.4 steps and 15,241 tokens. The short resolved runs suit a small context budget — [HF card](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- **`SWE-Gym/OpenHands-Sampled-Trajectories`**: no license tag on the HF repo, though the parent `SWE-Gym/SWE-Gym` is MIT. 6,055 rows, of which only 491 succeeded per the Nebius table. Average 18.9 turns — [HF API](https://huggingface.co/api/datasets/SWE-Gym/OpenHands-Sampled-Trajectories); [Nebius card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
- **`R2E-Gym/R2EGym-SFT-Trajectories`**: no license tag on the repo (`R2E-Gym/R2E-Gym-Subset` is Apache-2.0). 3,231 rows, about 155 MB, all successful, 16.1 average turns, Claude-Sonnet-3.5-v2 teacher — [HF API](https://huggingface.co/api/datasets/R2E-Gym/R2EGym-SFT-Trajectories); [Nebius card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
- **`zai-org/SWE-Dev-train`**: no license tag. `sft` split has 17,871 rows and `rft` split 2,276. OpenHands-based (SWE-Dev paper arXiv 2506.07636; SWE-Dev-7B scores 23.4% and 32B scores 36.6% on SWE-bench Verified). The card links to the old id `THUDM/SWE-Dev-train` — [HF card](https://huggingface.co/datasets/zai-org/SWE-Dev-train)
- **`nvidia/Nemotron-Cascade-SFT-SWE`**: CC-BY-4.0. Single-turn agentless localization, repair and test generation from DeepSeek-R1-0528. SWE-bench Verified repos excluded. Not multi-turn — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-SWE)
- **`thoughtworks/agentic-coding-trajectories`**: license `other` (derivative). Re-tokenizes 15,000 sessions, 5K each from Nebius SWE-rebench-openhands, SWE-smith Claude 3.7 and Kwai-Klear. Useful length statistics (cl100k): p50 deepest-prompt length is 23,840 tokens, p95 69,274, p99 89,876. Turns: p50 36, p95 87. 32% of sessions fall in 32-64K and 7% exceed 64K — [HF card](https://huggingface.co/datasets/thoughtworks/agentic-coding-trajectories)

**Terminal / CLI-harness trajectories (closest match to the Fable failure mode)**

- **`nvidia/Nemotron-Terminal-Corpus`**: CC-BY-4.0, 100K-1M rows. About 366K execution trajectories: about 226K are "dataset adapters" (math, code and SWE converted to terminal format) and about 140K are skill-based synthetic tasks (easy/medium/mixed). Paper is arXiv 2602.21193. Training Qwen3-8B on it raises Terminal-Bench 2.0 from 2.5% to 13.0%; Qwen3-14B goes from 4.0% to 20.2% — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus). Companion dataset: `nvidia/Nemotron-Terminal-Synthetic-Tasks` (CC-BY-4.0, task definitions) — [HF API search](https://huggingface.co/api/datasets?search=Nemotron-Terminal)
- **`laion/nemotron-terminal-security`** (CC-BY-4.0, 10K-100K rows) is the `security` skill partition of Nemotron-Terminal-Corpus, with a `conversations` column and difficulty labels. LAION also publishes other per-skill splits, e.g. `laion/nemotron-terminal-file_operations` and `laion/nemotron-terminal-scientific_computing` — [HF card](https://huggingface.co/datasets/laion/nemotron-terminal-security)
- **`open-thoughts/OpenThoughts-Agent-v1-SFT`**: Apache-2.0. About 15,200 traces from the Terminus-2 harness (the Terminal-Bench harness), max 32 turns, 64K context, GLM-4.6-AWQ teacher. Sources are nl2bash plus InferredBugs (C# and Java). The SFT-plus-RL model built from Qwen3-8B scores 4.9% on Terminal-Bench 2.0 and 15.7% on SWE-bench Verified. The companion RL set is `open-thoughts/OpenThoughts-Agent-v1-RL` (about 720 tasks) — [HF card](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT)
- `harborframework/terminal-bench`, `-2.0`, `-2.1` and `-3.0` (Apache-2.0) are **evaluation** task sets, not training data. They are listed here so they can be excluded during decontamination — [HF API search](https://huggingface.co/api/datasets?search=terminal-bench)

**Unverifiable / not found (flagged)**

- `Skywork/Skywork-SWE-trajectories` and `Skywork/Skywork-SWE-32B-trajectories`: the HF API returns 401 (do not exist publicly). Searching "Skywork-SWE" only finds third-party **eval** traces of the Skywork-SWE-32B model (DCAgent/DCAgent2 repos), not training data — [HF API search](https://huggingface.co/api/datasets?search=Skywork-SWE)
- `moonshotai/Kimi-Dev`: 401. No Kimi-Dev training-trajectory dataset was found, only eval traces of Kimi-Dev-72B — [HF API search](https://huggingface.co/api/datasets?search=Kimi-Dev)
- `all-hands/openhands-feedback` did not resolve; it now lives at **`OpenHands/openhands-feedback`** (MIT, <1K rows, last modified 2024-12) — [HF API search](https://huggingface.co/api/datasets?search=openhands-feedback)
- `ml-foundations-dev/terminal-bench-traces`: 401 (not found).

### Inferences
- Best fit for a CLI harness on Colab: Kwai-Klear mini-swe-agent-plus (bash-only, all resolved, MIT), Nemotron-Terminal-Corpus (terminal-native, CC-BY), and OpenThoughts-Agent-v1-SFT (Terminus-2, Apache). Then resolved rows from `nebius/SWE-rebench-openhands-trajectories` (`resolved==1`) and SWE-smith-trajectories (`tool` config, resolved only).
- At 30-40K tokens median per trajectory, 10K full trajectories is about 300-400M tokens per epoch, which is probably too much for about 200 Colab compute units on one A100. Filtering to trajectories of 16K tokens or fewer (roughly the 35% of sessions under 16K in the thoughtworks distribution), or using nebius/SWE-agent-trajectories resolved runs (about 8K tokens average), is the practical route. Mixing scaffolds (SWE-agent, OpenHands, mini-swe-agent, Terminus) addresses CoderForge's stated single-scaffold limitation.
- Normalize all of them to one OpenAI-style `messages`+`tool_calls` schema and the target model's chat template. Loss-mask tool and observation turns.
- Execution-free sets (SWE-Zero, SERA) have weaker quality guarantees than resolved-filtered ones and should be lower priority for teaching feedback loops.

### Gaps
- Exact per-dataset token-length distributions were found only for CoderForge, nebius/SWE-agent-trajectories and the thoughtworks re-tokenization. The others need a local tokenization pass.
- The resolved/unresolved split inside the Nemotron-SFT-SWE v1, v2 and v3 OpenHands subsets is not stated on the cards.
- No license tag on CoderForge (dataset level), SWE-Gym OpenHands-Sampled, R2EGym-SFT or zai-org/SWE-Dev-train. The license has to be inferred from upstream or the authors asked.

## 2. Function / tool calling (multi-turn)

### Takeaway
Permissive, multi-turn, tool-use SFT data is plentiful. NVIDIA Nemotron-Agentic sets (CC-BY-4.0) are the largest. Hermes, ToolACE and Glaive are Apache-2.0. APIGen-MT-5k is non-commercial and gated, so it is unsuitable for a fully open release. When2Call teaches when *not* to call a tool.

### Cited Findings
- **`nvidia/Nemotron-Agentic-v1`**: CC-BY-4.0, commercial OK. `interactive_agent` has 19,028 multi-turn rows (user/agent/tool simulated, LLM-judge filtered). `tool_calling` has 316,094 rows. About 5.5 GB, JSONL — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1)
- **`nvidia/Nemotron-SFT-Agentic-v2`**: CC-BY-4.0 plus Apache and MIT. `tool_calling` 707,052 rows, `customer_service_838_domains` 278,880 rows, `search_graph_walk` 5,968 rows. Simulated and judged with DeepSeek-V3.2 and GLM-4.6. About 20 GB — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2)
- **`nvidia/Nemotron-SFT-Agentic-v4`** (created 2026-10-09): license **OpenMDW-1.1**, not CC-BY. `tool_calling` 549,707 rows and `knowledge_agent` 84,957 rows. Generated with Qwen3.5-397B-A17B and GLM-5.2. Schema is `messages`, `tools` and `parallel_tool_calls`. Assistant turns carry `reasoning_content`, which can be removed for non-thinking training. The card says "commercial or non-commercial uses" — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v4)
- **`Salesforce/APIGen-MT-5k`**: **CC-BY-NC-4.0, gated**. 5,000 multi-turn τ-bench-style dialogues (retail and airline domains) from GPT-4o and DeepSeek-V3, with 3-stage verification. ShareGPT-like JSON. Because the domains are τ-bench's, there is a contamination risk if τ-bench is an evaluation target — [HF card](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)
- **`Salesforce/xlam-function-calling-60k`**: CC-BY-4.0 tag, gated (auto-approve), 10K-100K rows. Mostly single-turn — [HF API](https://huggingface.co/api/datasets/Salesforce/xlam-function-calling-60k)
- **`NousResearch/hermes-function-calling-v1`**: Apache-2.0. Files: `func-calling.json` (multi-turn), `glaive-function-calling-5k.json`, `json-mode-agentic.json`, `json-mode-singleturn.json`. ShareGPT `from`/`value` format with Hermes XML `<tool_call>` tags. Last modified 2026-01-03 — [HF card](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- **`Team-ACE/ToolACE`**: Apache-2.0, 10K-100K rows, English and Chinese. API pool of 26,507 APIs with dual rule- and model-based verification. Strong BFCL results — [HF card](https://huggingface.co/datasets/Team-ACE/ToolACE)
- **`glaiveai/glaive-function-calling-v2`**: Apache-2.0, 100K-1M rows, from 2023 — [HF API](https://huggingface.co/api/datasets/glaiveai/glaive-function-calling-v2)
- **`nvidia/When2Call`**: CC-BY-4.0, commercial OK. Configs: `train_sft`, `train_pref` (chosen/rejected), and `test` (mcq 3,652 / llm_judge 300). Teaches when to call a tool, when to ask a follow-up, and when to refuse. **Do not train on the `test` config** — [HF card](https://huggingface.co/datasets/nvidia/When2Call)
- **`nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1`** and **`...-Conversational-Tool-Use-Pivot-v1`** (CC-BY-4.0) are RL environments, not SFT traces — [HF API search](https://huggingface.co/api/datasets?search=Nemotron-Agentic)
- ToolBench: `ToolBench/ToolBench` returns 401 (no official HF repo). Only community mirrors exist, e.g. `Maurus/ToolBench` and `tuandunghcmut/toolbench-v1` (Apache-2.0 tag) — [HF API search](https://huggingface.co/api/datasets?search=ToolBench)

### Inferences
- For an open-weight release: Nemotron-Agentic-v1 `interactive_agent` (true multi-turn), Hermes `func-calling.json`, ToolACE and When2Call `train_sft`, at a few thousand samples total. Use APIGen-MT-5k only for research ablations because of the NC license.
- Tool-call formats differ (Hermes XML vs OpenAI JSON vs Qwen XML), so convert everything to one format that matches the inference harness.

### Gaps
- Average turns per dialogue were not stated on the Hermes, ToolACE or Glaive cards.
- BFCL-specific training sets were not separately verified.

## 3. Coding reasoning (single-turn, test-verified)

### Takeaway
The large NVIDIA and Microsoft sets (CC-BY-4.0) are the permissive choice. KodCode is test-verified but CC-BY-NC. CodeFeedback (used by V4) is Apache-2.0 but older and unverified. In a 10K-40K agentic mix, single-turn coding should be a minority slice, kept to preserve code-writing quality.

### Cited Findings
- **`nvidia/OpenCodeReasoning`**: CC-BY-4.0. 735,255 Python samples over 28,319 competitive-programming questions, R1-generated responses. Split rows are 567,850 and 167,405 — [HF card](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)
- **`nvidia/OpenCodeReasoning-2`**: CC-BY-4.0. 1.4M Python and 1.1M C++ samples over 34,799 questions. R1 solutions plus QwQ critiques. Has a `pass_rate` field (-1 when tests are insufficient), so it can be filtered to pass_rate == 1. Some questions must be pulled from original sources by script, each under its own license — [HF card](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2)
- **`nvidia/OpenCodeInstruct`**: CC-BY-4.0, 5M samples, with an `average_test_score` field (fraction of tests passed) for filtering — [HF card](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)
- **`microsoft/rStar-Coder`**: CC-BY-4.0. 418K problems and 580K long-reasoning solutions, with `verified` and `is_passed` flags and test-case subsets — [HF card](https://huggingface.co/datasets/microsoft/rStar-Coder)
- **`KodCode/KodCode-V1-SFT-R1`**: **CC-BY-NC-4.0**. About 268K/211K/4.4K rows by split, 12 subsets, R1 responses verified against unit tests. `KodCode/KodCode-V1` is also CC-BY-NC — [HF card](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1)
- **`m-a-p/CodeFeedback-Filtered-Instruction`**: Apache-2.0, 100K-1M rows, last modified 2024-02 — [HF API](https://huggingface.co/api/datasets/m-a-p/CodeFeedback-Filtered-Instruction)
- **`open-thoughts/OpenThoughts3-1.2M`**: Apache-2.0, 1M-10M rows (mixed math, code and science reasoning) — [HF API](https://huggingface.co/api/datasets/open-thoughts/OpenThoughts3-1.2M)
- `nvidia/Nemotron-Post-Training-Dataset-v1` (CC-BY-4.0, 10M-100M rows) and `-v2` (CC-BY-4.0, gated auto-approve, 1M-10M rows) are broad post-training blends that include code — [HF API](https://huggingface.co/api/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)

### Inferences
- Use OpenCodeReasoning-2 (pass_rate == 1) or rStar-Coder (`is_passed`) for a small (about 2-5K), test-verified coding slice under a permissive license. Prefer the non-thinking or short-response variants if the target model does not use long CoT, since R1 traces are very long.
- Drop KodCode from an open release unless the NC license is acceptable.

### Gaps
- No major 2026 single-turn coding-reasoning release was verified in this pass beyond the Nemotron families.

## 4. Defensive cybersecurity / secure coding / prompt-injection defense

### Takeaway
Permissive, defensive-oriented instruction data exists (Trendyol, Fenrir, AttackQA, Nemotron-Terminal security split, NVIDIA's IPI set). Secure-code preference data is weak. The CyberNative DPO set's own maintainers now report that training on it badly damaged coding ability with no security gain. Primus is ODC-By but gated. Several "securecode" sets are NC-SA.

### Cited Findings
- **`CyberNative/Code_Vulnerability_Security_DPO`**: Apache-2.0, 4,656 pairs, 11 languages. **Correction notice dated 2026-10-06 and 2026-11-01**: the `chosen`/`rejected` labels are unverified model outputs. In the maintainers' own test, DPO on Qwen2.5-Coder-1.5B (LoRA r16) cut HumanEval+MBPP pass@1 from 59.8% to 21.5%, with no measured security gain on CyberSecEval/CodeShield. CodeShield also flags 12.5% of the `chosen` answers. **Recommendation: exclude** — [HF card](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO)
- **`Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset`**: Apache-2.0. v2.0 (2025-07-30) has 53,202 system/user/assistant examples across 200+ defensive domains (threat intelligence, IR, SOC, cloud and K8s forensics, MITRE ATT&CK, NIST) — [HF card](https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset)
- **`AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1`**: Apache-2.0. v2.1.0 (2026-04-22) has 99,870 system/user/assistant rows. It claims "adversarial refusal tests" and public-source provenance. The card is self-promotional and unverified — [HF card](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1)
- **`sambanovasystems/attackqa`**: Apache-2.0, 10K-100K rows, QA derived from MITRE ATT&CK (subject to the ATT&CK terms of use) — [HF card](https://huggingface.co/datasets/sambanovasystems/attackqa)
- **`trendmicro-ailab/Primus-Instruct`** (<1K rows), **`Primus-Reasoning`** (1K-10K) and **`Primus-FineWeb`** (1M-10M, pretraining text): ODC-By, **gated** (auto-approve; the README needs login) — [HF API](https://huggingface.co/api/datasets/trendmicro-ailab/Primus-Instruct)
- **`laion/nemotron-terminal-security`**: CC-BY-4.0, 10K-100K multi-turn terminal trajectories on security skills, taken from Nemotron-Terminal-Corpus. This is the only agentic, execution-feedback security set found — [HF card](https://huggingface.co/datasets/laion/nemotron-terminal-security)
- **`nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1`**: CC-BY-4.0, 1K-10K rows, created 2026-04-01. An **RL/Gym** dataset (with `verifier_config`), not SFT traces. Tool-using agents learn to resist injections hidden in tool outputs. Fully synthetic — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1)
- Prompt-injection classifier and label sets (not chat SFT): `deepset/prompt-injections` (Apache-2.0, <1K), `neuralchemy/Prompt-injection-dataset` (Apache-2.0), `Lakera/mosscap_prompt_injection` (MIT) — [HF API search](https://huggingface.co/api/datasets?search=prompt-injection)
- NC-licensed (avoid for an open release): `scthornton/securecode-web` and `scthornton/securecode` (CC-BY-NC-SA-4.0) — [HF API search](https://huggingface.co/api/datasets?search=secure-code)
- CyberSecEval-derived repos (`walledai/CyberSecEval` MIT; `penfever/cyberseceval*-harbor`) are **evaluation** sets. Exclude them from training — [HF API search](https://huggingface.co/api/datasets?search=CyberSecEval)
- `nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0` exists (license `other`, 10K-100K rows). Its terms were not reviewed — [HF API search](https://huggingface.co/api/datasets?search=Nemotron-Agentic)

### Inferences
- V4's `sh111111111111111/agentic_red_team` is offensive-oriented. For a "defensive cybersecurity" model, replace it with Trendyol (or Fenrir) plus a small security-terminal slice from `laion/nemotron-terminal-security`. Add a few hundred synthetic IPI-resistance SFT examples, e.g. made by rolling out against the NVIDIA IPI environments. All are permissive.
- Keep security instruction data to a small share of the mix (for example 5-10%). The CyberNative result shows that low-quality security-labelled code data can wreck functional coding.

### Gaps
- The quality of Trendyol and Fenrir answers was not independently audited. Both are synthetic and self-described.
- No open, permissive multi-turn *secure-coding* trajectory set (agent fixes a vulnerability and runs tests) was found.

## 5. General-capability replay

### Takeaway
Small slices of smoltalk (Apache-2.0 for its new subsets) or smoltalk2 (which also contains tool-calling traces) are the cleanest replay sources. Tulu-3 is ODC-By but includes some non-commercial portions.

### Cited Findings
- **`HuggingFaceTB/smoltalk`**: 1M-10M rows. The new subsets (Smol-Magpie-Ultra, Smol-constraints, Smol-rewrite, Smol-summarize) are Apache-2.0. Reused public datasets keep their original licenses — [HF card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- **`HuggingFaceTB/smoltalk2`**: 1M-10M rows, last modified 2025-10-31. Has `Mid`, `SFT` and preference configs, including `smolagents_toolcalling_traces_think`, `hermes_function_calling_v1_no_think` and `xlam_traces_no_think` splits, with think/no_think variants. New subsets are Apache-2.0 per the card — [HF card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)
- **`allenai/tulu-3-sft-mixture`**: ODC-BY-1.0, 939,344 samples, `messages` format. The card warns: "Some portions of the dataset are non-commercial" — [HF card](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)

### Inferences
- About 5-10% replay (1-3K samples) from smoltalk's Apache-licensed subsets should be enough to limit forgetting in a LoRA or short full-FT run. Filter Tulu by per-row source to drop the NC portions.

### Gaps
- Per-subset licenses inside smoltalk2 were not individually verified.

## 6. Contamination guidance

### Takeaway
Decontaminate by source identifiers (repo + PR or issue number, base_commit, fix patch), not just n-grams. Exclude every benchmark task set that will be used for evaluation, including the user's own sandbox tasks.

### Cited Findings
- The agentleak audit found 348 of the 32,079 SWE-rebench-V2 tasks (1.1%) overlap Multi-SWE-bench, SWE-PolyBench or SWE-bench Multilingual. The share is 5.1% of Multi-SWE-bench, 11.3% of SWE-PolyBench and 4.0% of SWE-bench Multilingual. None of the 12 SWE-bench Verified/test repos appear in V2. SWE-Gym, R2E-Gym and SWE-smith showed no shared repos or PRs with the benchmarks checked. A 13-gram filter alone missed overlap when problem statements were LLM-rewritten. Recommendations: match on repo+PR, base commit and fix patch; use the published `v2_not_in_benchmarks.txt` list (31,731 tasks) — [agentleak GitHub](https://github.com/raimondasl/agentleak)
- The `nebius/SWE-rebench-V2` card itself says it excluded original SWE-bench repos but did not filter other benchmarks, and points to the agentleak overlap list — [HF card](https://huggingface.co/datasets/nebius/SWE-rebench-V2)
- CoderForge removed tasks sharing a (repo, base_commit) pair or problem statement with SWE-bench Verified — [Together blog](https://www.together.ai/blog/coderforge-preview)
- Nemotron-Cascade-SFT-SWE excluded every instance from repos present in SWE-bench Verified — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-SWE)
- Terminal-Bench eval sets are public on HF (`harborframework/terminal-bench-2.0` etc.). Nemotron-Terminal-Corpus and OpenThoughts-Agent report Terminal-Bench 2.0 scores, so their task generators were presumably kept separate, but neither card shown states an explicit TB decontamination step — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus); [HF card](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT)

### Inferences
- Practical recipe:
  1. Drop any training row whose `instance_id`, repo+PR or base_commit matches SWE-bench Verified, Lite, Multilingual or Pro, Multi-SWE-bench, or SWE-PolyBench.
  2. Run a 13-gram plus embedding check of problem statements against Terminal-Bench 2.x/3.0 task instructions and the in-house sandbox tasks.
  3. Never train on eval configs (When2Call `test`, CyberSecEval, Terminal-Bench, BFCL).
  4. Run an exact-match check of final patches against benchmark gold patches.

### Gaps
- No published overlap audit between Nemotron-Terminal-Corpus or OpenThoughts-Agent and Terminal-Bench 2.x/3.0 was found.
- No contamination analysis was found for the Nemotron-SFT-SWE v2, v3 or v3.5 trajectories against SWE-bench Verified beyond what the cards cite.
