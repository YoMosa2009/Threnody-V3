# Base Model Candidates for "Threnody" (local agentic coding model, Nov 2026)

Context: Fable-Coder-V4 = LoRA of Qwen/Qwen2.5-Coder-7B-Instruct (72.7% vs base 70.7% on a small custom benchmark). Target: runs on a 12GB GPU (RTX 3060) and/or 16-32GB system RAM; fine-tunable on Colab (A100 40/80GB, L4 24GB, T4 16GB; ~200 compute units). Research date: 2026-11-02. About 16 search/fetch calls; the main facts come from official Hugging Face model cards plus a few secondary guides, flagged below.

## Which models exist with open weights as of Nov 2026 (repo id, date, license)

### Takeaway
The 2025 candidates in the brief (Qwen3-2507, Qwen3-Coder-30B-A3B, Devstral Small 1.x) have been replaced. Qwen shipped Qwen3.5 (Feb-Mar 2026), Qwen3.6 (Apr 2026) and Qwen3.8 (Aug 2026, only 27B dense + 2.4T flagship). Google shipped Gemma 4 (E2B/E4B/12B/26B-A4B/31B). Z.ai shipped GLM-4.7-Flash (30B-A3B). Mistral shipped Devstral Small 2 (24B). NVIDIA shipped Nemotron 3 Nano 30B-A3B and later a "Nemotron 3.5 Lightning 30B-A3B". No Qwen3.8 9B or 35B-A3B has been officially released.

### Cited Findings
- **Qwen3.5 family**: Qwen3.5-397B-A17B released 2026-02-16; 122B-A10B, **35B-A3B**, **27B** on 2026-02-24; **9B, 4B, 2B, 0.8B** on 2026-03-02 (HF + ModelScope) — [QwenLM GitHub news](https://github.com/QwenLM/Qwen3.8); [Unsloth Qwen3.5 collection](https://huggingface.co/collections/unsloth/qwen35)
- **Qwen3.6**: Qwen3.6-35B-A3B released 2026-04-16 (`Qwen/Qwen3.6-35B-A3B`); Qwen3.6-27B (dense) released 2026-04-22 — [QwenLM GitHub news](https://github.com/QwenLM/Qwen3.8). License Apache-2.0 — [HF card Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Qwen3.8**: Qwen3.8-2.4T-A95B (2026-08-12) and `Qwen/Qwen3.8-27B` (2026-08-14), Apache-2.0 — [QwenLM GitHub](https://github.com/QwenLM/Qwen3.8); [HF card Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **No official Qwen3.8 35B-A3B or 9B.** A mid-August tweet points to an ms-swift commit listing a 35B-A3B, but it was "not officially announced". HF discussion threads on the 27B repo are requesting one — [X/@analogalok](https://x.com/analogalok/status/2088587801944273145); [HF discussion #120](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/120). Community repos such as "Qwen3.8-Distill-35B-A3B-Coder-Abliterated" are **unofficial distills, not Qwen releases** — [HF search](https://huggingface.co/models?other=qwen3_5_moe_text)
- Qwen3-Next-80B-A3B (2025-09-11) also exists. At 80B total it is too large for this target — [QwenLM GitHub](https://github.com/QwenLM/Qwen3.8)
- **Gemma 4**: it exists. Sizes are E2B (2.3B effective, 5.1B with embeddings), E4B (4.5B eff, 8B with embeddings), 12B "Unified" (11.95B), 26B-A4B MoE (25.2B total / 3.8B active, 128 experts with 8 active + 1 shared), and 31B dense (30.7B). License Apache 2.0 — [HF google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it); other repos: [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it), [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it), [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it). Exact release date not on card.
- **GLM-4.7-Flash**: `zai-org/GLM-4.7-Flash`, 30B-A3B MoE (~3B active) — [HF card](https://huggingface.co/zai-org/GLM-4.7-Flash)
- **Devstral Small 2**: `mistralai/Devstral-Small-2-24B-Instruct-2512` (Dec 2025), 24B dense, Apache 2.0. The larger Devstral 2 (123B) is modified-MIT with a revenue cap — [Livedoor news](https://news.livedoor.com/article/detail/30174682/); [Unsloth GGUF](https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF); [letsdatascience](https://letsdatascience.com/news/mistral-releases-devstral-2-for-code-agents-2e24c245)
- **NVIDIA Nemotron 3 Nano**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, released 2025-12-15, NVIDIA Nemotron Open Model License (not Apache) — [NVIDIA build card](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard). A newer `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` also exists — [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
- **gpt-oss-20b** is still a reference point. Qwen and Z.ai cards use it as a comparison baseline — [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B); [GLM-4.7-Flash card](https://huggingface.co/zai-org/GLM-4.7-Flash)
- Qwen3-Coder-Next (80B total / 3B active) exists; Q4 is ~35-40 GB, so it does not fit the target — [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/)

### Inferences
- Qwen3.5 and Qwen3.6 are architecturally one family (hybrid Gated DeltaNet + Gated Attention, MTP, vision encoder, 262K context). Tooling such as Unsloth or llama.cpp that supports one likely supports the other.
- Qwen3.8-27B is the newest dense model in the size range. There is no small (≤12GB-fit) Qwen3.8 yet, so for small sizes Qwen3.5-9B/4B are still the newest Qwen.

### Gaps
- I did not re-verify these this session: Qwen3-2507 4B/30B-A3B, Qwen3-Coder-30B-A3B-Instruct, GLM-4.5-Air/4.6, Mistral Small 3.x, Devstral Small 1.0/1.1, IBM Granite 4, Nemotron Nano 9B v2, Seed-Coder/Seed-OSS-36B, LFM2, Kimi/MiniMax small variants, Phi-4, SmolLM3. All existed by late 2025 (from prior knowledge), but newer releases above outclass them on every reported agentic metric I found. Treat them as unverified in this note.
- License for GLM-4.7-Flash not confirmed this session (earlier GLM releases were MIT; check the card).
- Gemma 4 and Qwen3.6-27B exact HF release dates not captured.

## Specs: params, active params, context, tool-calling, thinking modes, license

### Takeaway
Every strong candidate defaults to thinking-on with a switch to turn it off. The Qwen 3.5/3.6/3.8 models share a 262K native context, Apache-2.0, and the `qwen3_coder` tool parser (the same XML-style format as Qwen3-Coder). That format is a clean target for Claude-Code-style harnesses.

### Cited Findings
- **Qwen3.6-35B-A3B**: 35B total / 3B active (256 experts, 8 routed + 1 shared), 40 layers in a 3:1 Gated DeltaNet : Gated Attention hybrid, vision encoder. Native context 262,144, YaRN to ~1.01M. Thinking on by default; `enable_thinking: False` turns it off; `preserve_thinking` keeps earlier reasoning. vLLM/SGLang tool parser `qwen3_coder`. Apache-2.0 — [HF card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- **Qwen3.8-27B**: 27B dense (Hub shows 28B), same 3:1 DeltaNet/Attention hybrid, MTP, vision. Native 262,144, YaRN to 1M. Thinking on by default with `reasoning_effort` xhigh/medium/low; `preserve_thinking` on by default. Tool parser not stated on card. Apache-2.0 — [HF card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Qwen3.5-9B**: 9B (Hub metadata says 10B incl. vision), 32 layers with the same hybrid, hidden 4096, MTP, 262K native (YaRN to 1.01M), thinking on by default with `<think>` and `enable_thinking: False`. Apache-2.0 — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Gemma 4**: context 128K (E2B/E4B) and 256K (12B/26B-A4B/31B). Thinking turns on with `<|think|>` in the system prompt and outputs a `<|channel>thought ... <channel|>` block. The card claims "native support for structured tool use" but does not document the format. Apache 2.0 — [HF card](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **GLM-4.7-Flash**: 30B total / ~3B active. Card's SWE-bench/Terminal-Bench eval uses temp 0.7, top-p 1.0, 16,384 max new tokens — [HF card](https://huggingface.co/zai-org/GLM-4.7-Flash)
- **Devstral Small 2**: 24B dense, Apache 2.0 — [letsdatascience](https://letsdatascience.com/news/mistral-releases-devstral-2-for-code-agents-2e24c245)

### Inferences
- The `qwen3_coder` tool format makes the Qwen 3.5/3.6 bases a natural fit for OpenCode/OpenHands/Claude-Code-style harnesses, which vLLM/llama.cpp already serve with that parser.
- The hybrid linear-attention design keeps KV cache small. That helps long-horizon multi-turn sessions on 12GB more than in older pure-attention models (e.g. Qwen2.5-Coder-7B).

### Gaps
- Context and tool format for Devstral Small 2 and Nemotron 3 Nano not pulled from primary cards this session (prior knowledge: Devstral Small 2 ~256K, Mistral tool tokens; Nemotron 3 Nano hybrid Mamba-Transformer). Unverified here.
- Gemma 4's exact tool-call token format is undocumented on the card.

## Reported benchmark scores (SWE-bench Verified, Terminal-Bench, BFCL, tau2, LiveCodeBench, Aider)

### Takeaway
Among models in the size range, Qwen3.6/3.8 lead clearly on vendor-reported agentic coding. Qwen3.6-35B-A3B: SWE-V 73.4, TB2.0 51.5. Qwen3.6-27B: SWE-V ~77.2, TB2.0 ~59.3. Qwen3.8-27B: TB2.1 73.0, SWE-Pro 61.7. Gemma 4 is strong on LiveCodeBench and tau2 but weak on SWE-bench by Qwen's own comparison. Every number below is vendor-reported with an internal scaffold, and none comes from an independent leaderboard.

### Cited Findings
**Qwen3.6-35B-A3B card table.** SWE-bench uses an internal scaffold (bash + file-edit tools), temp 1.0, 200K context. TB2.0 uses Harbor/Terminus-2, 3h timeout, average of 5 runs — [HF card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B):

| Benchmark | Qwen3.6-35B-A3B | Qwen3.5-27B | Qwen3.5-35B-A3B | Gemma4-31B | Gemma4-26B-A4B |
|---|---|---|---|---|---|
| SWE-bench Verified | 73.4 | 75.0 | 70.0 | 52.0 | 17.4 |
| SWE-bench Multilingual | 67.2 | 69.3 | 60.3 | 51.7 | 17.3 |
| SWE-bench Pro | 49.5 | 51.2 | 44.6 | 35.7 | 13.8 |
| Terminal-Bench 2.0 | 51.5 | 41.6 | 40.5 | 42.9 | 34.2 |
| TAU3-Bench | 67.2 | 68.4 | 68.9 | 67.5 | 59.0 |
| MCPMark | 37.0 | 36.3 | 27.0 | 18.1 | 14.2 |
| MCP-Atlas | 62.8 | 68.4 | 62.4 | 57.2 | 50.0 |
| LiveCodeBench v6 | 80.4 | 80.7 | 74.6 | 80.0 | 77.1 |

- **Conflict**: the Qwen3.6 card gives Qwen3.5-27B 75.0 on SWE-V, while a secondary guide lists 72.4 and gives 69.2 for Qwen3.5-35B-A3B (card says 70.0) — [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/). Prefer the card numbers.
- **Qwen3.6-27B**: SWE-V 77.2, TB2.0 59.3, LCB v6 83.9, SWE-Pro 53.5. Source is a secondary guide citing the model card — [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/). The Qwen3.8-27B card lists Qwen3.6-27B at TB **2.1** = 63.4, LCB 83.9, SWE-Pro 53.5 — [HF Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Qwen3.8-27B**: Terminal-Bench 2.1 (Terminus) 73.0; SWE-bench Pro 61.7 (**Claude Code harness**, 256K ctx); LiveCodeBench v6 90.3; in-house QwenSWEBench 79.0; DeepSWE 1.1 42.2; NL2Repo 42.3. **No SWE-bench Verified, BFCL, tau2 or Aider numbers reported** — [HF card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Qwen3.5-9B / 4B** (vs gpt-oss-20b, Qwen3-30B-A3B-Thinking-2507): LCB v6 65.6 / 55.8 (gpt-oss-20b 74.6; Qwen3-30B-A3B-Thinking 66.0); **BFCL-V4 66.1 / 50.3** (Qwen3-30B-A3B-Thinking 42.4); **TAU2-Bench 79.1 / 79.9** (Qwen3-30B-A3B-Thinking 41.9). **No SWE-bench Verified or Terminal-Bench reported** for 9B/4B — [HF Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Gemma 4** (Google card): LCB v6 31B 80.0 / 26B-A4B 77.1 / 12B 72.0 / E4B 52.0; Tau2 (avg of 3) 76.9 / 68.2 / 69.0 / 42.2. No SWE-bench, Terminal-Bench or BFCL on the card — [HF card](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **GLM-4.7-Flash**: SWE-bench Verified 59.2, against Qwen3-30B-A3B-Thinking-2507 22.0 and gpt-oss-20b 34.0 (same card) — [HF card](https://huggingface.co/zai-org/GLM-4.7-Flash)
- **Devstral Small 2**: SWE-V 68.0 (Mistral-reported). Reported as 56.4 under the open-source harness on the official leaderboard; one table says 65.8. Terminal-Bench 2 22.5 per HF card (another listing says 32.0) — [letsdatascience](https://letsdatascience.com/news/mistral-releases-devstral-2-for-code-agents-2e24c245); [search summary of bodegaone/morphllm](https://www.bodegaone.ai/local-llms)
- **Nemotron 3 Nano 30B-A3B**: NVIDIA reports "SWE-Bench (OpenHands)" 38.8, not labeled Verified. Nemotron 3.5 Lightning 30B-A3B reports SWE-V ~51.6-52.8 — [NVIDIA card](https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b/modelcard); [HF Nemotron-3.5-Lightning](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
- Qwen3-Coder-Next (80B-A3B): SWE-V 70.6, SWE-rebench Pass@5 64.6 — [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/)

### Inferences
- Against Qwen2.5-Coder-7B (Fable-V4's base), every 2026 Qwen candidate is a generational jump in agentic ability: tool-calling (BFCL), multi-turn tau-style tasks and SWE-type tasks. Base choice will likely matter far more than the +2pt LoRA gain V4 got.
- Qwen's own table puts Gemma 4 26B-A4B at 17.4 SWE-V. That points to a format or scaffold mismatch rather than true capability, but either way it makes Gemma 4 a risky base for harness work.
- Different benchmark versions (TB 2.0 vs 2.1; TAU3 vs tau2) block direct cross-generation comparison.

### Gaps
- No independent swebench.com / tbench.ai leaderboard or BFCL leaderboard entries fetched (the morphllm page returned HTTP 429). All scores are vendor-reported.
- No Aider Polyglot scores found for any 2026 candidate.
- No SWE-V for Qwen3.5-9B/4B; BFCL for Qwen3.6/3.8 not reported on cards.

## Memory footprint and speed on a 12GB GPU; is 30/35B-A3B + CPU expert offload practical?

### Takeaway
Yes, MoE offload works. Qwen3.6-35B-A3B UD-Q4_K_M runs at **~39 tok/s on an RTX 3060 12GB + 32GB DDR4** with `--n-cpu-moe 24` (9.8GB VRAM), and still ~36 tok/s with a 16K prompt. Dense 27B models (Q4 ~16-17GB) do not fit 12GB and slow to single-digit tok/s when offloaded. Qwen3.5-9B (Q4 ~6.6GB) fits fully with room for long context.

### Cited Findings
- **Qwen3.6-35B-A3B GGUF sizes**: UD-IQ1_M 10 GB; UD-Q3_K_M 16.6 GB; UD-Q4_K_M 22.1 GB; UD-Q4_K_XL 22.4 GB; UD-Q5_K_M 26.5 GB; UD-Q6_K 29.3 GB; Q8_0 36.9 GB — [insiderllm Qwen3.6 MoE guide](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/); [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/)
- **RTX 3060 12GB, 32GB DDR4-2133, headless, UD-Q4_K_M**: `--n-cpu-moe 40` → 28.1 tok/s (2.5 GB VRAM); 32 → 31.8 (6.1 GB); **24 → 38.9 tok/s (9.8 GB)**; 20 → fails to load. Context: 38.5 tok/s at 4K, 38.2 at 8K; 16K prompt decoded at 35.8 tok/s; prompt processing ~386-413 tok/s. With a display attached, use `-ncmoe 26-28` — [insiderllm Qwen3.6 MoE guide](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/)
- Same guide: ~611.5 MB of expert weights streamed per token at full offload. Recommended RAM is 32GB+ for Q4 offload, 64GB+ for Q6. A GTX 1650 4GB still did ~20 tok/s with all experts on CPU — [same](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/)
- RTX 5070 Ti 16GB + 32GB RAM with `--cpu-moe`: ~25-35 tok/s (community-reported) — [same](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/)
- **Qwen3.6-27B dense**: Q4_K_M GGUF ~16.8 GB. Reports: ~6 tok/s on an RTX 5080 16GB with aggressive quant; 25.6 tok/s via llama-server at 65K context on an unspecified (larger) GPU — [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/)
- **Qwen3.5-9B** Q4 ~6.6 GB; **Qwen3.5-4B** ~3.4 GB; Qwen3.5-27B ~16 GB; Qwen3.5-35B-A3B ~20 GB — [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/)
- **Devstral Small 2** Q4 ~14 GB; it needs about 3-bit to fit comfortably on 12GB — [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/); [search summary, zimaspace](https://shop.zimaspace.com/blogs/tech-ai-hub/best-local-coding-models-8gb-12gb-16gb-24gb-vram)
- RTX 3090 baseline for Qwen3.6-35B-A3B fully resident: ~158 tok/s, and Qwen3.6 is ~30% slower than Qwen3.5-35B-A3B on the same card per one tester — [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/)

### Inferences
- For "ordinary people with a 3060 + 32GB RAM", a 35B-A3B at Q4 with expert offload (~35-39 tok/s) is fast enough for agentic CLI loops. 32GB RAM is the floor, and a 16GB-RAM user cannot run it at Q4.
- A 9B dense model is the only top candidate that serves both hardware targets (12GB VRAM alone, or 16GB RAM CPU-only at reduced speed).
- A fine-tuned Threnody could ship in two sizes from the same family (9B and 35B-A3B), sharing one tool format and chat template.

### Gaps
- No measured tok/s for Qwen3.5-9B, GLM-4.7-Flash or Gemma 4 26B-A4B specifically on a 12GB card.
- Qwen3.5-9B CPU-only speed with 16GB RAM not measured in sources.

## Fine-tuning practicality on Colab (Unsloth, QLoRA for MoE, VRAM at 8k-16k)

### Takeaway
Unsloth supports Qwen3.5 (and by extension Qwen3.6) fine-tuning, but **advises against 4-bit QLoRA for all Qwen3.5 models and especially MoE**. Published bf16 LoRA VRAM: 9B ≈ 22GB (L4 is tight, A100 40GB comfortable), 27B ≈ 56GB (A100 80GB), 35B-A3B ≈ 74GB (A100 80GB only). These figures are at short sequence length. 8-16K sequences push 27B/35B beyond a single 80GB card unless context is reduced or heavy gradient checkpointing is used.

### Cited Findings
- Unsloth Qwen3.5 bf16 LoRA VRAM: 0.8B 3GB; 2B 5GB; 4B 10GB; **9B 22GB; 27B 56GB; 35B-A3B 74GB**; 122B-A10B 256GB. Full fine-tune ≈ 4x LoRA — [Unsloth Qwen3.5 fine-tune docs](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- "4-bit QLoRA on Qwen3.5 is not recommended" (dense or MoE) because quantization differences are higher than normal. MoE QLoRA is also limited by BitsandBytes — [Unsloth docs](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Free Colab notebooks exist for 0.8B/2B/4B (bf16 LoRA) and a 4B GRPO notebook. 27B and 35B-A3B have **A100 notebooks**, the 27B one specifying an 80GB A100 — [Unsloth docs](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Unsloth's MoE update claims ~12x faster MoE training, >35% less VRAM, ~6x longer context. Router-layer fine-tuning is off by default. The example uses `max_seq_length=2048` and says to scale up with "unsloth" gradient checkpointing — [Unsloth docs](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Qwen3.6: Unsloth says it can be run and fine-tuned in Unsloth Studio, and its HF uploads (`unsloth/Qwen3.6-27B`, `unsloth/Qwen3.6-35B-A3B`) list Colab notebook options. No dedicated Qwen3.6 Colab notebook was confirmed — [Unsloth Qwen3.6 docs](https://unsloth.ai/docs/models/qwen3.6); [unsloth/Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B)
- Unsloth's Qwen3 (2025) MoE (30B-A3B) fine-tuning support is documented — [Unsloth Qwen3 blog](https://unsloth.ai/blog/qwen3)

### Inferences
- **9B**: bf16 LoRA at ~22GB baseline fits an A100 40GB with headroom for 8-16K sequences, so it is the cheapest viable serious run. L4 24GB probably works only at short context. T4 16GB is not viable for 9B bf16; T4 fits the 4B (~10GB).
- **35B-A3B**: needs an A100 80GB even at short context. 16K multi-turn trajectories will likely exceed 80GB unless sequences are trimmed, batch is 1, and gradient checkpointing is on. The budget would allow only a few runs (A100 80GB availability on Colab is also not guaranteed). One option is to train on Colab only for small experiments and rent a single H100/A100-80 elsewhere for the final run.
- Since QLoRA is discouraged, the old Fable-V4 recipe (QLoRA on 7B) does not carry over directly to Qwen3.5/3.6 bases.

### Gaps
- No official VRAM figures at 8K/16K sequence length for any of these. The figures above are at default short context.
- Colab compute-unit burn rates per GPU not verified this session (prior knowledge only: A100 >> L4 > T4 per hour).
- Unsloth support status for Qwen3.8-27B, Gemma 4, GLM-4.7-Flash and Devstral Small 2 not verified this session.

## Recommendation: top 3 bases for Threnody

### Takeaway
1) **Qwen/Qwen3.5-9B** as the primary, fully-local, cheaply-trainable base (direct successor to the 7B); 2) **Qwen/Qwen3.6-35B-A3B** as the high-capability "big Threnody" for 12GB + 32GB RAM via expert offload; 3) **Qwen/Qwen3.8-27B** (or Qwen3.6-27B) as a stretch/16-24GB-tier option. GLM-4.7-Flash and Devstral Small 2 are fallbacks for diversity.

### Cited Findings
- Qwen3.5-9B: Apache-2.0, 262K ctx, thinking toggle; BFCL-V4 66.1 and TAU2 79.1, beating Qwen3-30B-A3B-Thinking-2507 (42.4 / 41.9) — [HF card](https://huggingface.co/Qwen/Qwen3.5-9B). Q4 ~6.6GB fits 12GB fully — [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/). bf16 LoRA ~22GB — [Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Qwen3.6-35B-A3B: SWE-V 73.4, TB2.0 51.5, MCPMark 37.0, Apache-2.0, `qwen3_coder` tool parser — [HF card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). ~38.9 tok/s on an RTX 3060 12GB + 32GB RAM — [insiderllm MoE guide](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/). bf16 LoRA ~74GB (Qwen3.5-35B-A3B figure) — [Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Qwen3.8-27B: TB2.1 73.0, SWE-Pro 61.7 (Claude Code harness), Apache-2.0 — [HF card](https://huggingface.co/Qwen/Qwen3.8-27B). 27B dense Q4 ~16-17GB will not fit 12GB — [insiderllm](https://insiderllm.com/guides/best-local-coding-models-2026/). LoRA ~56GB (Qwen3.5-27B figure) — [Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- Alternatives: GLM-4.7-Flash SWE-V 59.2 (30B-A3B) — [HF](https://huggingface.co/zai-org/GLM-4.7-Flash). Devstral Small 2 SWE-V 68.0 vendor / 56.4 open harness, 24B dense, ~14GB Q4 — [letsdatascience](https://letsdatascience.com/news/mistral-releases-devstral-2-for-code-agents-2e24c245). Gemma 4 26B-A4B is weak on SWE-bench (17.4, per Qwen's table) — [HF Qwen3.6 card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

### Inferences
- **#1 Qwen3.5-9B**:
  - Hardware: the only top candidate that meets both hardware targets without offload tricks.
  - Training cost: trains on a single A100 40GB in bf16 LoRA, so many iterations fit in ~200 units.
  - Tool format: same template and tool format as the larger Qwen3.6 models.
  - Weakness: no reported SWE-V/TB scores, so Threnody's fine-tune (agent-trajectory SFT) has the most headroom to show gains, and needs our own SWE-style eval.
- **#2 Qwen3.6-35B-A3B**: best vendor-reported agentic coding of anything that runs at usable speed on a 3060 + 32GB RAM. Training is the bottleneck: ~74GB+ bf16 LoRA, QLoRA discouraged, so use a few A100-80GB runs or rented compute. A sensible plan is to develop the data and recipe on the 9B, then do a final run on the 35B-A3B.
- **#3 Qwen3.8-27B (or Qwen3.6-27B)**: highest quality in range and the newest (Aug 2026). It misses the 12GB target at Q4 (needs 16-24GB VRAM) and needs an A100 80GB for LoRA. Pick it only if the target audience widens to 16-24GB cards.
- Watch for an official Qwen3.8-35B-A3B or 9B. If one appears (the ms-swift commit hints at a 35B-A3B), it would likely replace #2 or #1 directly with the same tooling.

### Gaps
- All rankings rely on vendor-reported scores. Independent leaderboard verification (swebench.com bash-only/mini-swe-agent, tbench.ai) was not obtained.
- No head-to-head data on how much SFT/LoRA on agent trajectories improves Qwen3.5-9B vs 35B-A3B.
