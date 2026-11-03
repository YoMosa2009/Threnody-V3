# Memory Reduction Techniques for Running Threnody Locally (4B-32B, long-context agentic coding), as of Nov 2026

Scope note: these notes come from 15 tool calls (searches and fetches). The hard numbers come mostly from primary sources: llama.cpp READMEs/PRs, the Unsloth docs, Cerebras' HF model card, the Google blog, the REAP and Minitron papers, and the Gemma 3 tech report. Some figures are COMPUTED by the researcher from architecture configs or bpw tables. Those are marked "[computed]". Figures marked "[unverified]" come from blogs, calculators, or single-user reports.

## 1. GGUF quant types: bpw, size, measured quality deltas, imatrix, Unsloth Dynamic 2.0

### Takeaway
On an 8B model (Llama 3 8B, the canonical llama.cpp scoreboard), Q8_0 and Q6_K are effectively lossless (KLD 0.0014 / 0.0055). Q5_K_M is near-lossless (KLD 0.011). Q4_K_M with imatrix is the knee of the curve (KLD 0.028, about 92% same-top-token). IQ4_XS buys 0.44 GiB for slightly more error (KLD 0.036). Quality falls off fast below about 3.5 bpw (IQ3_XXS KLD 0.18; IQ2_XXS 0.81). imatrix helps most at 3-4 bits (Q4_K_M KLD drops about 10%, Q3_K_M about 17%). Unsloth Dynamic 2.0 gives a further 3-8% KLD reduction at the same size. For a 7.6B coder that runs long agentic sessions, ship Q8_0, Q6_K, Q5_K_M, Q4_K_M (imatrix), IQ4_XS (imatrix), and at most one IQ3 tier labelled "degraded".

### Cited Findings
**Bits per weight and size (Llama 3.1 8B, llama.cpp quantize README):** — [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

| Quant | bpw | Size (GiB, 8B) | tg t/s @128 (README's test GPU) |
|---|---|---|---|
| F16 | 16.00 | 14.96 | 29.2 |
| Q8_0 | 8.50 | 7.95 | 50.9 |
| Q6_K | 6.56 | 6.14 | 58.7 |
| Q5_K_M | 5.70 | 5.33 | 67.2 |
| Q4_K_M | 4.89 | 4.58 | 71.9 |
| Q4_K_S | 4.67 | 4.36 | 76.7 |
| IQ4_NL | 4.68 | 4.38 | 76.6 |
| IQ4_XS | 4.46 | 4.17 | 77.5 |
| Q3_K_M | 4.00 | 3.74 | 71.7 |
| IQ3_M | 3.76 | 3.52 | 70.2 |
| IQ3_XXS | 3.25 | 3.04 | 74.0 |
| Q2_K | 3.16 | 2.95 | 79.9 |
| IQ2_M | 2.93 | 2.74 | 74.4 |
| IQ2_XXS | 2.38 | 2.23 | 79.9 |

- The same README says Llama 3.1 8B at Q4_K_M goes from 32.1 GB to 4.9 GB. — [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

**Measured quality vs FP16 (Llama 3 8B scoreboard, Wikitext, "WT 10m" = imatrix from 10M Wikitext tokens):** — [llama.cpp perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)

| Quant | imatrix | GiB | PPL (FP16 = 6.2316) | ΔPPL | Mean KLD | RMS Δp |
|---|---|---|---|---|---|---|
| q8_0 | none | 7.96 | 6.2343 | +0.0027 | 0.00136 | 1.20% |
| q6_K | none | 6.14 | 6.2534 | +0.0217 | 0.00545 | 2.30% |
| q5_K_M | none | 5.33 | 6.2886 | +0.0570 | 0.01076 | 3.16% |
| q4_K_M | WT 10m | 4.58 | 6.3829 | +0.1513 | 0.02815 | 5.25% |
| q4_K_M | none | 4.58 | 6.4071 | +0.1755 | 0.03127 | 5.52% |
| iq4_NL | WT 10m | 4.35 | 6.4556 | +0.2240 | 0.03574 | 6.00% |
| iq4_XS | WT 10m | 4.14 | 6.4597 | +0.2281 | 0.03633 | 6.04% |
| q4_0 | none | 4.34 | 6.7001 | +0.4685 | 0.07194 | 8.43% |
| q3_K_M | WT 10m | 3.74 | 6.7343 | +0.5026 | 0.08436 | 9.29% |
| q3_K_M | none | 3.74 | 6.8885 | +0.6569 | 0.10191 | 10.20% |
| iq3_M | WT 10m | 3.53 | 6.8983 | +0.6667 | 0.10253 | 10.51% |
| iq3_XXS | WT 10m | 3.05 | 7.4584 | +1.2268 | 0.18363 | 13.84% |
| q2_K | WT 10m | 2.96 | 8.6478 | +2.4162 | 0.33222 | 18.88% |
| iq2_M | WT 10m | 2.74 | 8.6008 | +2.3692 | 0.32599 | 18.52% |
| iq2_XXS | WT 10m | 2.24 | 14.0918 | +7.8601 | 0.81202 | 28.58% |

- "Same top token" agreement on Llama 3 8B: q8_0 97.67%, q6_K 96.03%, q4_K_M 91.90%, q2_K 71.14%. — [llama.cpp perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
- Llama 3 8B is much more quantization-sensitive than Llama 2 7B. At q4_K_M, mean KLD is 0.0313 for Llama 3 vs 0.0127 for Llama 2. At q2_K it is 0.445 vs 0.109. — [llama.cpp perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
- Calibration amount: there was "no consistent improvement from using more Wikitext tokens for the importance matrix". q2_K KLD stays at about 0.331-0.337 whether the imatrix used 1k or 10M tokens. — [llama.cpp perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)
- An imatrix is applied with `--imatrix file`. It can be limited to tensors with `--include-weights` / `--exclude-weights`, for example only `attn_v` and `ffn_down`. — [llama.cpp quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- A third-party measurement on Mistral-7B with an imatrix gives median KLD of Q6_K 0.0032, Q5_K_M 0.0043, Q4_K_M 0.0075, IQ4_XS 0.0088, Q3_K_M 0.0171. These are median, not mean, values and come from a different model, so they are not directly comparable to the table above. [unverified secondary aggregation] — [ithome article](https://ithelp.ithome.com.tw/articles/10403312)

**Unsloth Dynamic 2.0 (and later 3.0/3.1):**
- Dynamic 2.0 on Gemma 3 27B gives these KLD values (standard imatrix → UD) at about +0.1-0.25 GB: IQ2_XXS 0.536 → 0.521; Q2_K_XL 0.2297 → 0.2209 (Unsloth says "around 7.5%" better); Q3_K_XL 0.0878 → 0.0806 (-8%); Q4_K_XL 0.0249 → 0.0237 (-5%), 15.41 → 15.64 GB. — [Unsloth Dynamic 2.0 docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- Gemma 3 27B MMLU 5-shot by Unsloth quant: Q8_0 71.60, Q6_K 71.87, Q5_K_M 71.77, Q4_K_M 71.23, Q4_K_XL 71.47, Q3_K_M 70.70, IQ3_XXS 68.27, Q2_K_XL 68.70, IQ2_XXS 59.20, IQ1_S 41.87. BF16 is 71.5. — [Unsloth Dynamic 2.0 docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- Unsloth's calibration set for Dynamic 2.0 is ">1.5M tokens" of "hand-curated and cleaned data", chat-oriented. The later Dynamic 3.0 calibration set was refined for "agentic coding, chat, and multilingual". Unsloth deliberately does not compute KLD on its own calibration data, and it warns that Wikipedia-based imatrix data inflates scores on Wikipedia-based KLD tests. — [Unsloth Dynamic 2.0 docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- Unsloth reports a "Divergence-300 @32" agentic eval: 300 held-out prompts from Terminal-Bench 2.1, DeepSWE, Harbor, MathArena, and long-document sources, decoded greedily for 32 tokens. It also reports a Dynamic 3-bit DeepSeek V3.1 GGUF scoring 75.6% on Aider Polyglot. — [Unsloth Dynamic 2.0 docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)

**Sizes for 30B-class models [computed]:** these use bpw × params / 8. Qwen3-30B-A3B has about 30.5B params. Expected sizes are Q8_0 ≈ 32.4 GB, Q6_K ≈ 25.0 GB, Q5_K_M ≈ 21.7 GB, Q4_K_M ≈ 18.6 GB, IQ4_XS ≈ 17.0 GB, Q3_K_M ≈ 15.2 GB, IQ3_XXS ≈ 12.4 GB, IQ2_M ≈ 11.2 GB. Real files differ a little because embeddings and the output layer are kept at higher precision. For the 7.6B Qwen2.5-Coder base, the 4.36 GiB Q4_K_M from V4 is consistent with about 4.9 bpw. At the same rates: Q8_0 ≈ 7.5 GiB, Q6_K ≈ 5.8 GiB, Q5_K_M ≈ 5.0 GiB, IQ4_XS ≈ 4.0 GiB, IQ3_XXS ≈ 2.9 GiB.

### Inferences
- Quality per GiB levels off above Q5_K_M. Most of the error appears between Q4 and Q3 (KLD roughly triples from Q4_K_M to Q3_K_M). For coding, a single wrong token, such as a bracket, identifier, or tool-call JSON key, can break a turn. A 92% same-top-token rate at Q4_K_M therefore matters more for agents than PPL suggests. Q5_K_M/Q6_K are worth recommending wherever they fit.
- Since the Wikitext imatrix gains are real but modest, an imatrix built from Threnody's own domain (code, tool-call JSON, multi-turn agent traces, long files) should help at least as much on the target distribution. Unsloth's move toward agentic and coding calibration data supports this. A practical recipe is to calibrate on about 0.5-2M tokens drawn from the SFT/eval distribution: chat-template-formatted agent trajectories, diffs, and tool outputs, including some long (16k+) samples, plus some general text. The README result that more tokens do not consistently help means a 1M-token set is enough.
- Measure KLD on a held-out agentic set, not Wikitext. Unsloth's warning about calibration/test overlap applies.
- Qwen2.5/Qwen3-family models are probably at least as quant-sensitive as Llama 3, being heavily trained, but that was not measured here.

### Gaps
- I found no published KLD table for Qwen2.5-Coder-7B or Qwen3-8B/Qwen3-Coder-30B across quant types. Threnody should produce its own with `llama-perplexity --kl-divergence`.
- Dynamic 3.0/3.1 numeric KLD tables are images on the Unsloth page and could not be extracted.
- No controlled study was found of code-specific vs general imatrix effects on pass@1 or SWE-style tasks.

## 2. Quantization-aware training (QAT) for small models

### Takeaway
QAT recovers roughly 35-67% of the int4 PTQ accuracy loss in Google's and PyTorch/Unsloth's measurements. However, a well-made Dynamic/imatrix PTQ quant often matches an official QAT Q4_0 release at a smaller size. QAT is feasible through Unsloth plus TorchAO on a single GPU for 7-8B, since it is just fine-tuning with fake-quant. It is worth trying only if Threnody's Q4 KLD on agentic data turns out to be unacceptable.

### Cited Findings
- Google's Gemma 3 QAT ran about 5,000 steps using the BF16 checkpoint's probabilities as targets. It "reduces the perplexity drop by 54%" (llama.cpp perplexity eval) when quantizing to Q4_0. Gemma 3 27B weights go from 54 GB (BF16) to 14.1 GB (int4). — [Google Developers Blog](https://developers.googleblog.com/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)
- Note that 54% is a reduction in the perplexity *degradation*, not in perplexity itself. — [Google Developers Blog](https://developers.googleblog.com/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)
- In Unsloth's MMLU tests on Gemma 3 27B, the Google QAT Q4_0 file (17.2 GB) scored 70.64 vs Unsloth Q4_K_XL (15.64 GB) at 71.47. Unsloth says its dynamic 4-bit is "2GB smaller whilst having +1% extra accuracy vs the QAT version." Google's file keeps embeddings in fp16. — [Unsloth Dynamic 2.0 docs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- A community re-quant of the 27B QAT gave wikitext PPL 8.2291 vs 8.2323 for the official QAT Q4_0, a near tie at a smaller size with Q6_K embeddings. — [stduhpf QAT small GGUF (mirror)](https://gitcode.com/hf_mirrors/stduhpf/google-gemma-3-27b-it-qat-q4_0-gguf-small/blob/main/README.md)
- TorchAO/Unsloth QAT: INT4 weight-only QAT recovered 66.9% of the accuracy degradation for Gemma3-4B on GPQA (+1.0 abs) and 45.5% for Gemma3-12B on BBH (+2.1 abs). For Qwen3-4B on WikiText it recovered about 34-36% of the PPL increase: bf16 9.148, int4 PTQ 9.775, int4 QAT 9.563. — [PyTorch blog: QAT in TorchAO II](https://pytorch.org/blog/quantization-aware-training-in-torchao-ii/); [Unsloth QAT blog](https://unsloth.ai/docs/blog/quantization-aware-training-qat); [torchao PR #3214](https://github.com/pytorch/ao/pull/3214/files)
- Unsloth exposes `qat_scheme` with options fp8-int4, fp8-fp8, int8-int4, and int4. It requires torchao==0.14.0 and fbgemm-gpu-genai==1.3.0. — [Unsloth QAT blog](https://unsloth.ai/docs/blog/quantization-aware-training-qat)

### Inferences
- The TorchAO int4 scheme (group-wise int4) is not the same as llama.cpp's Q4_K_M or IQ4_XS formats. QAT gains do not automatically carry over to K-quant GGUFs. Google targeted Q4_0 specifically. A QAT pass aimed at GGUF would need a fake-quant that mimics Q4_0, or ideally Q4_K. This is a mismatch risk. [inference]
- Colab feasibility for 7.6B: Fable-Coder-V4 was presumably LoRA fine-tuned on Colab, and QAT adds fake-quant overhead to the same training loop. LoRA plus QAT on an A100/L4-class Colab GPU is plausible for a short (a few thousand step) self-distillation pass. Full-parameter QAT of a 7.6B on a T4 is not realistic. [inference; no measured Colab QAT runtime found]
- A low-cost middle option is a short self-distillation QAT stage after the final SFT, using BF16 Threnody logits as targets, the way Google did it. Then compare KLD and the agentic eval against plain imatrix Q4_K_M before committing.

### Gaps
- No published QAT results for Qwen2.5-Coder or for coding/agentic benchmarks were found.
- No measured QAT wall-clock or VRAM figures on Colab were found.

## 3. KV cache memory: formula, llama.cpp KV quantization, sliding-window / hybrid attention

### Takeaway
For GQA models, KV bytes/token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element. For the 7.6B Qwen2.5 architecture that is about 56 KiB/token: 1.75 GiB at 32k and 7 GiB at 128k in FP16. A q8_0 K+V cache roughly halves this with negligible loss (KLD +0.0008 on Llama 3 8B). q4_0 K+V cuts it to about 28% but costs about as much as a weight-quant step (KLD 0.037), and K is much more sensitive than V. Hybrid/SWA architectures (Gemma 3 5:1 local:global, 1024 window; Qwen3-Next linear attention) cut long-context KV by about 3-6×.

### Cited Findings
- llama.cpp KV-quant measurements (PR #7412, weights f16, Llama-class model):
  - f16/f16: PPL 6.2322, 16 bpv.
  - K f16 + V q8_0: PPL 6.2328, KLD 0.00074, 12.25 bpv.
  - K q8_0 + V q8_0: PPL 6.2344, KLD 0.00098, 8.5 bpv.
  - K f16 + V q4_0: PPL 6.2518, KLD 0.0050, 10.25 bpv.
  - K q5_0 + V q5_0: PPL 6.2789, KLD 0.0099, 5.5 bpv.
  - K q4_0 + V q4_0: PPL 6.4384, KLD 0.0365, 4.5 bpv.
  - For comparison, q4_0 *weights* with an f16 cache give KLD 0.0717.
  - The author's conclusions: "K cache is much more sensitive" than V, weight quantization is the bigger error source, and q8_0 KV shows no significant loss.
  - [llama.cpp PR #7412](https://github.com/ggml-org/llama.cpp/pull/7412)
- Flags are `-ctk`/`--cache-type-k` and `-ctv`/`--cache-type-v` with `-fa on`. On CUDA, mixed or unusual K/V type combinations need a build with `-DGGML_CUDA_FA_ALL_QUANTS=ON`. Otherwise attention can fall back to a slow path. — [llama.cpp issue #24485 (via search)](https://github.com/ggml-org/llama.cpp/issues/24485); [llmhardware.io guide](https://llmhardware.io/guides/kv-cache-quantization-ollama-llama-cpp)
- Ollama's docs describe q8_0 KV as usually having no noticeable quality impact and q4_0 as a "small-medium loss" that is more noticeable at long context. Ollama requires Flash Attention for quantized KV (OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE). — [llmhardware.io guide summarizing Ollama docs](https://llmhardware.io/guides/kv-cache-quantization-ollama-llama-cpp)
- Speed: one Metal benchmark on a 3B model found prompt processing and generation about one third slower with FA and a quantized K or V. Blog claims of parity within 5% on CUDA are unsourced. [conflicting/unverified] — [llama.cpp issue #8918](https://github.com/ggml-org/llama.cpp/issues/8918); [bmdpat blog](https://bmdpat.com/blog/kv-cache-q4-vs-q8-in-llama-cpp-explained-2026)
- Gemma 3 interleaves 5 local (sliding window 1024) layers per global layer. At 32k, "global only" carries about 60% KV memory overhead, reduced to under 15% with 1:3 and a 1024 window. Perplexity impact is "minimal, even with 7-to-1". — [Gemma 3 Technical Report](https://arxiv.org/pdf/2503.19786)
- Qwen3-Coder-30B-A3B (and REAP variant) attention config: 48 layers, 32 Q heads, 4 KV heads, 262,144 native context. — [Cerebras REAP model card](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B)

**KV size calculations [computed; architecture params from model configs. Qwen2.5-7B: 28 layers, 4 KV heads, head_dim 128 (HF config.json); Qwen3-Coder-30B-A3B: 48 layers, 4 KV heads, head_dim 128 per the REAP card; Qwen3-8B: 36 layers, 8 KV heads, head_dim 128 (HF config, not re-fetched here)]:**

| Model | KV/token f16 | 32k f16 | 64k f16 | 128k f16 | 128k q8_0 (×0.53) | 128k q4_0 (×0.28) |
|---|---|---|---|---|---|---|
| Qwen2.5-7B / Fable-Coder base | 56 KiB | 1.75 GiB | 3.5 GiB | 7.0 GiB | ~3.7 GiB | ~2.0 GiB |
| Qwen3-8B (dense) | 144 KiB | 4.5 GiB | 9 GiB | 18 GiB | ~9.6 GiB | ~5.1 GiB |
| Qwen3(-Coder)-30B-A3B | 96 KiB | 3.0 GiB | 6 GiB | 12 GiB | ~6.4 GiB | ~3.4 GiB |

(Ratios use bpv: q8_0 = 8.5/16, q4_0 = 4.5/16. Compute buffers of a few hundred MB to about 1 GB come on top.)

### Inferences
- The base architecture choice matters a lot for long context. Qwen2.5-7B's 4 KV heads make it unusually cheap per token (56 KiB). Qwen3-8B's 8 KV heads and 36 layers cost 2.6× more. If Threnody moves to a Qwen3-8B-class base, 128k context is only practical with q8_0 or q4_0 KV, even on 16 GB GPUs.
- Recommended default: `-fa on -ctk q8_0 -ctv q8_0` everywhere. It is nearly free in quality and halves the KV cache. For 8 GB cards at 64k+, use `-ctk q8_0 -ctv q4_0` (V is less sensitive) before going to q4_0 K. Avoid q4_0 K for code agents.
- A hybrid-attention base (Gemma 3/4-style SWA, Qwen3-Next/Qwen3.5-style Gated DeltaNet + attention) would cut 128k KV by roughly 3-6× at the architecture level. That is a base-model choice, not something that can be retrofitted to a Qwen2.5 fine-tune.

### Gaps
- No rigorous long-context (64k-128k) quality study of KV quantization on code or agent tasks was found. The PR #7412 numbers are short-context Wikitext. Long-context error accumulation from q4 K is plausible but not quantified.
- Qwen3-Next memory figures (3:1 linear:full attention ratio) were not fetched or verified in this session.

## 4. MoE expert offloading in llama.cpp (--n-cpu-moe, -ot)

### Takeaway
With `--n-cpu-moe N` (`-ncmoe`), or `-ot ".ffn_.*_exps.=CPU"`, the expert FFN weights stay in system RAM. Attention, shared tensors, and the KV cache go on the GPU. This makes 20-35B A3B-class MoEs usable on 8-12 GB GPUs at roughly 25-40 tok/s, but the evidence is community reports, not controlled benchmarks. Decode speed is bound by system RAM bandwidth.

### Cited Findings
- A single user reported Qwen3.6-35B-A3B at IQ3_XXS with 64k context on an RTX 3070 Ti 8 GB going from 8.7 tok/s to "40+" tok/s after adding `-ncmoe`. The suggested start for 8 GB is `-ncmoe 25`, targeting about 6.9 GB VRAM. [unverified single report] — [Medium: "One flag tripled my tokens per second on an 8GB GPU"](https://medium.com/coding-nexus/one-flag-tripled-my-tokens-per-second-on-an-8gb-gpu-238f55d1e3bd)
- gpt-oss-20b on an RTX 3060 12 GB reached about 30-31 tok/s in the llama.cpp community benchmark thread (Discussion #15396), described as CPU-offload bound. Calculator estimates are 29-38 tok/s. The suggested start is `--n-cpu-moe 6` at 16k. One report saw 12.3 GiB at 24k context, causing a spill. Enabling XMP/DOCP on DDR4 (2400 → 3200) added 10-17% throughput. [secondary/partly calculator-based] — [runaihome guide](https://runaihome.com/blog/gpt-oss-20b-local-ai-hardware-guide-2026/); [aliteq web story](https://aliteq.com/web-stories/gpt-oss-20b-8gb-12gb-gpu-moe-offload-2026); [willitrunai](https://willitrunai.com/de/can-run/gpt-oss-20b-on-rtx-3060-12gb)
- CPU-only floor: Qwen3-VL-30B-A3B Q4_K_M on an i7-8700 with no GPU ran at about 10.6 tok/s, reading about 3.3B weights per token. — [InventiveHQ MoE-on-CPU benchmark](https://inventivehq.com/blog/moe-on-cpu-benchmark)

### Inferences
- For Threnody, a ~30B-A3B MoE coder base would give much higher quality than a 7.6B dense model and still run on 8-12 GB GPUs with 32 GB system RAM. It would need about 18 GB at Q4_K_M, or about 15 GB REAP-25B at Q4_K_M [computed]. The cost is a 32 GB RAM requirement and prompt-processing speed that is limited for 100k-token agent contexts.
- Tune by starting with all experts on CPU (`-ot exps=CPU` or a high `--n-cpu-moe`) and lowering N until VRAM is about 0.5-1 GB below full, leaving room for the KV cache at the target context.

### Gaps
- No controlled tok/s and VRAM table for Qwen3-30B-A3B or Qwen3-Coder-30B-A3B with `--n-cpu-moe` on 8/12 GB was found. Discussion #15396 was not opened directly.
- Prompt-processing (prefill) speed under expert offload at 64k+ is not quantified. This matters for agentic re-reads of long contexts.

## 5. Structural compression: REAP expert pruning, layer pruning, Minitron

### Takeaway
REAP is the most practical option here. Cerebras' 20%-pruned Qwen3-Coder-REAP-25B-A3B (128 → 103 experts/layer) matches the 30B on coding and agentic benchmarks within about ±2 points while saving 1/6 of memory, and the calibration data was code, function calling, and SWE trajectories. Minitron-style width pruning plus distillation works (8B → 4B with 94B tokens) but needs a compute budget far beyond Colab. Plain layer pruning (ShortGPT) is cheap but lossy and was not verified for code here.

### Cited Findings
- REAP scores experts by router gate values × average activation norm and prunes one-shot. The paper reports "near-lossless" code generation (Δacc ≤ 2%) at 50% expert pruning for Qwen3-Coder-480B and Kimi-K2. It is an ICLR 2026 paper. — [REAP paper arXiv 2510.13999](https://arxiv.org/pdf/2510.13999); [Cerebras blog](https://www.cerebras.ai/blog/reap); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10006871)
- The REAP repo tested Qwen3-30B-A3B, ERNIE-4.5-21B-A3B, Mixtral-8x7B, GLM-4.5-Air, and Llama-4-Scout. It reports a mean accuracy drop of 1.9% (2.6% without logit normalization). — [CerebrasResearch/reap GitHub](https://github.com/CerebrasResearch/reap)
- Qwen3-Coder-30B-A3B vs REAP-25B-A3B (20% pruned), original → pruned:
  - HumanEval 92.1 → 94.5; HumanEval+ 87.8 → 89.0
  - MBPP 87.6 → 87.3; MBPP+ 73.5 → 72.8
  - LiveCodeBench 35.2 → 35.2
  - BFCL-v3 Overall 63.2 → 62.2 (Multi-Turn 29.6 → 30.5)
  - τ²-bench Airline 39.3 → 40.7, Retail 62.6 → 62.0, Telecom 33.6 → 32.2
  - Calibration data: evol-codealpaca, xlam-function-calling-60k, SWE-smith-trajectories.
  - [HF: cerebras/Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B)
- Minitron: Llama-3.1-Minitron 4B (width-pruned from 8B, then distilled) used 94B tokens, vs 7T for Qwen2 1.5B. Width variant scores were IFEval 52.4, MT-Bench 6.3, BFCL 64.9. TensorRT-LLM speedups vs the 8B were 1.8× (width) and 2.7× (depth). Width pruning beat depth pruning on accuracy. — [Minitron paper arXiv 2408.11796](https://arxiv.org/html/2408.11796v3); [NVIDIA blog](https://developer.nvidia.com/blog/?p=87164)

### Inferences
- REAP applies only to MoE bases. If Threnody stays on a 7.6B dense Qwen2.5-Coder, REAP is irrelevant. If Threnody moves to Qwen3-Coder-30B-A3B (or a successor A3B coder), then fine-tuning on it, followed by a REAP prune calibrated on Threnody's own agent trajectories, is a realistic way to ship a ~25B-A3B GGUF that fits 16 GB RAM + 8 GB VRAM at Q4. Cerebras' public code makes this feasible on a single large GPU. Exact hardware needs are unverified.
- Minitron-style pruning of a 7.6B dense model to about 4B would need tens of billions of distillation tokens. That is out of reach for a Colab-scale project. Quantization gives better quality per GB for this budget.
- Layer pruning (ShortGPT etc.) is not recommended without a recovery fine-tune. Multi-step reasoning and code are known to be fragile. [inference; ShortGPT sources not fetched]

### Gaps
- SWE-bench / Terminal-Bench numbers for REAP-pruned coders were not on the model card.
- Results for 50%-pruned 30B coders (e.g., a REAP-15B) were not verified.
- No ShortGPT or layer-pruning code-benchmark numbers were collected in this session.

## 6. Other: speculative decoding, mmap, runtimes, AWQ/GPTQ/EXL3

### Takeaway
Limited verified data was collected on these topics in this session. The points below are mostly about how the tools work, and they are flagged accordingly.

### Cited Findings
- llama.cpp mmaps GGUF weights by default. Quantized tensors offloaded to GPU are copied to VRAM, and the KV cache is allocated up front for the full `-c`. Context length set at load time therefore drives memory, not tokens actually used. [general llama.cpp behavior; no specific source fetched]
- In Ollama, KV quantization is configured globally via environment variables and requires Flash Attention. — [llmhardware.io guide](https://llmhardware.io/guides/kv-cache-quantization-ollama-llama-cpp)

### Inferences
- Speculative decoding (`-md draft.gguf`, e.g., a 0.5B Qwen2.5-Coder draft for a 7B target) speeds up decode on code, which has high acceptance rates. It costs extra VRAM for the draft weights and the draft KV, so it is not a memory-reduction technique. Offer it only for 12 GB+ setups. [inference; no measured numbers collected]
- MLX on Apple Silicon uses unified memory. A 4-bit MLX quant of a 7.6B model is about 4.3 GB. Shipping an MLX 4-bit and 8-bit build alongside the GGUFs is cheap. [inference]
- AWQ/GPTQ (vLLM) and EXL2/EXL3 (ExLlama) target GPU-only inference. They make sense only for users with enough VRAM. GGUF covers more of Threnody's audience (CPU, partial offload, Ollama, LM Studio). [inference]

### Gaps
- No measured comparison of Ollama vs llama.cpp vs LM Studio vs MLX memory use was found.
- No speculative-decoding acceptance-rate or tok/s numbers for Qwen2.5-Coder draft pairs were collected.
- EXL3 quality-vs-bpw data was not collected.

## 7. Concrete recommendation: release artifacts and run settings

### Takeaway
For a 7.6B Qwen2.5-Coder-based Threnody, ship these GGUFs, all built with a code and agentic imatrix:

| Artifact | Size [computed] | Notes |
|---|---|---|
| Q8_0 | ~7.5 GiB | |
| Q6_K | ~5.8 GiB | |
| Q5_K_M | ~5.0 GiB | |
| Q4_K_M | ~4.4 GiB | the default; same as V4 |
| IQ4_XS | ~4.0 GiB | |
| IQ3_M or Q3_K_M | ~3.4 GiB | marked "low-RAM, degraded" |

Default runtime: `-fa on -ctk q8_0 -ctv q8_0`. Optionally add an MLX 4-bit/8-bit build. Do not ship IQ2 or IQ1 tiers for a 7.6B model: Llama 3 8B KLD is ≥0.33 at those levels.

### Cited Findings
- Quality ordering and sizes come from the llama.cpp tables in section 1. — [llama.cpp perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md); [quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- KV quant costs come from section 3. — [llama.cpp PR #7412](https://github.com/ggml-org/llama.cpp/pull/7412)

### Inferences
Suggested run profiles for a 7.6B Threnody (56 KiB/token KV) [computed memory budgets; +~0.5-1 GB compute buffers]:

| Hardware | Weights | Context | KV settings | Approx. total |
|---|---|---|---|---|
| 8 GB GPU | Q4_K_M (4.4 GiB) | 64k | q8_0/q8_0 (~1.9 GiB) | ~7 GiB, all on GPU |
| 8 GB GPU (128k) | IQ4_XS (4.0 GiB) | 128k | q8_0 K / q4_0 V (~2.8 GiB) | ~7.5 GiB, tight; or keep Q4_K_M and offload a few layers with `-ngl` |
| 12 GB (RTX 3060) | Q6_K (5.8 GiB) | 128k | q8_0/q8_0 (~3.7 GiB) | ~10.5 GiB; or Q8_0 at 64k |
| 16 GB+ | Q8_0 (7.5 GiB) | 128k | f16 KV (7 GiB) | ~15.5 GiB; or q8_0 KV for headroom |
| CPU-only 16 GB RAM | Q4_K_M | 32k-64k | q8_0 | decode bound by RAM bandwidth (rough estimate: ~8-15 tok/s on DDR4/DDR5 dual channel [unverified estimate]) |
| CPU-only 32 GB | Q5_K_M/Q6_K | up to 128k | q8_0 | prefill of long agent contexts on CPU will be slow (minutes at 100k) [inference] |
| Apple Silicon 16 GB | MLX 4-bit or GGUF Q4_K_M/Q5_K_M via Metal | 64k | q8_0 KV | Metal KV-quant speed penalty possible (issue #8918) |

- Calibrate the imatrix on about 1M tokens of chat-templated Threnody agent trajectories, code, diffs, and tool JSON, including long samples. Publish a KLD and same-top-token table per quant, measured on a held-out agentic set. This also creates a clear quality claim for the release.
- If a later Threnody moves to a 30B-A3B MoE coder base, add a REAP-pruned (~20%) variant. Document the `--n-cpu-moe` profiles: 8 GB → start around 25-30 offloaded layers; 12 GB → around 6-16; 32 GB system RAM required.
- QAT only if the agentic eval shows Q4_K_M regressions that matter. Otherwise imatrix/Dynamic-style PTQ gives about the same result for much less effort.

### Gaps
- Every Threnody-specific number above is an extrapolation from Llama 3 8B / Gemma 3 measurements and architecture arithmetic. Actual KLD and agentic pass rates per quant must be measured on Threnody itself.
- CPU-only tok/s for a 7.6B model was not directly measured in any source found.
