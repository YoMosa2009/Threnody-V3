# Teacher model for distilling agentic coding into Qwen/Qwen3.5-9B (single Colab A100 80GB, as of 2026-11-02)

Method note: the HF facts below (dates, licenses, parameter counts, tokenizer and template comparisons) were pulled straight from the HF API and `resolve/main` files and compared locally with md5 hashes and Python diffs. The local copies are in the session scratchpad. "Inferences" are my own arithmetic and judgment, not sourced claims.

## Q1. Which candidates released Jun–Nov 2026 fit on one A100 80GB, and how do they compare?

### Takeaway
**Qwen/Qwen3.8-27B is the only strong agentic-coding candidate released in the window that fits on one A100 80GB.** It is dense, Apache-2.0, has the same architecture class and BPE vocab as Qwen3.5-9B, and has an official FP8 checkpoint. The other in-window releases either do not fit or are not coding models: Qwen3.8-Flash-Next is about 180B, GLM-5.3-Flash about 321B, and Qwen-AgentWorld-35B-A3B is an environment world model. Gemma 4 and Devstral Small 2 were released before June 2026 and use incompatible tokenizers.

### Cited Findings
- **Qwen/Qwen3.8-27B**: repo created 2026-08-05, last modified 2026-08-14, license apache-2.0, 27,781,427,952 params, all BF16. That is about 55.6 GB of weights in bf16. — [HF API](https://huggingface.co/api/models/Qwen/Qwen3.8-27B)
- **Qwen/Qwen3.8-27B-FP8** (official): created 2026-08-13, apache-2.0. It holds 24.70B F8_E4M3 params and 3.08B BF16 params (embeddings, lm_head, vision and other unquantized modules), so about 28 GB on disk. The quantization config is `{"quant_method":"fp8","fmt":"e4m3","activation_scheme":"dynamic","weight_block_size":[128,128]}`. — [HF API](https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8), [config.json](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main/config.json)
- **Qwen3.8-27B benchmarks**, as published by Qwen on the model card (comparison values are Qwen3.6-27B / Qwen3.7-Plus / Opus 4.6 Max):
  - Terminal-Bench 2.1 (Terminus): 73.0 (63.4 / 64.0 / 78.2)
  - SWE-bench Pro: 61.7 (53.5 / 57.6 / 53.4)
  - NL2Repo-Bench: 42.3 (36.2 / 41.1 / 47.6)
  - DeepSWE 1.1: 42.2 (13.3 / 14.2 / –)
  - QwenSWEBench: 79.0 (49.3 / 59.2 / 63.8)
  - LiveCodeBench v6: 90.3 (83.9 / 89.6 / 88.8)
  - Other scores: IFBench 79.5, GPQA-D 89.2, HLE 30.8
  - The card also lists a "Muse Glimmer-30B" at 51.7 TB2.1 and 51.2 SWE-Pro.
  - The card gives no SWE-bench Verified, BFCL, or tau2 numbers.
  - Source: [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- Qwen3.8-27B card details:
  - Context is 262,144 native, extendable to about 1M with YaRN.
  - Recommended thinking-mode sampling: temperature=1.0, top_p=0.95, top_k=20, min_p=0, presence_penalty=0.
  - `reasoning_effort` accepts xhigh (default), medium, or low. The card notes that lower effort can increase total latency in multi-turn agent tasks.
  - `preserve_thinking` defaults to on and keeps reasoning from all historical turns.
  - Source: [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Qwen/Qwen3.6-27B** (created 2026-04-21, apache-2.0, official -FP8 exists) and **Qwen/Qwen3.6-35B-A3B** (2026-04-15, apache-2.0, -FP8 at 35.95B params) were both released *before* June 2026. — [HF API 3.6-27B](https://huggingface.co/api/models/Qwen/Qwen3.6-27B), [HF API 3.6-35B-A3B-FP8](https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B-FP8)
- On SWE-bench Pro and Terminal-Bench, Qwen3.6-27B scores 8 to 10 points below Qwen3.8-27B (53.5 vs 61.7, 63.4 vs 73.0). — [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Qwen/Qwen3.8-Flash-Next** (2026-08-24, license "other", arch `Qwen4ExpForConditionalGeneration`, 512 experts top-10): about 180.0B total params, so it does not fit on 80 GB even in FP8. Its BPE vocab is identical to Qwen3.5-9B and vocab_size is 248320. — [HF API](https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next), [config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/config.json)
- **Qwen/Qwen3.8-2.4T-A95B** (2026-08-08, plus FP8 and NVFP4/MXFP4 variants) is a 2.4T-parameter model that needs multi-GPU nodes. — [HF listing](https://huggingface.co/qwen/qwen3.8-2.4t-a95b), [vLLM blog](https://vllm.ai/blog/2026-08-12-qwen3.8)
- **Qwen/Qwen-AgentWorld-35B-A3B** (2026-06-22, apache-2.0, 34.7B params, Qwen3.5-MoE arch, same tokenizer): this is a *language world model for simulating agent environments* (MCP, Search, Terminal, SWE, and other domains). It is not a coding agent. It reports only AgentWorldBench (overall 56.39, SWE domain 65.63). — [model card](https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B)
- **zai-org/GLM-5.3-Flash** (2026-08-25, MIT): about 321.3B params with 288 experts and vocab_size 154880, so it does not fit and its tokenizer is incompatible. — [HF API](https://huggingface.co/api/models/zai-org/GLM-5.3-Flash), [config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json)
- **mistralai/Devstral-Small-2-24B-Instruct-2512** (created 2025-11-28, apache-2.0, 24.0B params, vocab 131072) is outside the date window and has an incompatible tokenizer. — [HF API](https://huggingface.co/api/models/mistralai/Devstral-Small-2-24B-Instruct-2512)
- **google/gemma-4-31B-it** and **gemma-4-26B-A4B-it** (both created 2026-03-11, apache-2.0, 31.3B and 25.8B params) are outside the date window. Only QAT variants, a 12B, and diffusiongemma appeared in May–June 2026. — [HF API gemma-4-31B-it](https://huggingface.co/api/models/google/gemma-4-31B-it), [HF google listing](https://huggingface.co/api/models?author=google&sort=createdAt&direction=-1)
- **NVIDIA Nemotron-3.5-Lightning-30B-A3B** exists on HF as BF16, NVFP4, NVFP4-DFlash, NVFP4-DSpark, and Base-BF16 repos. — [HF search](https://huggingface.co/api/models?search=Nemotron-3.5&author=nvidia)
- **Community quant**: palmfuture/Qwen3.8-27B-GPTQ-Int4 (third party), reported as tested on vLLM 0.25.1 with MTP. — [HF](https://huggingface.co/palmfuture/Qwen3.8-27B-GPTQ-Int4)

### Inferences
- **License**: Apache-2.0 places no restriction on using model outputs to train other models, so Qwen3.8-27B is clean for distillation. Qwen3.8-Flash-Next's "other" license would need review, but that model cannot run on one A100 anyway.
- **Ranking for this use case**:
  1. Qwen3.8-27B-FP8: best agentic scores, fits, same vocab.
  2. Qwen3.6-27B: same family and tokenizer, weaker, and pre-window.
  3. Qwen3.6-35B-A3B: fast MoE, same tokenizer, but weaker per the Qwen3.6 generation and pre-window.
  4. Everything else is ruled out by size or tokenizer.
- **bf16 vs FP8 for the 27B**:
  - bf16 at about 55.6 GB fits, but leaves only about 15–18 GB for KV and activations at gpu-memory-utilization 0.92.
  - FP8 at about 28 GB (less with `--language-model-only`, which skips the vision tower) leaves about 40+ GB.

### Gaps
- I did not retrieve Qwen3.8-27B SWE-bench Verified, BFCL v4, or tau2 scores. The model card does not list them; the Qwen blog at https://qwen.ai/blog?id=qwen3.8 was not fetched.
- I did not extract the Qwen3.6-27B SWE-bench Verified value; the README table is HTML and only the row names were captured.
- I did not fetch Nemotron-3.5-Lightning-30B-A3B's release date, license, benchmarks, or tokenizer. As a Nemotron model it almost certainly has a non-Qwen vocab, so it cannot be used for logit distillation; this is unverified.
- "Muse Glimmer-30B" appears as a comparison on the Qwen card, but I did not locate its repo.

## Q2. Does Qwen3.8-27B use the same tokenizer and chat template as Qwen3.5-9B?

### Takeaway
**The tokenizer is compatible for text tokens, but the chat template is not identical.**
- **Tokenizer**: the BPE vocab and merges are byte-identical, vocab_size is 248320 in both, and the architecture is the same (`Qwen3_5ForConditionalGeneration` / `qwen3_5_text`). Qwen3.8 only adds 7 audio/TTS special tokens at ids 248070–248076, which fall inside the padded region of Qwen3.5's 248320-row embedding. Logit and logprob distillation token-for-token is therefore feasible.
- **Chat template**: Qwen3.8's template differs in reasoning-effort injection and preserve_thinking defaults, so prompts must be rendered with the *student's* template and passed to the teacher as token IDs.

### Cited Findings
- md5 hashes compared across Qwen3.5-9B, Qwen3.6-27B, Qwen3.8-27B, and Qwen3.8-27B-FP8:
  - `vocab.json` (4e976260d822) and `merges.txt` (bc43d7e04aa9) are identical in all four.
  - `tokenizer.json` is identical between 3.5-9B and 3.6-27B but differs for 3.8. The only differing key is `added_tokens`: 26 entries in 3.5 vs 33 in 3.8. The new entries are id 248070 `<|audio_start|>`, 248071 `<|audio_end|>`, 248072 `<tts_pad>`, 248073 `<tts_text_bos>`, 248074 `<tts_text_eod>`, 248075 `<tts_text_bos_single>`, and 248076 `<|audio_pad|>`.
  - The `model.vocab` objects are equal (True).
  - Sources: [3.5-9B tokenizer.json](https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/tokenizer.json), [3.8-27B tokenizer.json](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/tokenizer.json)
- config.json comparison, Qwen3.5-9B vs Qwen3.8-27B:
  - Both use architecture `Qwen3_5ForConditionalGeneration` and text model_type `qwen3_5_text`, with vocab_size 248320, max_position_embeddings 262144, full_attention_interval 4, head_dim 256, 4 KV heads, and 1 MTP layer.
  - **Qwen3.5-9B**: 32 layers (8 full attention), hidden 4096, 16 attention heads, 32 linear value heads.
  - **Qwen3.8-27B**: 64 layers (16 full attention), hidden 5120, 24 attention heads, 48 linear value heads.
  - Qwen3.6-27B has the same shape as Qwen3.8-27B.
  - Sources: [3.5-9B config](https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/config.json), [3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/config.json)
- Chat template diff (`chat_template.jinja`, 3.5-9B 7756 B vs 3.8-27B 8952 B):
  - (a) 3.8 adds a `reasoning_effort` variable (xhigh default, or medium/low) and **injects a reasoning-instruction sentence into the system message by default** whenever thinking is on.
  - (b) 3.8 adds a `preserve_thinking` flag, default true, which keeps `<think>` content on *all* historical assistant turns. 3.5 keeps it only after the last user query.
  - (c) 3.5 parses `<think>` out of the content string when `reasoning_content` is absent; 3.8 removed this.
  - (d) 3.8 adds a tool-call arguments edge case (empty string) and changes string/JSON argument serialization.
  - The `<tool_call><function=...>` XML format is the same in both.
  - Sources: [3.5-9B template](https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/chat_template.jinja), [3.8-27B template](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/chat_template.jinja)

### Inferences
- **On-policy or logit distillation**: tokenize the student-rendered sequence with the Qwen3.5 template, then send those *exact token IDs* to the teacher, either through the vLLM completions endpoint with a token-ID prompt or offline `LLM.generate(prompt_token_ids=...)`. Do not let the teacher re-render with its own template, or the sequences will misalign.
  - Ids 248070–248076 never occur in student text, so this is safe.
  - Optionally mask teacher logits for ids ≥ 248070 when doing full-distribution KL.
- **Trajectory generation (SFT data)**:
  - Pass `chat_template_kwargs={"reasoning_effort": "medium"}`, or keep xhigh for quality, and decide on `preserve_thinking`.
  - At training time, re-render the stored messages with the **Qwen3.5-9B** template, which drops historical thinking before the last user turn. The injected reasoning-effort system sentence then does not leak into student data.
  - Because 3.5's template strips earlier-turn reasoning, per-turn samples are the faithful format for multi-turn agentic SFT.

### Gaps
- I did not test-encode sample strings with both tokenizers. The identical vocab and merges plus added-token diff strongly imply identical encodings for text, but pre-tokenizer and normalizer equality was only checked implicitly: the diff showed `added_tokens` as the only differing top-level key.

## Q3. Serving on Colab A100 in Nov 2026: versions, install, flags, FP8, throughput, context

### Takeaway
Use vLLM 0.32.0 (released 2026-10-18; 0.33.0 came out today, 2026-11-02, so treat it as too fresh) in a separate uv venv, because vLLM pins its own torch (0.32.0 → torch==2.13.0) and would clobber Colab's torch 2.11. Serve `Qwen/Qwen3.8-27B-FP8` with these flags:
- `--language-model-only`
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`
- optional MTP speculative decoding

On A100 (SM80), vLLM runs this FP8 checkpoint through the Marlin weight-only FP8 kernel (W8A16): memory halves, but there is no FP8 compute speedup.

### Cited Findings
- **vLLM versions on PyPI**:
  - Recent releases: 0.31.0 (2026-10-05), 0.32.0 (2026-10-18), 0.33.0 (2026-11-02).
  - 0.32.0 requires Python >=3.11,<3.15, `torch==2.13.0`, `transformers>=5.16.1,<5.20`, `flashinfer-python==0.7.0.post1`, and ships an abi3 manylinux_2_28 x86_64 wheel.
  - 0.33.0 pins `torch==2.14.0`.
  - Sources: [PyPI vllm](https://pypi.org/pypi/vllm/json), [PyPI vllm 0.32.0](https://pypi.org/pypi/vllm/0.32.0/json)
- **SGLang** latest is 0.5.23 (2026-10-30), which pins `torch==2.14.1`, `transformers==5.19.0`, and `flashinfer_python[cu13]==0.7.2`. — [PyPI sglang](https://pypi.org/pypi/sglang/json)
- **vLLM install docs**:
  - The docs recommend a fresh environment: `uv pip install vllm --torch-backend=auto`.
  - Default binaries are built with CUDA 12.9; CUDA 12.8 and 13.0 variants are offered via release-asset wheels, e.g. `vllm-${VER}+cu130-...whl` with `--extra-index-url https://download.pytorch.org/whl/cu130`.
  - `--torch-backend=auto` picks the PyTorch index from the installed driver.
  - Source: [vLLM GPU install docs (v0.32.0)](https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/getting_started/installation/gpu.cuda.inc.md)
- **vLLM day-0 Qwen3.8 support** was announced on 2026-08-12. The announcement notes Qwen3.8 is "built on the Qwen 3.5 architecture". Its serve flags include `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'`, and it advises a large reasoning token budget. — [vLLM blog](https://vllm.ai/blog/2026-08-12-qwen3.8)
- **vLLM Qwen3.5/3.6 recipe**:
  - Tool calling uses `--enable-auto-tool-choice --tool-call-parser qwen3_coder`; reasoning uses `--reasoning-parser qwen3`.
  - `--language-model-only` skips the vision encoder to free KV memory.
  - MTP is set with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
  - The recipe recommends official FP8 checkpoints.
  - Prefix caching for the Mamba-cache "align" mode is experimental.
  - If you hit a `num_cache_lines >= batch` assertion, reduce `--max-cudagraph-capture-size` (default 512).
  - The recipe was verified only on 8×H200 / MI300X, with no A100 notes and no throughput numbers.
  - Source: [vLLM recipes – Qwen3.5](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- **Qwen3.8-27B card serving instructions**: plain `vllm serve "Qwen/Qwen3.8-27B"` or the SGLang `launch_server`, using "the latest framework versions", with no minimum version given. — [model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **FP8 on Ampere in vLLM 0.32.0**:
  - The `Fp8Config.get_min_capability()` source returns 75, and a code comment says "For GPUs that lack FP8 hardware support, we can leverage the Marlin kernel for fast weight-only FP8 quantization".
  - The docs' hardware table marks Marlin (GPTQ/AWQ/FP8/FP4) as supported on Ampere, while llm-compressor FP8 W8A8 is not supported on Ampere.
  - Sources: [fp8.py v0.32.0](https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/model_executor/layers/quantization/fp8.py), [quantization README v0.32.0](https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/features/quantization/README.md)

### Inferences
These are my own arithmetic from config.json, not measured.

**Full-attention KV cache for Qwen3.8-27B** = 16 full layers × 2 (K,V) × 4 KV heads × 256 head_dim × 2 B (bf16) = **64 KiB per token**:
- 16K context ≈ 1 GiB per sequence
- 64K context ≈ 4 GiB per sequence
- 262K context ≈ 16 GiB per sequence

**Gated-DeltaNet recurrent state** per sequence (fixed, independent of length) ≈ 48 linear layers × 48 value heads × 128 × 128 = 37.7M elements:
- about 75 MB per sequence in bf16, or about 150 MB in fp32
- plus a small conv state

**Memory budget** at FP8 with `--gpu-memory-utilization 0.92` (about 73.6 GB):
- weights about 26–28 GB, plus activations and cudagraphs of a few GB
- that leaves about 40 GB for cache, which is roughly 32 concurrent sequences at 32K context (32 GB of KV + about 2.4–4.8 GB of state), or about 9 sequences at 128K
- **For small-repo agent tasks, `--max-model-len 65536` with `--max-num-seqs 32` is a reasonable start.**
- In bf16, only about 15 GB is left, so fewer than 4 concurrent sequences at 64K.

**Throughput** (rough estimate, no source found):
- The A100 has about 2.0 TB/s HBM. A W8A16 decode step reads about 27 GB of weights, so a single stream tops out around 60–70 tok/s.
- Batched decoding at 32–64 concurrency should give roughly **500–1,500 output tok/s aggregate**, more with MTP acceptance.
- Benchmark it with `vllm bench serve` on Colab before planning data volume.

**Colab**: Colab ships torch 2.11, while vLLM 0.32.0 needs torch 2.13. Installing into the system Python replaces torch and can break the training stack. Use a separate venv for the server, and run the student training stack in a separate venv or process.

**Launch commands** (the recommended path; not executed by me):
```bash
# Cell 1: isolated server env
pip install -q uv
uv venv /content/vllm-env --python 3.12
VIRTUAL_ENV=/content/vllm-env uv pip install "vllm==0.32.0" --torch-backend=auto
# Cell 2: launch (background); HF_HOME on local disk, ~28 GB download
nohup /content/vllm-env/bin/vllm serve Qwen/Qwen3.8-27B-FP8 \
  --served-model-name teacher \
  --language-model-only \
  --max-model-len 65536 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --max-logprobs 20 \
  --port 8000 > /content/vllm.log 2>&1 &
```
Client settings:
- Use the OpenAI SDK at `http://localhost:8000/v1` with temperature=1.0, top_p=0.95, and `extra_body={"top_k":20, "chat_template_kwargs":{"reasoning_effort":"medium"}}`. Medium or low cuts token cost; xhigh is the default.
- If MTP or prefix caching triggers errors on A100, drop `--speculative-config` first. Add `--enable-prefix-caching` only after a smoke test, since it is experimental for the hybrid cache.

### Gaps
- No primary source gives A100 throughput for Qwen3.8-27B, and none confirms whether vLLM 0.32.0's `--torch-backend=auto` picks cu129 or cu130 on Colab's driver.
- I did not verify SGLang's Marlin FP8 path on SM80. SGLang 0.5.23 pins a cu13 flashinfer, so A100 compatibility should be smoke-tested.
- `--max-logprobs` and `prompt_logprobs` flag names are from my prior knowledge of vLLM and were not re-verified in the 0.32.0 docs in this session.

## Q4. Logit-level / on-policy distillation: memory on one 80 GB GPU, or precompute teacher logprobs?

### Takeaway
Run the passes **sequentially, not co-resident**:
1. The student generates rollouts (vLLM with the 9B).
2. The teacher scores them in a separate vLLM pass (FP8 27B) using prompt logprobs on the exact student token IDs.
3. The trainer (9B bf16 + LoRA) consumes the stored per-token teacher logprobs.

Co-residence is technically possible on 80 GB, but only with chunked logits and a small teacher KV budget. It is fragile and slower than time-slicing.

### Cited Findings
- Vocab and architecture alignment between teacher and student (vocab_size 248320, identical BPE) — see Q2 sources: [3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/config.json), [3.5-9B config](https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/config.json)
- Student size is 9,653,104,368 params (bf16), about 19.3 GB. — [HF API Qwen3.5-9B](https://huggingface.co/api/models/Qwen/Qwen3.5-9B)
- Teacher FP8 size is 27.78B params total, about 28 GB. — [HF API Qwen3.8-27B-FP8](https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8)

### Inferences
**Co-resident budget at 16K tokens**:

| Component | Memory |
|---|---|
| Teacher FP8 in vLLM, weights + cudagraphs | ~30 GB |
| Teacher KV for one 16K scoring sequence (64 KiB/token) | ~1 GB |
| Student bf16 weights | 19.3 GB |
| LoRA params + Adam states (r=64) | < 1–2 GB |
| Student activations with gradient checkpointing at 16K | ~5–10 GB |
| **Full student logits at 16K × 248,320 in fp32** | **~15.9 GB (8 GB bf16), plus another copy for softmax/grad** |
| Same, if teacher full-vocab logits are also materialized | about the same again |

- That totals roughly 80+ GB unless logits are computed in chunks (e.g., 1–2K token chunks via a chunked or fused cross-entropy/KL).
- Conclusion: co-resident is only feasible with chunked loss, a teacher `gpu_memory_utilization` of about 0.40, and 8K rather than 16K sequences.

**Recommended approach**:
- Reverse-KL on-policy distillation needs only the teacher's logprob of each *sampled* student token, i.e. prompt logprobs with k=0 or 1. That is tiny to store: 4 bytes per token.
- Requesting top-k=20 teacher logprobs per position enables top-k forward KL. Storage is about 20 × (4 + 4) B = 160 B per token, so about 2.6 MB per 16K sequence, which is fine for thousands of trajectories on Drive.
- Time-slice on one GPU:
  1. Launch the student vLLM and sample.
  2. Kill it and launch the teacher vLLM with `prompt_logprobs`, offline `LLM(...)` scoring on `prompt_token_ids`.
  3. Kill it and train.
  4. Repeat per iteration.
- Teacher scoring is prefill-only, so it is fast: no decode, and it is compute-bound on A100 at bf16 compute speed through Marlin.

### Gaps
- I found no primary source measuring this exact pipeline's memory on an A100.
- I did not verify the exact vLLM 0.32.0 API for returning prompt logprobs via the OpenAI completions endpoint vs offline `SamplingParams(prompt_logprobs=k)`, or the `--max-logprobs` cap.
- I did not verify whether vLLM's prompt-logprob computation for the hybrid Gated DeltaNet model has any limitations with prefix caching. It is safer to disable prefix caching in the scoring pass.

## Recommendation (summary)
- **Teacher**: `Qwen/Qwen3.8-27B-FP8` (official, Apache-2.0, released 2026-08-13/14), served with vLLM 0.32.0 in a uv venv on the A100 using the command in Q3.
  - For trajectory generation, use temperature 1.0, top_p 0.95, top_k 20, and reasoning_effort medium (or xhigh for the hardest tasks).
  - Re-render the saved messages with Qwen3.5-9B's template for SFT.
- **Fallback teacher**: Qwen/Qwen3.6-27B-FP8 (same tokenizer, pre-window, weaker), or Qwen3.6-35B-A3B-FP8 for higher throughput.
- **On-policy distillation**: feasible, because vocab and token IDs align. Do teacher scoring as a separate vLLM prompt-logprobs pass on the student's exact token IDs, and store top-k or sampled-token logprobs. Do not co-locate teacher and student during training.
