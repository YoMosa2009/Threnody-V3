# Compute-efficient fine-tuning recipes for agentic coding on a single Colab GPU (as of Nov 2026)

Scope note: about 20 tool calls were used. Primary sources fetched were the Thinking Machines "LoRA Without Regret" post, the TRL SFT and GRPO trainer docs (main branch, current as of Nov 2026), the Unsloth Qwen3 and Memory-Efficient RL docs, and McCormick's Colab pricing table (checked 2026-03-10). Agentic SFT results come from the SWE-smith, SWE-Gym and Skywork-SWE papers via search snippets and abstracts. All of those are 2024-2025 results and are marked as older below. Every item without a URL is in Inferences or Gaps.

## Q1. Unsloth vs plain TRL/PEFT on Colab: speed and memory, Qwen3 and Qwen3-MoE QLoRA support, long context

### Takeaway
Unsloth remains the most memory-efficient single-GPU option. It claims about 2x speed and about 70% less VRAM, fits Qwen3-30B-A3B QLoRA in about 17.5 GB and Qwen3-14B on a 16 GB T4, and gets much longer context for RL than TRL. TRL is now a reasonable alternative for SFT: assistant-only loss is auto-patched for Qwen3, plus padding-free BFD packing, activation offloading and gradient checkpointing on by default. Neither source gives an independently benchmarked 16k-32k SFT VRAM number for 14B-32B models.

### Cited Findings
- Unsloth says Qwen3 and Qwen3-MoE fine-tuning is "2x faster with 70% less VRAM" and supports "8x longer context lengths". These are vendor claims. — [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen)
- Qwen3-30B-A3B QLoRA "works on just 17.5GB VRAM with Unsloth". Caveat: the full 16-bit MoE checkpoint is downloaded and converted to 4-bit on the fly, so it needs extra system RAM and disk, and pre-quantized 4-bit BnB MoE checkpoints cannot be loaded directly. — [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen)
- Qwen3-14B "fits comfortably in a Google Colab 16GB VRAM Tesla T4". The notebook default is max_seq_length=2048, and Qwen3's native context is 40,960. — [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen)
- Unsloth disables fine-tuning of the MoE router layer by default and advises against it. The page also mentions a "2026 Faster MoE" update for 30B-A3B and 235B-A22B. — [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen)
- Unsloth offers full fine-tuning (`full_finetuning=True`) and 8-bit loading (`load_in_8bit=True`). — [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen)
- No independent benchmark of the 17.5 GB figure was found. Spheron's third-party GPU tool estimates about 17 GB for INT4 30B-A3B inference, with LoRA adding about 50%. — [Spheron GPU guide](https://www.spheron.ai/tools/gpu-recommender/Qwen/Qwen3-30B-A3B/)
- TRL SFTConfig defaults and options:
  - `gradient_checkpointing=True` by default.
  - Supports `activation_offloading`, `use_liger_kernel`, and `packing=True` with `packing_strategy="bfd"`, which is padding-free and requires FlashAttention 2 or 3.
  - `max_length` defaults to 1024. Set it explicitly for long trajectories.
  - The default `learning_rate` is 2e-5 (a full-FT value). TRL's PEFT tip says adapters "typically use a higher learning rate (≈1e-4)".
  - When the model is passed as a string, dtype defaults to float32 unless `model_init_kwargs={"dtype": torch.bfloat16}` is set. This is a memory trap.
  — [TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)
- TRL says the Unsloth integration gives "up to 2× faster with up to 70% less VRAM". — [TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)

### Inferences
- On Colab, use Unsloth for 14B-32B dense and 30B-A3B QLoRA. Plain TRL+PEFT+bitsandbytes is fine for 4B-8B, especially when you want TRL's auto-patched assistant-only masking.
- For 30B-A3B, pick the A100 80GB or G4 runtime, not the 40GB one. The on-the-fly 4-bit conversion needs system RAM and disk, and the High-RAM option is tied to the 80GB A100.
- The predecessor's 4096 max length truncates most multi-turn agent trajectories. SWE-style trajectories are routinely more than 16k tokens. Target 16k-32k on A100 80GB/G4 with Unsloth and gradient checkpointing, or filter or trim trajectories to 8k-16k on L4.

### Gaps
- No primary-source VRAM table for Unsloth SFT of Qwen3-8B/14B/32B at 16k or 32k context on A100 40/80GB or L4. The older Unsloth blog tables (Llama 3.x era) were not re-fetched.
- No measured tokens/sec for Unsloth vs TRL on L4/A100 in 2026.
- Support for newer families (e.g., Qwen3-Coder, Qwen3.5, Devstral 2) was not verified in this pass.

## Q2. Assistant-only loss masking for multi-turn tool trajectories, tool role, thinking tokens, preserving thinking mode

### Takeaway
Train on assistant tokens only. In TRL, use `assistant_only_loss=True` on a conversational dataset with `messages` plus a `tools` column. TRL auto-patches the Qwen3 chat template with `{% generation %}` markers, so tool-role outputs, user turns and the system prompt get no loss. The predecessor trained on full text, which spends most of the gradient on tool outputs, file contents and prompts. That is a likely reason it barely moved. To keep Qwen3's thinking mode, Unsloth recommends about 75% reasoning to 25% non-reasoning data.

### Cited Findings
- `assistant_only_loss=True` computes loss "only on the assistant responses, ignoring user or system messages". It requires `{% generation %}` / `{% endgeneration %}` in the chat template, and "for known model families (e.g. Qwen3), TRL automatically patches the template". — [TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)
- Completion-only loss is the default for prompt-completion datasets and can be combined with `assistant_only_loss`. — [TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)
- SFTTrainer "fully supports" tool-calling SFT. The dataset needs `tool_calls` in assistant messages, `tool`-role messages for tool responses, and a `tools` column of JSON schemas. — [TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)
- Pre-tokenized datasets can supply `labels` with -100 directly, or `assistant_masks`. This is an escape hatch for custom masking, for example masking failed turns. — [TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)
- Thinking content can be included in assistant completions as `<think>...</think>` before the answer (TRL example). — [TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)
- Unsloth: fine-tuning on non-reasoning data may reduce reasoning ability. To retain it, use about 75% reasoning / 25% non-reasoning data. Their notebook mixes NVIDIA open-math-reasoning (75%) with FineTome (25%). — [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen)
- TRL also offers `loss_type="dft"` (Dynamic Fine-Tuning, arXiv 2508.05629) as an alternative SFT loss aimed at better generalization. — [TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)

### Inferences
- Check the mask by decoding the tokens where labels != -100 for one trajectory. Only assistant reasoning, assistant text and tool-call JSON should appear, never `<tool_response>` content.
- Qwen3's stock template strips `<think>` blocks from earlier assistant turns in multi-turn history. When training multi-turn thinking trajectories, either keep reasoning only on the last turn or use a template that preserves it, and make training match how the model is served. This comes from general knowledge of the Qwen3 template and was not verified in this pass.
- Unsloth's `train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")` is the usual ChatML pattern. This is from prior knowledge, not fetched. With it, `tool` turns (`<|im_start|>tool` or Qwen3's `<tool_response>` inside user turns) are masked only if they fall in the instruction region. Verify by decoding.

### Gaps
- Unsloth's `train_on_responses_only` docs for tool-role and thinking-token handling were not found (searches and the chat-templates page returned nothing).
- No controlled study found quantifying the gain from assistant-only vs full-text loss on agentic benchmarks.

## Q3. LoRA hyperparameters (rank, alpha, rsLoRA, LR, epochs, packing, NEFTune), LoRA vs full FT

### Takeaway
"LoRA Without Regret" (Thinking Machines, Sep/Oct 2025) supports these settings:
- Apply LoRA to all layers, especially MLP/MoE. Attention-only is clearly worse.
- Use an LR about 10x the full-FT LR. The TRL PEFT default is about 1e-4, and 1e-4 to 2e-4 is typical.
- Keep the effective batch small (about 32 sequences or fewer).
- Use enough rank for the dataset size.

The predecessor's LR 2e-5 is the full-FT default, about 10x too low for LoRA. That plausibly explains its +2 points.

### Cited Findings
- The optimal LoRA LR is about 10x full-FT. The fitted multiplier is 9.8 across 14 Llama/Qwen models, and about 15x for short runs under about 100 steps. — [Thinking Machines, LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- With 1/r scaling (alpha/r, alpha=32), the optimal LR is roughly rank-independent, varying less than 2x between rank 4 and 512. Rank 1 wants a lower LR. — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- LoRA on all layers, especially MLP and MoE, beats attention-only. Attention-only rank 256 underperforms MLP-only rank 128 at about equal parameter counts (Llama-3.1-8B). — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- For MoE, per-expert rank = total rank / number of active experts (8 for Qwen3). — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- LoRA tolerates large batches less well than full FT. The penalty grows with batch size, regardless of rank, and both methods do best at small batches. — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- LoRA matches full FT while the dataset fits its capacity and underperforms beyond it. SFT is estimated at about 1 bit per token of loss. — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- LoRA uses about 2/3 of full-FT FLOPs per pass, excluding attention FLOPs. — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- TRL hosts a "LoRA Without Regret" recipe page. — [TRL docs](https://huggingface.co/docs/trl/main/en/lora_without_regret)
- TRL: adapters "typically use a higher learning rate (≈1e-4)". `neftune_noise_alpha` is supported in SFTConfig. — [TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)
- "LoRA Learns Less and Forgets Less" (Biderman et al., 2024, older): standard low-rank LoRA underperforms full FT on code/math targets but better preserves out-of-domain ability, and keeps more base-like solution diversity on HumanEval. — [arXiv 2405.09673](https://arxiv.org/html/2405.09673v2)

### Inferences
Suggested recipe for 4B-14B dense:

| Setting | Value |
|---|---|
| LoRA rank | r=64-128 (32 is acceptable for under about 2k trajectories) |
| alpha | 32 (Thinking Machines parametrization). Or alpha=r with rsLoRA. Either way, keep alpha/r behavior consistent when changing rank. |
| target modules | all linear layers (q,k,v,o,gate,up,down) |
| LR | 1e-4 to 2e-4, cosine, 3-5% warmup |
| effective batch | 16-32 sequences |
| epochs | 2-3 over a curated set |
| packing | bfd packing / padding-free for short samples only. Long trajectories gain little. |
| NEFTune | optional, low priority (no agentic evidence found) |

With that recipe:
- About 1,000-3,000 trajectories at batch 16 gives about 120-560 steps.
- Full FT is not feasible for 14B+ on one Colab GPU, and LoRA is the right default.

### Gaps
- rsLoRA was not fetched. No 2026 primary evidence on rsLoRA vs alpha=32 for agentic SFT.
- No agentic-coding evidence for NEFTune was found.
- The TRL "LoRA Without Regret" page (specific recommended rank/LR per task) was not fetched in full.

## Q4. Avoiding catastrophic forgetting: replay ratios, low rank, merging

### Takeaway
The cheapest protections are LoRA itself (learns less, forgets less), mixing in general/reasoning replay data, and optional post-hoc scaling or merging of the adapter toward the base. Evidence that linear base+adapter merging helps agentic coding specifically is thin. A 2026 paper argues that plain interpolation is suboptimal for RL-trained agents.

### Cited Findings
- LoRA better maintains out-of-domain performance than full FT. — [LoRA Learns Less and Forgets Less](https://arxiv.org/html/2405.09673v2)
- Informal reproduction (blog, not peer-reviewed): 10% replay recovers about 73% of average forgetting, and LoRA barely forgets even without replay. — [DEV Community](https://dev.to/javieraguilarai/the-forgetting-you-dont-measure-3h11)
- Replay at scale for continual learning, folding each LoRA into the base: [arXiv 2505.12512](https://arxiv.org/pdf/2505.12512)
- ICML 2026: naive merging of public LoRA adapters cost -10.5pp on GSM8K and -18.3pp on HumanEval. Task Arithmetic at λ=0.5 stayed within a standard error of the unmerged baseline. — [Audit Before You Merge, ICML 2026](https://icml.cc/virtual/2026/74593)
- MergeBench: merging works better on stronger base models, and coefficient tuning and sparsification help preserve pretrained abilities. — [MergeBench (arXiv 2505.10833)](https://arxiv.org/pdf/2505.10833v4)
- AWS Nova docs describe linear merges (e.g., 0.3 fine-tuned / 0.7 base) as a way to recover general skills like coding. This is vendor guidance, not benchmarked. — [AWS Nova model merge](https://docs.aws.amazon.com/nova/latest/userguide/nova-model-merge.html)
- A 2026 paper (Reinforced Agent Merging) says simple weight interpolation of RL-trained agentic models "remains suboptimal". — [arXiv 2601.13572](https://www.arxiv.org/pdf/2601.13572)
- Unsloth's 75/25 reasoning/non-reasoning mix (Q2) doubles as replay for thinking ability. — [Unsloth Qwen3 docs](https://unsloth.ai/docs/models/qwen)

### Inferences
- Use 10-25% replay: general chat, plus reasoning traces generated by the base model itself (self-distillation keeps it on-distribution). Track 2-3 regression evals (e.g., a general benchmark, HumanEval/LiveCodeBench, a tool-call format test) at each checkpoint.
- Treat adapter scaling as a cheap knob: merge at λ in {0.5, 0.75, 1.0} and evaluate all three. This costs only eval time.
- The predecessor's DPO collapse (6 pairs x 80 steps) fits a tiny dataset being overfit for many epochs. Preference or RL stages need hundreds to thousands of prompts, a KL or reference anchor, and early stopping.

### Gaps
- No controlled study found on linear LoRA-to-base merging for agentic coding benchmarks (SWE-bench, Terminal-Bench, BFCL).
- TIES/DARE results specific to code agents were not found.

## Q5. Rejection-sampling fine-tuning, self-distillation with execution feedback, distillation from larger open teachers

### Takeaway
All the published small-model SWE agents in Q7 are trained on filtered successful trajectories, which is rejection sampling or distillation. The trajectories come from a stronger teacher and are kept only if tests pass. On this budget, the expensive part is generating trajectories (teacher inference plus sandbox execution), not training. Generate off-Colab with a cheap or free hosted open teacher, verify with tests, and spend Colab units on SFT.

### Cited Findings
- SWE-smith: expert trajectories were generated on task instances and only successful ones kept. With 500 successful trajectories, Qwen2.5-Coder-32B reached 28.2% on SWE-bench Verified. With 5,016 trajectories, SWE-agent-LM-32B reached 40.2%. (2025, older.) — [SWE-smith, arXiv 2504.21798](https://arxiv.org/pdf/2504.21798)
- SWE-Gym: 491 sampled agent trajectories (OpenHands) lifted Qwen2.5-Coder-32B to 20.6% Verified and 15.3% Lite. Performance was still rising at 491 trajectories. (Dec 2024, older.) — [SWE-Gym, arXiv 2412.21139](https://arxiv.org/pdf/2412.21139v2)
- A third-party summary gives the zero-shot baseline as 7.0% Verified. — [emergentmind](https://www.emergentmind.com/topics/swe-gym)
- Skywork-SWE: more than 8,000 runtime-validated successful trajectories gave 38.0% pass@1 Verified, rising to 47.0% with test-time scaling. Gains showed "no signs of saturation" as data grew. (Jun 2025, older.) — [Skywork-SWE, arXiv 2506.19290](https://arxiv.org/pdf/2506.19290)
- Public trajectory datasets exist and can serve as distillation data without generating your own:
  - SWE-smith trajectories: [HF SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/main)
  - Kwai-Klear 66k mini-swe-agent-plus trajectories, which trained Qwen3-8B to a reported 39.0% Verified: [mirror](https://gitcode.com/hf_mirrors/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k)

### Inferences
- The highest-impact, cheapest move is to SFT on 1-5k filtered, verified multi-turn trajectories from existing open datasets (SWE-smith trajectories, Klear 66k subset), deduplicated and length-filtered to your max length, plus 10-25% replay.
- Self-generated RFT on Colab is possible for 4B-8B: sample k=4-8 rollouts per task with vLLM, run tests, keep passing ones. Sandbox execution of SWE-style repos (Docker) is not available on Colab, so restrict this to function-level or unit-test tasks that run in-process.

### Gaps
- Kimi-Dev, Devstral, and Nemotron trajectory counts and recipes were not retrieved in this pass.
- No cost figures found for free or cheap 2026 inference providers for teacher rollouts.
- "SWE-Prime: Fewer Trajectories, Better Performance" (arXiv 2608.27449) appeared in search but was not read. It may be directly relevant to data efficiency.

## Q6. RL with verifiable rewards (GRPO/DAPO/Dr.GRPO/GSPO) on Colab

### Takeaway
Single-turn or short multi-turn GRPO with unit-test rewards is feasible on Colab:
- Qwen3-4B fits on a T4 with Unsloth Standby.
- 8B-14B fit on an A100 80GB with QLoRA.
- TRL's GRPOTrainer now defaults to DAPO-style token-level loss with beta=0, has GSPO via `importance_sampling_level="sequence"`, and supports multi-turn tool environments via `environment_factory`.

Full SWE-style multi-turn agentic RL (long rollouts, Docker sandboxes) on one Colab GPU is not demonstrated in any source found.

### Cited Findings
- Unsloth Memory-Efficient RL (Standby, shared vLLM weights) numbers:
  - Qwen3-4B on a T4 with num_gen=2 and Standby at 0.95 utilization: about 14.5 GiB, about 32K KV cache. Without Standby at 0.9 or 0.8, it ran out of memory.
  - Qwen2.5-3B on an A100 40GB with 8 generations: about 6K sequence length without Standby, 10K+ with it. TRL managed about 1K.
  - Qwen3-32B LoRA 16-bit on an H100 80GB: about 6,144 context.
  - Settings: `UNSLOTH_VLLM_STANDBY=1` before import, `gpu_memory_utilization=0.9-0.95`, `max_lora_rank=32`, at least 2 generations.
  — [Unsloth Memory-Efficient RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl)
- Unsloth long-context GRPO blog (Feb 2025, older): 90% less VRAM. Llama 3.1-8B at 20K context needed 54.3 GB vs 510.8 GB for TRL+FA2. — [Unsloth GRPO blog](https://unsloth.ai/blog/grpo)
- Unsloth docs report Qwen3-8B GRPO reaching 110K context on an H100 80GB (vLLM + QLoRA), per search snippet. — [Unsloth GRPO long context](https://unsloth.ai/docs/new/grpo-long-context)
- TRL GRPOConfig (current main):
  - Defaults: `loss_type='dapo'` (token-level normalization), `beta=0.0` (no reference model loaded), `epsilon=0.2` (DAPO recommends `epsilon_high=0.28`), `num_generations=8`, `scale_rewards='group'`.
  - `dr_grpo` removes length bias. `importance_sampling_level="sequence"` gives GSPO, which is "often" more stable per the GSPO paper.
  - `environment_factory` provides stateful multi-turn tool environments with `reset()`/`get_reward()` (experimental). `max_tool_calling_iterations` caps turns.
  - vLLM colocate is the default mode (`vllm_gpu_memory_utilization=0.3`), and `vllm_enable_sleep_mode` offloads weights.
  — [TRL GRPO Trainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- Thinking Machines: LoRA matches full FT for RL "even with small ranks", including rank 1, because policy gradient learns about 1 bit per episode. — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Multi-turn tool RL with trajectory-level GRPO rewards gives a weak signal. GTPO (turn-level credit, 2026) is proposed for this. — [ACL Anthology listing](https://aclanthology.org/people/anoop-deoras/unverified/)
- CodeGrep (2026) trained a 14B retrieval agent with GRPO + LoRA r=32 on an 8xB200 node, not single-GPU. — [arXiv 2608.05886](https://arxiv.org/pdf/2608.05886)

### Inferences
Suggested Colab RL stage, after SFT, on 4B-8B:

| Setting | Value |
|---|---|
| method | GRPO/DAPO with LoRA r=16-32 |
| LR | about 1e-5 to 5e-6 for the LoRA |
| generations | 4-8 per prompt |
| max completion | 2-4k |
| data | 300-1,000 prompts with in-process unit tests (LiveCodeBench/MBPP-style, or self-written tool-use tasks) |
| steps | 200-500 |

Rewards:
- 1.0 for all tests passing, partial credit by fraction of tests passed.
- A small format reward for valid tool-call JSON.
- A penalty for timeouts.

Filter out prompts where all k samples pass or all fail, since they give zero advantage. GSPO (sequence-level) is the safer choice for MoE (30B-A3B).

The earlier DPO collapse is a warning: use hundreds of prompts, not 6, and monitor an eval set every 25-50 steps.

### Gaps
- No source reports SWE-bench gains from single-GPU multi-turn agentic RL on 4B-32B models. Expected gain magnitude is unknown.
- No Unsloth GRPO VRAM figure for Qwen3-8B/14B on A100 40GB or L4.

## Q7. Published small-model agentic coding SFT results (trajectory counts vs gains)

### Takeaway
2024-2025 results show steep gains from a few hundred to a few thousand verified trajectories on 32B coders: about 500 trajectories gives 20-28% Verified, about 5k gives 40%, about 8k gives 38%. All report non-saturating scaling. An 8B model reached a reported 39% with 66k trajectories.

### Cited Findings

| Model | Trajectories | SWE-bench Verified | Source |
|---|---|---|---|
| SWE-Gym, Qwen2.5-Coder-32B | 491 | 7.0% → 20.6% (32.0% with verifier) | [arXiv 2412.21139](https://arxiv.org/pdf/2412.21139v2); baseline from [emergentmind](https://www.emergentmind.com/topics/swe-gym) |
| SWE-smith, 32B | 500 | 28.2% | [arXiv 2504.21798](https://arxiv.org/pdf/2504.21798) |
| SWE-smith, 32B | 5,016 | 40.2% | [arXiv 2504.21798](https://arxiv.org/pdf/2504.21798) |
| Skywork-SWE-32B | about 8,000 (model card: 8,209) | 38.0% (47.0% with test-time scaling) | [arXiv 2506.19290](https://arxiv.org/pdf/2506.19290) |
| Klear-Agent, Qwen3-8B | about 66k SWE-smith-based | 39.0% (reported) | [dataset README mirror](https://gitcode.com/hf_mirrors/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k) |

### Inferences
- With roughly 26-37 A100-hours, a 4B-14B QLoRA SFT on about 1-3k long trajectories at 16k context is plausible. Expect clear gains on the specific scaffold/format trained on, and smaller transfer to other harnesses. Match the training scaffold to the evaluation harness (e.g., OpenAgent-style tool schema).

### Gaps
- Devstral, Kimi-Dev, Nemotron, and SWE-Mirror recipes were not retrieved.
- No 2026 small-model (4B-14B) agentic SFT papers with trajectory-count ablations were read in full. SWE-Prime (arXiv 2608.27449) is a candidate.

## Q8. Colab practicalities: compute-unit burn rates, session limits, Drive checkpoints

### Takeaway
As of March 2026, measured burn rates are about 1.19 CU/h (T4), 1.71 (L4), 5.40 (A100 40GB), 7.52 (A100 80GB), and 8.71 (G4 = RTX PRO 6000 Blackwell, about 96 GB). At these rates 200 CU buys about 37 A100-40GB hours, about 27 A100-80GB hours, about 23 G4 hours or about 117 L4 hours. That is more than the 15-20 A100-hour assumption in the brief.

### Cited Findings

| GPU | CU/h | $/h | Hours per 100 CU |
|---|---|---|---|
| T4 | 1.19 | $0.12 | 84 |
| L4 | 1.71 | $0.17 | 58.5 |
| A100 40GB | 5.40 | $0.54 | 18.5 |
| A100 80GB (via the High-RAM toggle) | 7.52 | $0.75 | 13.3 |
| G4 (`nvidia-smi` shows RTX PRO 6000 Blackwell, about 96 GB) | 8.71 | $0.87 | 11.5 |

- Measurements checked 2026-03-10 in southern California. Prices may fluctuate with demand and location. — [McCormick, Colab GPUs Features & Pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
- Compute units cost $10 per 100. — [McCormick](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
- Other sources conflict: one 2026 article puts the T4 at 1.76 CU/h, and an unofficial estimate puts the A100 at 7.5 (likely the 80GB). The live rate is shown in Colab's Resources tab. — [search summary incl. deploybase.ai](https://deploybase.ai/articles/google-colab-fine-tune-llm-free-vs-pro-gpu-comparison)
- A German deal post says Google AI Pro subscriptions include 200 Colab CU per month and mentions A100 80GB and 96GB G4 access. This is a secondary consumer source. — [mydealz](https://www.mydealz.de/deals/google-ai-pro-200-google-colab-compute-units-wert-21eur-monatlich-inklusive-auch-im-5eurmonat-promo-abo80gb-nvidia-a100-96gb-g4-gpu-2857205)
- Trainer resume: `trainer.train(resume_from_checkpoint=True)` loads the last checkpoint in `output_dir`, including optimizer and scheduler state. Options include `save_steps`, `save_total_limit` and `push_to_hub` with `hub_strategy="every_save"`. — [TRL SFT Trainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)

### Inferences
- The G4 (96 GB Blackwell) may be the best value per VRAM for 30B-A3B or 32B at long context, but the Unsloth/bitsandbytes/vLLM stack on Blackwell was not checked. The A100 80GB is the safe choice for long-context QLoRA of 14B-32B. The L4 (24 GB, bf16, about 3x cheaper per hour than the A100 40GB) suits 4B-8B SFT, data generation and evals.
- Set `output_dir` to a Google Drive path, or push adapters to a private HF repo with `hub_strategy="every_save"`. Save every 25-50 steps with `save_total_limit=2-3`, and resume with `resume_from_checkpoint=True` after disconnects. Saving only LoRA adapters plus optimizer state keeps checkpoints small.

### Gaps
- No official 2026 Colab session-length or idle-timeout limits were found for Pro/Pro+/pay-as-you-go (the McCormick page does not cover them).
- The official Colab (consumer) pricing page was not retrieved. The cloud.google.com page is Colab Enterprise.
