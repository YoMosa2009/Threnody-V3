# Threnody — Master Plan

> *threnody* (n.) — a song of lamentation for the dead; a wailing ode.
> Anthropic named its models after literary forms — Haiku, Sonnet, Opus, Fable. A threnody is
> one too: the darkest one. Penderecki's *Threnody to the Victims of Hiroshima* is among the most
> frightening pieces of music ever written. The name is meant to sound like an elegy for every
> bug it meets.

**Threnody-9B** is the open-weight successor to Fable-Coder-V4: an agentic coding model for
multi-turn, long-horizon work inside CLI coding harnesses (Claude Code–style agents, OpenCode,
Aider, OpenHands, mini-swe-agent), small enough to run on a 12 GB consumer GPU.

License: Apache-2.0 (weights + code). All training data permissively licensed.

## Decisions (with evidence)

| Decision | Choice | Why |
|---|---|---|
| Base model | `Qwen/Qwen3.5-9B` (Apache-2.0, 2026-03) | Fits 12 GB VRAM at Q4–Q6; BFCL-V4 66.1, TAU2 79.1; native 262K context; native tool format (`qwen3_coder`); thinking/non-thinking. See `reports/`. |
| Not continuing from V4 | — | V4 ≈ its base (+2 pts on 16 tasks), contaminated with bench tasks, Qwen2.5 arch can't take its LoRA. See `docs/01_V4_ANALYSIS.md`. |
| Fine-tune method | bf16 LoRA r=64, α=32 (rsLoRA off), all linear layers of the language model, LR 1e-4, cosine, eff. batch 16, assistant-only loss | "LoRA Without Regret" (LR ≈10× full-FT); Unsloth advises against 4-bit QLoRA on Qwen3.5 |
| Sequence length | 16,384 (packing off, length-filtered data) | Agent trajectories are long; V4's 4k truncated them |
| Data | ~6k multi-turn trajectories + tool + reasoning + defensive security + replay; all decontaminated against OmniAgent-Bench | `docs/02_DATA.md` |
| Forgetting | 15% replay + adapter merge-strength sweep (0.5 / 0.75 / 1.0) | ICML'26 task-arithmetic finding; LoRA forgets less |
| RL | Optional GRPO on in-process unit-test tasks if budget remains | Multi-turn SWE RL needs Docker — not possible in Colab |
| Hardware | Colab **A100** (user's account, ~200 CU), all work in the browser notebook | user requirement |
| Storage | Google Drive `MyDrive/Threnody/` | Colab VMs are ephemeral |
| Code | Public GitHub `YoMosa2009/Threnody`, cloned by the notebook | approved by user |

## RAM-footprint strategy
1. **Architecture win (free):** Qwen3.5 is hybrid — only 8/32 layers are full attention (4 KV heads × 256 dim). KV cache ≈ 32 KiB/token in fp16 → 128K context ≈ 4 GiB fp16 / **2 GiB q8_0**. Qwen2.5-7B (V4) needs ~56 KiB/token (7 GiB at 128K).
2. **Text-only release:** drop the ~0.45B-param vision tower (no `mmproj` needed) — coding model doesn't need it.
3. **imatrix quant ladder** calibrated on Threnody's own agentic traces: Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, IQ3_M (labelled degraded).
4. **Recommended runtime flags:** `-fa on -ctk q8_0 -ctv q8_0` (+0.0008 KLD).
5. **Speed:** MTP head kept in HF weights for vLLM/SGLang speculative decoding.
6. Measure KLD per quant on held-out agentic data and publish.

## Phases & budget (A100 40GB ≈ 5.4 CU/h, 80GB ≈ 7.5 CU/h)
| # | Phase | Est. CU |
|---|---|---|
| 0 | Baseline OmniAgent-Bench on Colab: Qwen2.5-Coder-7B, Fable-V4, Qwen3.5-9B (same quant Q4_K_M, temp 0, llama.cpp CUDA) | ~5 |
| 1 | Build dataset (CPU/High-RAM runtime mostly) | ~5 |
| 2 | Pilot SFT (200 steps) → measure throughput, sanity-check loss masks | ~10 |
| 3 | Full SFT (~2 epochs) | ~80–100 |
| 4 | Merge sweep + quick eval | ~10 |
| 5 | (Optional) GRPO on unit-test tasks | ~30 |
| 6 | GGUF conversion, imatrix, quant ladder, KLD | ~10 |
| 7 | Final OmniAgent-Bench + extended eval | ~5 |
| | **Reserve** | ~25 |

## Evaluation
- **OmniAgent-Bench** (user's suite, 16 tasks) — run unmodified for V4 comparability; models in non-thinking mode at temp 0 with `max_tokens` large enough, plus thinking-mode run with `<think>` stripped.
- Report per-category and per-task; note that 16 tasks ⇒ ±6 pts noise per task flip.
- Optional extended closed-loop eval (multi-turn sandbox like V4's AGENTIC_EVAL) if budget allows.

Progress log: `docs/PROGRESS.md`.

## Out of scope for Threnody-9B v1 (decided 2026-11-03)
- **Biology / chemistry / general science**: not a v1 target (v1 data is coding/agent/terminal + security/tool/IF regen).
  Planned for a later Threnody version with its own science data from a June-2026+ open-weight teacher.
