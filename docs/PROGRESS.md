# Threnody — Progress Log

## 2026-11-02
- Created `G:\Threnody`; cloned `Fable-Coder-V4` and `OmniAgent-Bench` into `refs/`.
- V4 post-mortem written → `docs/01_V4_ANALYSIS.md`. Key: V4 ≈ base; benchmark contamination; LR 10× too low; single-turn only; no loss masking.
- Deep research (4 tracks + synthesis) → `reports/Agentic coding LLM training research.md`, notes in `research_notes/`.
- **Name chosen: Threnody** (base `Qwen/Qwen3.5-9B`). Plan → `docs/00_PLAN.md`.
- Verified Qwen3.5-9B config: hybrid Gated-DeltaNet/full-attention (1 in 4 full), 262K ctx, multimodal wrapper `Qwen3_5ForConditionalGeneration`, MTP head, chat template with interleaved `<think>` and XML `<tool_call><function=..>` format.
- Wrote `pipeline/decontam.py` (13-gram + entry-point + literal filter vs OmniAgent-Bench).
- User approved: public GitHub repo `YoMosa2009/Threnody`; Google Drive for checkpoints. All compute on Colab A100.
