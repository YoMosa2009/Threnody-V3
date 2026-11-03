# Threnody

> *threnody* — a song of lamentation for the dead. Haiku, Sonnet, Opus, Fable… and the darkest form of all.

**Threnody-9B** is an open-weight agentic coding model built for long, multi-turn work inside CLI coding
harnesses. It is the successor to [Fable-Coder-V4](https://github.com/YoMosa2009/Fable-Coder-V4), rebuilt
from scratch on `Qwen/Qwen3.5-9B` and trained on execution-grounded agent trajectories.

> **Status: in training.** Results, weights and GGUFs will be published here when they're ready.
> Progress log: [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Why it's different from Fable-Coder-V4
| | Fable-Coder-V4 | Threnody-9B |
|---|---|---|
| Base | Qwen2.5-Coder-7B (2024) | Qwen3.5-9B (2026), hybrid linear/full attention |
| Context | 4k (Modelfile) | 262k native |
| Training data | 3.5k single-turn samples | multi-turn agent trajectories with real execution feedback, tool calling, defensive security, replay |
| Loss | whole sequence | assistant turns only |
| Decontamination | none (bench tasks were in the train set) | 13-gram + entry-point filter vs OmniAgent-Bench |
| Thinking | none | interleaved `<think>` before each tool call; can be turned off |

## Low memory by design
- Only 8 of 32 layers use full attention, so the KV cache is ~32 KiB/token (fp16), about 2 GiB at 128k context with a `q8_0` cache.
- Text-only GGUFs (no vision tower).
- Quants from Q8_0 down to IQ3_M, built with an importance matrix calibrated on agent traces; a KL-divergence report is published for every quant.

## Repository
```
docs/       plan, V4 post-mortem, progress log
reports/    deep-research report (base models, data, recipes, memory)
pipeline/   decontam, build_data, train_sft, merge, export_gguf, bench_colab
notebooks/  Threnody.ipynb — the Colab A100 notebook that runs everything
```

## License
Apache-2.0. All training data is permissively licensed (see `docs/00_PLAN.md`).
