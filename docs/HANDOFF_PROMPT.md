# Threnody — handoff prompt (paste into Claude on the other computer)

---

You are taking over **Threnody**: an open-weight fine-tune of **Qwen/Qwen3.5-9B** aimed at being extremely strong at multi-turn, long-horizon agentic coding and CLI/harness use, while staying good at knowledge, tool use and cyber, with a small RAM footprint. Repo: https://github.com/YoMosa2009/Threnody (public, Apache-2.0). Clone it and read `docs/STATUS.md`, `docs/PROGRESS.md` (newest entries at the bottom) and `docs/00_PLAN.md` first.

## Where things stand (2026-11-03 ~18:30 UTC)
- All training data is built and sits in Google Drive at `MyDrive/Threnody/`:
  - `data_final/` (3,707 train / 75 val, 52.4M tokens, with a pre-tokenized cache `tokenized_32768.pt`): Open-SWE-Traces (Qwen3.8-27B teacher, 5 harnesses), Qwen3.8 terminal traces, plus our own verified Qwen3.8-27B teacher data (71 agent trajectories that passed hidden tests and verified before finishing; 731 regenerated answers).
  - `markers/*.done`: teacher stages and data stages are done. **sft is NOT done** (no `sft.done`), and `sft_v2/` has no checkpoints yet.
- Training has never completed. Attempt 1 crashed (dtype bug, fixed in `pipeline/train_sft.py`). Attempt 2 ran fine on a G4 GPU at ~76 s/step (232 steps ≈ 5 h), but the Colab runtime died at step 41 when its browser tab closed. Checkpoints now save every 20 steps.
- Baselines (Q4_K_M, temp 0): Qwen3.5-9B base = OmniAgent-Bench 92.7% nothink / 84.8% think, AgentLoop 12/14 nothink / 14/14 think. Fable-Coder-V4 = 72.7% and 0/14.

## Your job
1. **Start the run in Google Colab in the user's real Chrome browser** (use Claude in Chrome or computer use; Colab's Drive sign-in opens a pop-up that the Claude desktop app's built-in browser blocks). The user is logged into Google and Drive in that Chrome.
   - Open `https://colab.research.google.com/github/YoMosa2009/Threnody/blob/main/notebooks/Threnody.ipynb`. If Colab serves an old copy without the `# RUN-ALL` / `# WATCHDOG` / `# LOG TAIL` cells, open it pinned to the latest commit SHA instead (`.../blob/<sha>/notebooks/Threnody.ipynb`).
   - Runtime > Change runtime type: **G4** (RTX PRO 6000 Blackwell, 96 GB) with High-RAM. An A100 is acceptable **only if it shows 80 GB**; 40 GB A100s and L4s are too small (training uses ~64 GB at 32k context).
   - Run cells in order: `# 1. GPU + Drive` → `# 2. Code + Python deps` → `# 3. llama.cpp (CUDA)` → `# RUN-ALL` → `# WATCHDOG`. Do not run cells 4–9 (baselines are done; RUN-ALL does SFT, merge and benchmarks itself).
   - **The user must click "Connect to Google Drive" / approve the OAuth themselves**, and accept any Colab warnings or terms themselves. Never enter credentials.
   - **Keep that Chrome tab open for the whole run.** Colab ends the session when no tab has the notebook open.
2. **Confirm training actually steps** (run the `# LOG TAIL` cell after ~8 min): you should see `train=3707 ...`, `LoRA on 248 linear modules`, and a progress bar advancing at roughly 75–80 s/it. Earlier failures left the GPU idle for hours unnoticed, so check this every time.
3. **Monitor.** `pipeline/watchdog.sh` writes `MyDrive/Threnody/heartbeat.txt` every 5 min and `MyDrive/Threnody/ALERT.txt` on a traceback, OOM, idle GPU, silent sft log (45 min), or when the pipeline finishes. Check them about every 30 min (Drive connector or the LOG TAIL cell); a heartbeat older than 20 min means the runtime died, so reconnect, rerun cells 1–3, RUN-ALL and WATCHDOG (markers make everything resume; SFT resumes from the newest checkpoint). Full status report every 1.5 h. Email the user (mosaalsaegh@gmail.com, they asked for this) when they must approve Drive again or something major breaks.
4. **After SFT**, RUN-ALL merges the LoRA at strengths 1.0 and 0.75, converts to Q4_K_M, rebuilds llama.cpp for non-A100 GPUs, and runs OmniAgent-Bench + AgentLoop in think and nothink modes → `MyDrive/Threnody/bench_threnody/summary.json`. Compare against the base model; the chosen merge must not regress on either suite.
5. Then follow "Next steps" in `docs/STATUS.md`: harder held-out eval (AgentLoop-Hard / recent LiveCodeBench subset), on-policy distillation from Qwen3.8-27B, final `pipeline/export_gguf.py` quant ladder + KLD report, Hugging Face upload, README results.

## Rules from the user
- **Teachers/data must come from June-2026-or-newer models only** (currently Qwen/Qwen3.8-27B-FP8, 2026-08, Apache-2.0). No 2025-era trajectories. No closed-model (Claude/GPT) outputs; their terms forbid training competing models.
- Only Qwen3.5-9B (no larger variants). All compute on Colab (~100+ credits left of the original 200; check the balance in Colab and keep idle GPU time to zero).
- Log every session in `docs/PROGRESS.md`, keep `docs/STATUS.md` current, commit and push to `main` (end commit messages with your Co-Authored-By line).

## Known issues / gotchas
- `causal_conv1d` isn't installed, so the DeltaNet conv runs a slow PyTorch fallback. Installing a matching wheel before the next run could speed training up noticeably; don't restart a healthy run just for this.
- `gen_teacher.py --mode code` crashes with a pyarrow "offset overflow" in the code_contests shuffle buffer, which is why the teacher_code data is empty. Fix it (smaller buffer or no shuffle) before any teacher re-run.
- Drive FUSE only syncs a file to Drive once its writer closes it; the watchdog writes local files and copies them, so heartbeat/ALERT are fresh but `logs/sft.log` in Drive can lag (read it inside the VM with LOG TAIL).
- A `!` cell that launches a nohup'd process hangs the cell; the notebook uses `subprocess.Popen(..., start_new_session=True)`. `pkill -f` patterns use the `[r]un_all` bracket trick so they don't kill their own shell.
- The Drive-cached llama.cpp binaries are built for sm_80 (A100); `run_all.sh` rebuilds for other GPUs before eval.
