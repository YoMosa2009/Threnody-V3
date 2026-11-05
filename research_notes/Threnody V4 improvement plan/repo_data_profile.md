# Repository data profile: V4 signals from the V3 artifacts

_Primary analysis of files in YoMosa2009/Threnody-V3 (main @ a56e31a), 2026-11-05. All numbers below are computed directly from the repository files; nothing here is from external sources._

## 1. What per-item data exists

| File | Grain | Rows | Usable for V4? |
|---|---|---|---|
| `results/v3_final.json` | one summary per model × suite | ~6 suites × 3 models | Summary only; no per-task data |
| `results/v3_gate_s0.75.json`, `v3_gate_s1.0.json` | summary + gate checks per candidate | 2 | Summary only; conflicts with gate code (see docs/KNOWN_DISCREPANCIES.md §4) |
| `results/v3_training_report.json` | run metadata + 6 loss points | 1 | Loss values only at steps 0/10/…/50; no log_history |
| `results/v3_quant_report.json` | one row per quant | 7 | KLD / same-top-token on calibration text |
| `results/v3_redteam_report.json` | 4 free-text categories | 4 | No counts |
| `results/biochem_hard_final.json` | **one row per question × model** | **844 × 2 (Base Q4_K_M, V2 Q4_K_M)** | **Only per-item dataset in the repo** |

**V4 implication:** no per-task agent trajectories, run logs, or `log_history.json` are published for V3. V4 must log per-task results (pass/fail, turns, tool calls, max_repeat, seconds, error, full trajectory) and training logs, or nothing can be audited.

## 2. biochem_hard_final.json profile

Columns: `id` (identifier), `category` (dimension: biology/chemistry), `subdomain` (dimension, 5 values), `answer` (A–J), `predicted` (A–J or null), `correct` (bool), `raw_response` (text, truncated to 200 chars in storage).

Subdomain sizes: MMLU-Pro biology 250, MMLU-Pro chemistry 250, MMLU college_biology 144, college_chemistry 100, medical_genetics 100 (= 844).

### 2.1 Finding: scores are dominated by answer-extraction failure (max_tokens = 128)

| Model | Rows with **no extracted answer** | Reported accuracy | Accuracy **when an answer was extracted** |
|---|---|---|---|
| Base (Q4_K_M) | **603 / 844 (71.4%)** | 214/844 = 25.4% | 214/241 = **88.8%** |
| V2 (Q4_K_M) | **514 / 844 (60.9%)** | 300/844 = 35.5% | 300/330 = **90.9%** |

Per subdomain (Base): MMLU-Pro chemistry produced an extractable answer in only **12/250 (5%)** of rows, so the reported 2.0% chemistry score reflects truncation, not knowledge. college_biology: 66/66 extracted answers were correct (100%).

Cause (code): `eval/biochem_hard/eval_biochem_hard.py` `ask(base_url, q, max_tokens=128)`, with no CLI flag to raise it. Multi-step chemistry answers rarely finish within 128 tokens and never print "The answer is (X)". The stored `raw_response` is cut to 200 characters, so it cannot be re-scored.

**V4 implication:** BioChem-Hard as implemented mostly measures **brevity under a 128-token cap**. That explains why its numbers (Base 25.4%) disagree so badly with the MMLU-Pro sample in `sci_eval.py` (`max_tokens=1024`, Base 76.0%). V2's "advantage" here (+10.1 pp) is mostly V2 answering within budget more often (extracted-answer rate 39.1% vs 28.6%). Retire this suite or fix it: raise max_tokens to ≥2048 non-thinking / ≥8192 thinking, store full responses, add a fallback extractor, and report the extraction-failure rate as its own metric.

### 2.2 Finding: non-deterministic IDs prevent paired comparison

College-subject IDs are `mmlu_{subject}_{random.randint(10000, 99999)}`, regenerated every run. Only the 500 MMLU-Pro rows can be paired across models. The Base file also has a duplicate ID (`mmlu_college_biology_24314`, a random collision).

Paired MMLU-Pro rows (n = 500): Base-only correct 10, V2-only correct 45. A McNemar test would call this significant, but per §2.1 it mostly reflects answer length, not knowledge.

**V4 implication:** use stable IDs (dataset question_id or a hash of the question text) so that paired tests (McNemar / paired bootstrap) are possible.

### 2.3 Answer-letter distribution

Gold answers concentrate on A–D (575/844 = 68%) even though there are up to 10 options, because MMLU college subsets have only 4 options. No strong position bias in predictions beyond that.

## 3. AgentLoop harness structure (from code)

- `eval/agentloop/tasks.py` defines **14 tasks**. `harness.py` uses these same 14 built-in tasks for every schema unless `--tasks_file` is given.
- **So "AgentLoop In-Harness (14)" and "AgentLoop Held-Out opencode (14)" are the same 14 tasks run with different tool schemas.** The held-out part is the *tool schema* (`HELDOUT_SCHEMAS = ["opencode", "codex"]`), not the tasks. The tasks' own repos/tests may overlap with training data (V3 claimed 13-gram decontamination vs AgentLoop, but the decontam output is not published).
- Same tasks, two schemas: Base 11/14 (agentloop) vs 12/14 (opencode); V3 10/14 vs 13/14. A 3-task swing between schemas on identical tasks shows that schema/format effects are as large as the claimed improvements. The 14-task suites cannot separate "learned coding skill" from "fits this tool format".
- `codex` (second held-out schema) is evaluated only in suite `full`. `run_v3.sh` used `mid`, so **codex was never run for V3.**
- Harness sampling: temperature 0.0 non-thinking (0.6 thinking), top_p 0.95, max_turns 30, max_tokens 8192. Greedy decoding with no repetition/presence penalty is a known loop risk for Qwen-family models; check the official recommended sampling settings.
- `max_repeat` counts only *consecutive* identical actions. A–B–A–B oscillation is not detected.
- AgentLoop-Hard is 30 tasks from `hard_eval.jsonl` in the teacher directory (Drive). It is not in the repo, so it cannot be reproduced from GitHub.

## 4. Training-run signals

- 662 trajectories / 2,895,493 tokens ≈ **4,374 tokens per trajectory**, at max_len 32,768. Roughly 8% of the 35M-token target.
- Loss: 1.482 → 0.554 train, 0.613 eval (only 6 points). The train/eval gap (0.06) is small, but with 56 steps and no curve shape, under- or over-fitting cannot be judged.
- Instability at adapter strength 1.0 (V2: rep 27–29 at s0.5–s1.0; V3: s1.0 opencode max_repeat 4, In-Harness 5, Hard 6) with recovery at reduced strength suggests that the full update moves the model too far for the amount and quality of data. That points to an LR/steps/data-scale mismatch. It is a hypothesis to test in V4, not a finding.
- The checkpoint gate (`ckpt_gate.py`, `fast_score` = AgentLoop passes + opencode passes + 14 × sci) selects on the "held-out" schema. This is selection leakage.

## 5. Quantization data

Same-top-token vs BF16 (on the imatrix calibration text): Q8_0 97.756, Q6_K 96.664, Q5_K_M 96.127, Q4_K_M 94.058, IQ4_XS 93.161, IQ3_M 89.615. Median KLD rises roughly 85× from Q8_0 to IQ3_M (0.000088 → 0.007465). No task evaluation was run per quant. Benchmarks used Q8_0 (V3 pipeline), but `biochem_hard_final.json` used Q4_K_M, so comparisons across these files also mix quantization levels.

## 6. Top V4 signals from the data

1. **Evaluation bugs are a larger source of error than model differences** (BioChem-Hard truncation; same-task "held-out" suite; selection leakage; missing codex run). Fix evaluation before training.
2. **Data scale is about 10× below plan.** 662 trajectories is small compared with published agent SFT sets (verify against the external research notes).
3. **Reduced adapter strength was used as a stabilizer.** Find out why the full update is unstable (LR, data volume, loop-inducing data, greedy decoding) instead of diluting it.
4. **No per-item logs.** Make full trajectory logging and stable IDs a requirement.
