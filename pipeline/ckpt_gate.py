"""Checkpoint gate: runs beside train_sft.py. Every --every steps it merges the new checkpoint on the CPU, builds
a Q8_0 GGUF (same build as the base reference), runs the fast suite on the GPU next to training, and scores it.
If a checkpoint falls more than --margin below the best so far, it writes the STOP file: training stops and the
best checkpoint is kept. State: <run_dir>/gate_state.json. Log lines starting with "GATE" are picked up by the
watchdog and turned into alerts.

    python ckpt_gate.py --run_dir $DRIVE/sft_v3 --base_fast $DRIVE/eval_v2/base_fast/results.json \
        --stop_file /content/STOP_TRAINING --every 40
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates  # noqa: E402


def trainer_alive():
    return subprocess.run(["pgrep", "-f", "[t]rain_sft.py"], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--base_fast", required=True)
    ap.add_argument("--stop_file", required=True)
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--margin", type=float, default=2.5, help="allowed drop (fast-score points) below the best")
    a = ap.parse_args()
    state_p = os.path.join(a.run_dir, "gate_state.json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {"scores": {}, "best": None}
    base = gates.fast_score(json.load(open(a.base_fast)))
    print(f"GATE base fast score {base:.2f}", flush=True)
    while True:
        cks = sorted((int(p.rsplit("-", 1)[1]), p) for p in glob.glob(os.path.join(a.run_dir, "checkpoint-*")))
        todo = [(n, p) for n, p in cks if n % a.every == 0 and str(n) not in state["scores"]]
        if not todo:
            if not trainer_alive():
                break
            time.sleep(120)
            continue
        n, p = todo[0]
        time.sleep(90)  # let the trainer finish writing the checkpoint
        keep = os.path.join(a.run_dir + "_keep", f"checkpoint-{n}")
        os.makedirs(keep, exist_ok=True)
        for f in ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json"):
            if os.path.exists(os.path.join(p, f)):
                shutil.copy(os.path.join(p, f), keep)
        merged, out = "/content/gate_merged", os.path.join(a.run_dir, "gates", f"ck{n}")
        try:
            subprocess.run([sys.executable, os.path.join(HERE, "merge.py"), "--adapter", keep, "--out", merged,
                            "--device", "cpu"], check=True, stdout=subprocess.DEVNULL)
            gguf = gates.build(merged, f"ck{n}", "Q8_0", "/content/gate_models")
            shutil.rmtree(merged, ignore_errors=True)
            res = gates.evaluate(gguf, "fast", out, tag=f"ck{n}", port=8091)
            os.remove(gguf)
        except Exception as e:
            print(f"GATE ERROR checkpoint {n}: {type(e).__name__}: {e}", flush=True)
            state["scores"][str(n)] = None
            json.dump(state, open(state_p, "w"), indent=2)
            continue
        s = gates.fast_score(res)
        state["scores"][str(n)] = s
        best = state["best"]
        if best is None or s > state["scores"][str(best)]:
            state["best"] = best = n
        json.dump(state, open(state_p, "w"), indent=2)
        nt = res["nothink"]
        print(f"GATE ck{n}: score {s:.2f} (base {base:.2f}, best ck{best} {state['scores'][str(best)]:.2f}) | "
              f"AL {nt['al']['passed']}/14 opencode {nt['al_opencode']['passed']}/14 sci {nt['sci']['overall']}% "
              f"max_repeat {max(nt['al']['max_repeat'], nt['al_opencode']['max_repeat'])}", flush=True)
        if s < state["scores"][str(best)] - a.margin:
            open(a.stop_file, "w").write(f"ck{n} score {s:.2f} < best ck{best} {state['scores'][str(best)]:.2f} - {a.margin}")
            print(f"GATE STOP: regression at ck{n}; keeping ck{best}", flush=True)
            break
    print(f"GATE DONE best=ck{state['best']} scores={state['scores']}", flush=True)


if __name__ == "__main__":
    main()
