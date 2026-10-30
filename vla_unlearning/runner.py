"""Runs one stream of learn/unlearn ops (e.g. L A, L B, L C, U B, L D, L E, U E) for one
method and seed, and writes the success matrix: one row per op, one column per task.

- Every op runs as its own subprocess, so GPU memory is fully released between ops.
- Checkpoints are stored by op prefix: results/ckpt/<learn settings>/seed<k>/L_A/L_B/L_C/<method>/U_B/...
  Ops before the first U are shared by all methods, so L_A L_B L_C is trained once.
- Finished ops and evals are skipped on re-run (delete an op folder to redo it).
"""

import json
import os
import subprocess
import sys

import pandas as pd
import yaml

from vla_unlearning.evaluate import evaluate
from vla_unlearning.paths import CONFIGS, DATA, RESULTS, ROOT


def load_yaml(kind, name):
    with open(CONFIGS / kind / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def op_dir(stream, ops_so_far, method, seed):
    learn = stream["learn"]
    root = RESULTS / "ckpt" / f"{stream['suite']}_steps{learn['steps']}_bs{learn['batch']}" / f"seed{seed}"
    names = [op.replace(" ", "_") for op in ops_so_far]
    first_u = next((i for i, op in enumerate(ops_so_far) if op.startswith("U")), len(names))
    if first_u < len(names):
        names.insert(first_u, method)  # from the first U on, the model depends on the method
    return root.joinpath(*names)


def sh(args, log_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    with open(log_path, "a") as log:
        subprocess.run([sys.executable, "-m", *map(str, args)], cwd=ROOT, env=env, stdout=log, stderr=log, check=True)


def run(stream_name, method_name, seed):
    stream = load_yaml("streams", stream_name)
    method = load_yaml("methods", method_name) if method_name != "none" else None
    suite = DATA / stream["suite"]
    task_dir = lambda t: suite / "tasks" / f"task_{stream['tasks'][t]}"  # noqa: E731
    replay_dir = lambda t: suite / "replay" / f"task_{stream['tasks'][t]}"  # noqa: E731
    instructions = json.loads((suite / "tasks.json").read_text())
    env_name = lambda t: "libero_sim/" + instructions[str(stream["tasks"][t])].replace(" ", "_")  # noqa: E731

    prev, known, seen, rows = stream.get("base_model", "nvidia/GR00T-N1.6-3B"), [], [], []
    for i, op in enumerate(stream["ops"]):
        kind, task = op.split()
        out = op_dir(stream, stream["ops"][: i + 1], method_name, seed)
        out.mkdir(parents=True, exist_ok=True)
        log = out / "log.txt"
        ckpt = out / "model"
        print(f"[{op}] {out}", flush=True)

        if kind == "L":
            if not ckpt.exists():
                learn = stream["learn"]
                sh(["vla_unlearning.learn", "--init", prev, "--task", task_dir(task),
                    "--replay", *[replay_dir(t) for t in known],
                    "--out", out, "--steps", learn["steps"], "--lr", learn["lr"], "--batch", learn["batch"],
                    "--seed", seed], log)  # fmt: skip
            known.append(task)
            seen.append(task)
        else:
            known.remove(task)
            if not ckpt.exists():
                assert method is not None, "stream has U ops: pass a method"
                if method["method"] == "purge":  # only drop the task from the replay buffer
                    ckpt.symlink_to(prev)
                else:
                    sh(["vla_unlearning.unlearn", "--init", prev, "--forget", task_dir(task),
                        "--retain", *[replay_dir(t) for t in known],
                        "--method", json.dumps(method), "--out", out, "--seed", seed], log)  # fmt: skip

        # Evaluate every task seen so far (unlearned ones included) + any extra tasks (oracle).
        cache = out / "eval.json"
        sr = json.loads(cache.read_text()) if cache.exists() else {}
        todo = [t for t in seen + stream.get("also_eval", []) if t not in sr]
        if todo:
            with open(log, "a") as f:
                done = evaluate(ckpt, [env_name(t) for t in todo], stream["eval"], f)
            sr.update({t: done[env_name(t)] for t in todo})
            cache.write_text(json.dumps(sr, indent=2))

        cost = out / "cost.json"
        minutes = json.loads(cost.read_text())["gpu_minutes"] if cost.exists() else None
        rows.append({"op": op, **sr, "unlearn_minutes": minutes})
        print(f"[{op}] success rates: {sr}", flush=True)
        prev = ckpt

    res = RESULTS / stream_name / method_name / f"seed{seed}"
    res.mkdir(parents=True, exist_ok=True)
    columns = ["op", *stream["tasks"], "unlearn_minutes"]
    pd.DataFrame(rows).reindex(columns=columns).to_csv(res / "matrix.csv", index=False)
    (res / "config.yaml").write_text(yaml.safe_dump({"stream": stream, "method": method, "seed": seed}))
    print(f"wrote {res / 'matrix.csv'}")
