"""Base-level checks of the machinery for running without Slurm (scripts/run_local.sh): two real worker
processes sharing a plan (local claims, --wait), taking over the claim of a dead worker, the calibration
plan, and the report on a fake calibration run. Plain Python (no torch, no GPU).
    python tests/test_calibration.py
"""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

os.environ["VLU_RUNS_DIR"] = tempfile.mkdtemp(prefix="vlu_cal_")  # before importing vlu
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate import VARIANTS, items  # noqa: E402
from submit import merge  # noqa: E402
from vlu.config import RUNS_DIR  # noqa: E402
from vlu.nodes import Node, run_seed  # noqa: E402
from vlu.worker import claim_dir, load_plan, plan_path, someone_working, status, try_claim  # noqa: E402

ENV = {**os.environ, "PYTHONPATH": str(ROOT)}

# --- Two local workers: a trainer, and an eval-only worker that must wait for it ------------------------
# The workers run the real vlu.worker loop with a stub command that just creates the output files.
STUB = r'''
import sys, time, pathlib
import vlu.worker as w
from vlu.nodes import Node
def command(run, item):
    node = Node.from_path(run, item["node"])
    files = [node.weights_path] if item["kind"] == "train" else [node.eval_dir(item["weights"], item["seed"]) / f"{t}.json" for t in w.eval_todo(run, item)]
    return [sys.executable, "-c", "import pathlib, sys, time; time.sleep(1.5)\nfor f in sys.argv[1:]:\n p = pathlib.Path(f); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')", *map(str, files)]
w.command = command
real_sleep = time.sleep
w.time.sleep = lambda s: real_sleep(0.5)  # poll faster than the real 60 s
sys.argv = ["worker", "--run", "two", "--kinds", sys.argv[1], *sys.argv[2:]]
w.main()
'''
run_seed("two", 0)
plan = merge([], [{"kind": "train", "node": "LA", "save_at": []}, {"kind": "eval", "node": "LA", "weights": "final", "seed": 0, "tasks": ["A"], "videos": 0},
                  {"kind": "train", "node": "LA/LB", "save_at": []}, {"kind": "eval", "node": "LA/LB", "weights": "final", "seed": 0, "tasks": ["A", "B"], "videos": 0}])
plan_path("two").write_text(json.dumps(plan))
trainer = subprocess.Popen([sys.executable, "-c", STUB, "train"], env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(0.5)  # as run_local.sh: training GPUs start first; nothing is ready for the evaluator yet
evaluator = subprocess.Popen([sys.executable, "-c", STUB, "eval", "--wait"], env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
out_t, out_e = trainer.communicate(timeout=120)[0], evaluator.communicate(timeout=120)[0]
assert trainer.returncode == 0 and evaluator.returncode == 0, (out_t, out_e)
assert all(status("two", i) == "done" for i in load_plan("two")), [status("two", i) for i in load_plan("two")]
assert "start eval:LA:final:s0" in out_e and "start eval:LA/LB:final:s0" in out_e and "start train" not in out_e, out_e
print("workers: the eval-only worker waited for training, then ran both evals; everything done")

# --- A claim of a dead local worker is taken over; a live one is not ------------------------------------
item = {"kind": "train", "node": "LC", "save_at": []}
dead = subprocess.Popen([sys.executable, "-c", "pass"]); dead.wait()
claim = claim_dir("two", item); claim.mkdir(parents=True)
(claim / "owner.json").write_text(json.dumps({"job": f"local{dead.pid}@{os.uname().nodename}"}))
assert not someone_working("two") and try_claim(claim), "the dead worker's claim must be taken over"
assert someone_working("two") and not try_claim(claim), "our own live claim must hold"
print("claims: a dead local worker's claim is taken over, a live one holds")

# Many workers racing to take over the same stale claim: exactly one may win.
race_item = {"kind": "train", "node": "LD", "save_at": []}
race = claim_dir("two", race_item); race.mkdir(parents=True)
(race / "owner.json").write_text(json.dumps({"job": f"local{dead.pid}@{os.uname().nodename}"}))
RACER = r"""
import sys, time
import vlu.worker as w
real_alive = w._job_alive
def slow_alive(job):  # everyone checks the owner first, then all act one after another
    alive = real_alive(job)
    time.sleep(max(0.0, float(sys.argv[1]) + 0.5 - time.time()) + 0.05 * int(sys.argv[2]))
    return alive
w._job_alive = slow_alive
time.sleep(float(sys.argv[1]) - time.time())  # start together
won = w.try_claim(w.claim_dir("two", {"kind": "train", "node": "LD", "save_at": []}))
print("WON" if won else "lost", flush=True)
time.sleep(max(0.0, float(sys.argv[1]) + 2.0 - time.time()))  # a winner keeps working: its claim stays live
"""
start = str(time.time() + 6)  # time for every process to start (imports) before the race
racers = [subprocess.Popen([sys.executable, "-c", RACER, start, str(i)], env=ENV, stdout=subprocess.PIPE, text=True) for i in range(8)]
results = [r.communicate(timeout=60)[0].strip() for r in racers]
assert sum(r.endswith("WON") for r in results) == 1 and sum(r == "lost" for r in results) == 7, results
assert json.loads((race / "owner.json").read_text())["job"].startswith("local")
(race / "owner.json").write_text("")  # a half-written owner file is "being created", not a crash
assert someone_working("two")
print("claims: 8 workers racing for one stale claim -> exactly 1 wins; a half-written owner file does not crash")

# A stream that unlearns everything it learned has the base model as retrain oracle (S3-like).
import argparse
import submit
submit.STREAMS["ALL_GONE"] = ["LA", "UA"]
ns = argparse.Namespace(run="two", nodes="", stream="ALL_GONE", methods="ga", no_oracle=False, zero_shot="", save_at="",
                        eval_weights="final,step_002000", eval_seeds="0", videos=0, no_eval=False, features=False, fisher=[])
built = submit.build_items(ns)
base_evals = [i for i in built if i["kind"] == "eval" and i["node"] == ""]
assert [i["weights"] for i in base_evals] == ["final"] and base_evals[0]["tasks"] == ["A"], base_evals
assert not any(i["kind"] == "train" and i["node"] == "" for i in built)
print("submit: a base-model oracle is evaluated (final weights only), not trained")

# --- The calibration plan --------------------------------------------------------------------------
cal = items()
trains = [i["node"] for i in cal if i["kind"] == "train"]
assert trains[:2] == ["LA", "LA/LB"] and len(trains) == 2 + len(VARIANTS)
assert all(Node.from_path("cal", n).parent.path == "LA/LB" for n in trains[2:])  # every variant shares LA/LB
assert {i["node"] for i in cal if i["kind"] == "eval"} == {"", "LA", "LA/LB"} | {n for n in trains[2:] if "purge" not in n}
la = next(i for i in cal if i["node"] == "LA" and i["kind"] == "train")
assert la["save_at"] == [2000, 5000] and next(i for i in cal if i["node"].endswith("UB-ssd@ssd_alpha=5"))["save_at"] == []
print(f"calibration plan: LA, LA/LB and {len(VARIANTS)} variants of UB sharing LA/LB; snapshots only where the standard loop trains")

# --- The report on a fake calibration run ----------------------------------------------------------
random.seed(0)
run_seed("cal", 0)
plan_path("cal").write_text(json.dumps(merge([], cal)))
def fake_eval(node, weights, seed, tasks):
    d = node.eval_dir(weights, seed); d.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        ok = [random.random() < 0.7 for _ in range(50)]
        (d / f"{t}.json").write_text(json.dumps({"success_rate": sum(ok) / 50, "episodes": [{"success": s} for s in ok]}))
def fake_train(node, steps, probe_every):
    node.dir.mkdir(parents=True, exist_ok=True)
    (node.dir / "op.json").write_text(json.dumps({"op": str(node.last_op), "notes": {"dampened_fraction": 0.01} if "ssd" in node.path else {}}))
    log = [{"step": s, "probe": {t: round(random.random(), 3) for t in "ABCDE"}, "steps_per_s": 2.0, "peak_mem_gb": 30.0, "time": 1000.0 + s}
           for s in range(0, steps + 1, probe_every)]
    (node.dir / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")
    (node.dir / "weights.safetensors").write_text("x")
for i in cal:
    node = Node.from_path("cal", i["node"])
    if i["kind"] == "train":
        fake_train(node, 10000 if node.depth < 3 else 1000, 500 if node.depth < 3 else 50)
    else:
        fake_eval(node, i["weights"], i["seed"], i["tasks"])
(RUNS_DIR / "cal" / "_failed").mkdir(parents=True)  # one failed item, with its log
(RUNS_DIR / "cal" / "_logs").mkdir(parents=True)
(RUNS_DIR / "cal" / "_logs" / "x.log").write_text("Traceback ... CUDA out of memory\n")
bad = {"kind": "train", "node": "LA/LB/UB-vlaforget@vlaforget.lr=2e-4", "save_at": []}
Node.from_path("cal", bad["node"]).weights_path.unlink()
(RUNS_DIR / "cal" / "_failed" / "train_LA~LB~UB-vlaforget@vlaforget.lr=2e-4.json").write_text(
    json.dumps({"item": bad, "exit_code": 1, "log": str(RUNS_DIR / "cal" / "_logs" / "x.log")}))
out = Path(tempfile.mkdtemp()) / "report"
subprocess.run([sys.executable, str(ROOT / "scripts" / "calibration_report.py"), "--run", "cal", "--out", str(out)], env=ENV, check=True)
summary = (out / "summary.md").read_text()
for needed in ["## Learning", "## Speed and memory per op", "## Unlearning variants", "## Stream metrics", "oracle_gap",
               "ga@unlearn_lr=3e-5", "dampened_fraction", "| failed |"]:
    assert needed in summary, needed
assert "CUDA out of memory" in next((out / "failed").iterdir()).read_text()
assert (out / "nodes" / "LA" / "train_log.jsonl").exists() and (out / "nodes" / "base" / "eval").exists()
assert not list(out.rglob("*.safetensors")), "the report must not contain weights"
print("report: summary.md with every section, failed-item logs, per-node files, no weights")

# results.py --ci --csv: every row has one field per column although cells contain commas.
import csv
table = stubs_csv = Path(tempfile.mkdtemp()) / "sr.csv"
subprocess.run([sys.executable, str(ROOT / "scripts" / "results.py"), "--run", "cal", "--ci", "--csv", str(table)],
               env=ENV, check=True, capture_output=True)
rows = list(csv.reader(open(table)))
assert len(rows) > 3 and all(len(r) == len(rows[0]) == 6 for r in rows) and "[" in rows[2][1], rows[:3]
print("results: --ci CSV has 6 well-formed columns per row")
print("ALL CALIBRATION CHECKS PASSED")
