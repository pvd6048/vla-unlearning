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
from vlu.drift.paths import features_path  # noqa: E402
from vlu.nodes import Node, run_seed  # noqa: E402
from vlu.worker import claim_dir, item_key, load_plan, plan_path, someone_working, status, try_claim  # noqa: E402

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
import pathlib, sys, time
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
# A winner keeps working, so its claim stays live, until every racer has decided (not for a fixed time: the
# serialized checks take ~1.9 s, and a winner that had exited would rightly be taken over).
decided = pathlib.Path(sys.argv[3])
(decided / sys.argv[2]).touch()
while len(list(decided.iterdir())) < 8 and time.time() < float(sys.argv[1]) + 60:
    time.sleep(0.05)
"""
start = str(time.time() + 6)  # time for every process to start (imports) before the race
decided = tempfile.mkdtemp(prefix="vlu_race_")
racers = [subprocess.Popen([sys.executable, "-c", RACER, start, str(i), decided], env=ENV, stdout=subprocess.PIPE, text=True)
          for i in range(8)]
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
GA = "ga@unlearn_lr=1e-6"  # a method and its rate (vlu/config.py has none for GA until it is calibrated)
ns = argparse.Namespace(run="two", nodes="", stream="ALL_GONE", methods=GA, no_oracle=False, zero_shot="", save_at="",
                        eval_weights="final,step_002000", eval_seeds="0", videos=0, no_eval=False, features=False, fisher=[])
built = submit.build_items(ns)
base_evals = [i for i in built if i["kind"] == "eval" and i["node"] == ""]
assert [i["weights"] for i in base_evals] == ["final"] and base_evals[0]["tasks"] == ["A"], base_evals
assert not any(i["kind"] == "train" and i["node"] == "" for i in built)
print("submit: a base-model oracle is evaluated (final weights only), not trained")

# --save-at and snapshot evals apply to L ops only (U ops have their own short budget; SSD/VLA-Forget save none).
submit.STREAMS["SNAP"] = ["LA", "LB", "UB"]
built = submit.build_items(argparse.Namespace(**{**vars(ns), "stream": "SNAP", "methods": f"ssd,{GA}", "save_at": "2000"}))
assert all(i["save_at"] == [] for i in built if i["kind"] == "train" and "/U" in i["node"]), built
assert {i["weights"] for i in built if i["kind"] == "eval" and "/U" in i["node"]} == {"final"}
assert any(i["save_at"] == [2000] for i in built if i["kind"] == "train" and i["node"] == "LA/LB")
assert {i["weights"] for i in built if i["kind"] == "eval" and i["node"] == "LA/LB"} == {"final", "step_002000"}
print("submit: --save-at and snapshot evals only for L ops")

# An op that cannot be applied is rejected when planning (here: SSD on the parent's only task, S3-like),
# not later in a worker job.
submit.STREAMS["ONLY"] = ["LA", "UA"]
for method, ok in (("ssd", False), ("salun", False), (GA, True)):
    try:
        submit.build_items(argparse.Namespace(**{**vars(ns), "stream": "ONLY", "methods": method}))
        assert ok, f"{method} on a parent's only task must be rejected at plan time"
    except ValueError as error:
        assert not ok and "needs a retained task" in str(error), (method, error)
print("submit: SSD / SalUn on a parent's only task rejected when planning; GA accepted")

# A method without a learning rate yet (vlu/config.py unlearn_lrs) is rejected when planning.
from vlu.config import RECIPE  # noqa: E402
uncalibrated = sorted({"ga", "graddiff", "npo", "npo_rt", "salun"} - set(dict(RECIPE.unlearn_lrs)))
for method in uncalibrated:
    try:
        submit.build_items(argparse.Namespace(**{**vars(ns), "stream": "SNAP", "methods": f"ssd,{method}"}))
        raise AssertionError(f"{method} has no learning rate: it must be rejected at plan time")
    except ValueError as error:
        assert f"{method} has no learning rate yet" in str(error), error
print(f"submit: methods without a learning rate yet ({', '.join(uncalibrated) or 'none'}) rejected when planning")

# A trained node whose op.json records other values for the fields its op depends on is reported.
from dataclasses import asdict  # noqa: E402
from vlu.config import RECIPE  # noqa: E402


def trained(path, recipe):
    node = Node.from_path("two", path)
    node.dir.mkdir(parents=True, exist_ok=True)
    node.write_json("op.json", {"op": str(node.last_op), "recipe": recipe})
    node.weights_path.write_text("x")
    return {"kind": "train", "node": path, "save_at": []}


current = json.loads(json.dumps(asdict(RECIPE)))
old_ssd = {k: v for k, v in current.items() if k != "ssd_reference"}  # trained before the field existed
items_ = [trained("LA", {**current, "save_every": 1000, "ssd_alpha": 99.0}),  # L op: neither field matters
          trained("LA/LB/UB-ssd@ssd_alpha=10", {**old_ssd, "ssd_alpha": 10.0}),
          trained("LA/LB/UB-salun", {**current, "unlearn_lr": 1e-5}),  # its rate is the method's (unlearn_lrs)
          trained("LA/LB/UB-ga@unlearn_lr=3e-7", {**current, "unlearn_lr": 3e-7}),  # the op's own rate: unchanged
          trained("LA/LB/UB-vlaforget", {**current, "vlaforget": {**current["vlaforget"], "lr": 5e-5}})]
warnings = submit.warn_recipe_changes("two", items_)
assert len(warnings) == 3 and "UB-ssd@ssd_alpha=10" in warnings[0] and "ssd_reference: '<not recorded>' -> now 'retain'" in warnings[0], warnings
assert f"UB-salun was trained with another recipe (unlearn_lr: 1e-05 -> now {dict(RECIPE.unlearn_lrs)['salun']!r})" in warnings[1], warnings
assert "vlaforget.lr: 5e-05 -> now 0.0002" in warnings[2], warnings
print("submit: trained nodes with another recipe in the fields their op uses are reported (bookkeeping and other ops' fields ignored)")

# A worker re-checks an item after claiming it: another worker may have just finished or failed it.
import vlu.worker as w  # noqa: E402
done_item = {"kind": "eval", "node": "LA", "weights": "final", "seed": 7, "tasks": ["A"], "videos": 0}
out_dir = Node.from_path("two", "LA").eval_dir("final", 7)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "A.json").write_text("{}")
failed_item = {**done_item, "seed": 8}
w.failed_file("two", failed_item).parent.mkdir(parents=True, exist_ok=True)
w.failed_file("two", failed_item).write_text("{}")
assert not w.take("two", done_item) and not w.take("two", failed_item)
assert not w.claim_dir("two", done_item).exists() and not w.claim_dir("two", failed_item).exists()
print("worker: a claimed item that turns out done or failed is released, not run again")

# Slurm: a job that has ended (still COMPLETING, or gone) does not hold its claims.
from types import SimpleNamespace  # noqa: E402
real_run = w.subprocess.run
try:
    for state, alive in [("RUNNING", True), ("PENDING", True), ("COMPLETING", False), ("TIMEOUT", False), ("", False)]:
        w.subprocess.run = lambda *a, state=state, **k: SimpleNamespace(returncode=0, stdout=state + "\n", stderr="")
        assert w._job_alive("12345") is alive, state
finally:
    w.subprocess.run = real_run
print("worker: Slurm jobs in RUNNING/PENDING hold claims; COMPLETING/ended ones do not")

# An override must name a hyperparameter, not the whole VLA-Forget config.
from vlu.nodes import Op  # noqa: E402
try:
    Op.parse("UB-vlaforget@vlaforget=1")
    raise AssertionError("'vlaforget' is not a hyperparameter")
except ValueError:
    pass
print("op tokens: '@vlaforget=...' rejected (only vlaforget.<field>)")

# env.sh in GPU render mode must clear a Mesa EGL override inherited from a mesa-mode shell.
egl = subprocess.run(["bash", "-c", 'source slurm/env.sh; echo "${__EGL_VENDOR_LIBRARY_FILENAMES-unset}"'], cwd=ROOT, text=True,
                     capture_output=True, env={**ENV, "VLU_RENDER": "gpu", "__EGL_VENDOR_LIBRARY_FILENAMES": "/x/50_mesa.json"})
assert egl.stdout.strip() == "unset", egl
print("env.sh: GPU rendering clears an inherited Mesa EGL vendor override")

# env.sh sets OMP_NUM_THREADS=1, which GNU nproc honours: the CPU count must not come from plain nproc
# (in the calibration it made every eval run 1 simulator process; on Slurm it would give 0 workers).
cpus = subprocess.run(["bash", "-c", 'source slurm/env.sh; echo "$VLU_CPUS $(nproc)"'], cwd=ROOT, text=True,
                      capture_output=True, env={**ENV, "OMP_NUM_THREADS": "1"}).stdout.split()
# The reference is nproc without the OMP variables (it also counts a cgroup CPU limit, which the
# affinity mask alone does not show).
plain = subprocess.run(["nproc"], text=True, capture_output=True, check=True,
                       env={k: v for k, v in ENV.items() if k not in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT")}).stdout.strip()
assert cpus[0] == plain and cpus[1] == "1", (cpus, plain)
workers = subprocess.run(["bash", "-c", 'for c in 1 2 3 24; do VLU_CPUS=$c; eval "$(grep -m1 "^WORKERS=" slurm/eval.sbatch)"; echo -n "$WORKERS "; done'],
                         cwd=ROOT, text=True, capture_output=True, env={k: v for k, v in ENV.items() if k != "VLU_EVAL_WORKERS"}).stdout.split()
assert workers == ["1", "1", "2", "23"], workers
print(f"env.sh: VLU_CPUS = {cpus[0]} CPUs although nproc says 1; eval.sbatch workers for 1/2/3/24 CPUs: {workers}")

# --- The calibration plan --------------------------------------------------------------------------
cal = items()
trains = [i["node"] for i in cal if i["kind"] == "train"]
assert trains[:2] == ["LA", "LA/LB"] and len(trains) == 2 + len(VARIANTS)
assert all(Node.from_path("cal", n).parent.path == "LA/LB" for n in trains[2:])  # every variant shares LA/LB
assert {i["node"] for i in cal if i["kind"] == "eval"} == {"", "LA", "LA/LB"} | {n for n in trains[2:] if "purge" not in n}
la = next(i for i in cal if i["node"] == "LA" and i["kind"] == "train")
assert la["save_at"] == [2000, 5000] and next(i for i in cal if i["node"].endswith("UB-ssd@ssd_reference=retain"))["save_at"] == []
print(f"calibration plan: LA, LA/LB and {len(VARIANTS)} variants of UB sharing LA/LB; snapshots only of LA (the evaluated ones)")
# Drift features of every node whose weights differ (purge's are LA/LB's), right after its training;
# the Fisher of A and B only at LA/LB, the parent of every variant.
feats = [i for i in cal if i["kind"] == "features"]
assert [i["node"] for i in feats] == ["", "LA", "LA/LB"] + [n for n in trains[2:] if "purge" not in n], feats
assert all(i["weights"] == "final" and i["fisher"] == (["A", "B"] if i["node"] == "LA/LB" else []) for i in feats), feats
assert all(i.get("optional") for i in feats) and not any(i.get("optional") for i in cal if i["kind"] != "features")
assert "optional" not in merge([dict(feats[0])], [{**feats[0], "optional": False}])[0], "required wins when merging"
position = {(i["kind"], i["node"]): k for k, i in enumerate(cal)}
assert all(position[("features", n)] > position[("train", n)] for n in trains if ("features", n) in position)
print(f"calibration plan: drift features of {len(feats)} nodes after their training, optional; Fisher of A and B at LA/LB")

# calibrate.py writes exactly the current calibration: items of an earlier round that are no longer
# planned are dropped (their node directories stay), so they cannot run again under new code.
prune_env = {**ENV, "VLU_RUNS_DIR": tempfile.mkdtemp(prefix="vlu_prune_")}
stale = {"kind": "train", "node": "LA/LB/UB-ssd@ssd_alpha=20", "save_at": []}
(Path(prune_env["VLU_RUNS_DIR"]) / "cal").mkdir(parents=True)
(Path(prune_env["VLU_RUNS_DIR"]) / "cal" / "plan.json").write_text(json.dumps([stale, cal[1]]))
out_p = subprocess.run([sys.executable, str(ROOT / "scripts" / "calibrate.py")], env=prune_env, text=True, capture_output=True, check=True).stdout
written = json.loads((Path(prune_env["VLU_RUNS_DIR"]) / "cal" / "plan.json").read_text())
assert "no longer planned (kept on disk, not run again): train:LA/LB/UB-ssd@ssd_alpha=20" in out_p, out_p
assert stale not in written and len(written) == len(cal), (len(written), len(cal))
print("calibrate: items of an earlier round no longer planned are dropped from the plan")

# --- run_local.sh --drift-only: features-only workers on the calibration plan -------------------------
# Of the models trained so far (here base, LA, LA/LB and two variants) the features and LA/LB's Fisher are
# extracted; untrained variants are left waiting; no training or eval runs; the workers exit 0 (features
# items are optional), so the drift step can run on what exists.
FEATURES_STUB = r'''
import sys, time
import vlu.worker as w
from vlu.drift.paths import features_path, fisher_path
from vlu.nodes import Node
def command(run, item):
    assert item["kind"] == "features", item
    node = Node.from_path(run, item["node"])
    files = [features_path(node, item["weights"])] + [fisher_path(node, t, item["weights"]) for t in item.get("fisher", [])]
    return [sys.executable, "-c", "import pathlib, sys, time; time.sleep(0.5)\nfor f in sys.argv[1:]:\n p = pathlib.Path(f); p.parent.mkdir(parents=True, exist_ok=True); p.write_text('x')", *map(str, files)]
w.command = command
real_sleep = time.sleep
w.time.sleep = lambda s: real_sleep(0.2)
sys.argv = ["worker", "--run", "cal", "--kinds", "features", "--wait"]
w.main()
'''
drift_env = {**ENV, "VLU_RUNS_DIR": tempfile.mkdtemp(prefix="vlu_drift_only_")}
subprocess.run([sys.executable, str(ROOT / "scripts" / "calibrate.py")], env=drift_env, check=True, capture_output=True)
trained_now = ["LA", "LA/LB", f"LA/LB/UB-{VARIANTS[1]}", f"LA/LB/UB-{VARIANTS[2]}"]
for path in trained_now:
    d = Path(drift_env["VLU_RUNS_DIR"]) / "cal" / path
    d.mkdir(parents=True, exist_ok=True)
    (d / "weights.safetensors").write_text("x")
workers = [subprocess.Popen([sys.executable, "-c", FEATURES_STUB], env=drift_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
           for _ in range(2)]
outs = [p.communicate(timeout=120)[0] for p in workers]
assert all(p.returncode == 0 for p in workers), outs
log = "".join(outs)
check = subprocess.run([sys.executable, "-c", f"""
import json
from vlu.drift.paths import features_path, fisher_path
from vlu.nodes import Node
done = [p for p in ["", "LA", "LA/LB"] + ["LA/LB/UB-" + v for v in {VARIANTS!r}] if features_path(Node.from_path("cal", p)).exists()]
fisher = [t for t in "AB" if fisher_path(Node.from_path("cal", "LA/LB"), t).exists()]
print(json.dumps([done, fisher]))"""], env=drift_env, capture_output=True, text=True, check=True)
done, fisher = json.loads(check.stdout)
assert done == [""] + trained_now and fisher == ["A", "B"], (done, fisher)
assert "start train" not in log and "start eval" not in log and log.count("start features") == len(trained_now) + 1, log
assert "optional item(s) not done" in log, log  # the untrained variants' features, reported
print(f"drift only: features-only workers extracted the {len(done)} trained models' features and LA/LB's Fisher, "
      "trained and evaluated nothing, and exited 0")

# --drift-only retries only failed features items: a failed training stays recorded (and reported as failed).
failed_dir = Path(drift_env["VLU_RUNS_DIR"]) / "cal" / "_failed"
failed_dir.mkdir(parents=True, exist_ok=True)
bad_train = {"kind": "train", "node": f"LA/LB/UB-{VARIANTS[3]}", "save_at": []}
bad_features = {"kind": "features", "node": f"LA/LB/UB-{VARIANTS[3]}", "weights": "final", "fisher": [], "optional": True}
for item in (bad_train, bad_features):
    name = item_key(item).replace("<base>", "base").replace("/", "~").replace(":", "_")
    (failed_dir / f"{name}.json").write_text(json.dumps({"item": item, "exit_code": 1, "log": "x"}))
subprocess.run([sys.executable, str(ROOT / "scripts" / "submit.py"), "--run", "cal", "--retry-failed", "--retry-kinds", "features",
                "--workers", "none"], env=drift_env, check=True, capture_output=True)
left = sorted(f.name for f in failed_dir.iterdir())
assert left == [f"train_LA~LB~UB-{VARIANTS[3]}.json"], left
print("submit: --retry-failed --retry-kinds features forgets only the features failures")

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
    elif i["kind"] == "eval":
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
assert "## Drift metrics\n\nNot computed" in summary and not (out / "drift").exists()
print("report: summary.md with every section, failed-item logs, per-node files, no weights")

# With a drift report (scripts/calibration_drift.py, tested in tests/test_calibration_drift.py) its
# summary is part of summary.md and its files are packed; only its own: drift/ and failed/ of an
# earlier pack are emptied. A report computed from other files than the run now has is marked stale.
import calibration_drift  # noqa: E402
(out / "drift").mkdir()
(out / "drift" / "representation.csv").write_text("from an earlier pack\n")
(out / "failed" / "train_LA.log").write_text("a failure of an earlier pack, rerun since\n")
drift = RUNS_DIR / "cal" / "drift_report"
drift.mkdir()
(drift / "params.log").write_text("Traceback ...\n")
report = lambda: subprocess.run([sys.executable, str(ROOT / "scripts" / "calibration_report.py"), "--run", "cal", "--out", str(out)],  # noqa: E731
                                env=ENV, check=True)
report()
summary = (out / "summary.md").read_text()
assert "Not computed" in summary and (out / "drift" / "params.log").exists(), "logs of a drift step that stopped early"
assert not (out / "drift" / "representation.csv").exists() and not (out / "failed" / "train_LA.log").exists()
(drift / "summary.md").write_text("## Drift metrics (scripts/calibration_drift.py)\n\n| drift | x |\n")
(drift / "params.csv").write_text("level,group,metric,drift_i,drift_j,value\n")
(drift / "inputs.json").write_text(json.dumps(calibration_drift.inputs("cal")))
report()
summary = (out / "summary.md").read_text()
assert "## Drift metrics (scripts/calibration_drift.py)\n\n| drift | x |" in summary and "Not computed" not in summary
assert "Stale" not in summary and (out / "drift" / "params.csv").exists()
used = json.loads((drift / "inputs.json").read_text())
# LA, LA/LB and the variants' weights, but purge's (its weights are LA/LB's) and the failed one's
assert "LA/LB/weights.safetensors" in used and len(used) == 2 + len(VARIANTS) - 2, sorted(used)
retrained = Node.from_path("cal", "LA/LB/UB-ga@unlearn_lr=1e-7").weights_path
os.utime(retrained, ns=(0, 0))  # rewritten since the report
Node.from_path("cal", "LA/LB/UB-ga@unlearn_lr=3e-7").weights_path.unlink()
features = features_path(Node.from_path("cal", "LA"))
features.parent.mkdir()
features.write_text("x")  # extracted since
report()
stale = next(line for line in (out / "summary.md").read_text().splitlines() if "Stale" in line)
assert ("changed or removed since: LA/LB/UB-ga@unlearn_lr=1e-7/weights.safetensors, LA/LB/UB-ga@unlearn_lr=3e-7/weights.safetensors; "
        f"new since: LA/drift/{features.name}.") in stale, stale
print("report: the drift summary goes into summary.md, its files into drift/ (logs also without a summary), "
      "nothing of an earlier pack; a stale drift report is marked")

# results.py --ci --csv: every row has one field per column although cells contain commas.
import csv
table = stubs_csv = Path(tempfile.mkdtemp()) / "sr.csv"
subprocess.run([sys.executable, str(ROOT / "scripts" / "results.py"), "--run", "cal", "--ci", "--csv", str(table)],
               env=ENV, check=True, capture_output=True)
rows = list(csv.reader(open(table)))
assert len(rows) > 3 and all(len(r) == len(rows[0]) == 6 for r in rows) and "[" in rows[2][1], rows[:3]
print("results: --ci CSV has 6 well-formed columns per row")
print("ALL CALIBRATION CHECKS PASSED")
