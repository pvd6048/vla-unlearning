"""Base-level checks of the main run (scripts/main_run.py, scripts/main_report.py, run by
scripts/run_local.sh --main): the plan of stream S1 with its oracles and control branch, adding methods
later, refusing methods without a learning rate, and the report on a fake run with known numbers.
Plain Python (no torch, no GPU).
    python tests/test_main_run.py
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["VLU_RUNS_DIR"] = tempfile.mkdtemp(prefix="vlu_main_")  # before importing vlu
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import main_run  # noqa: E402
from submit import merge  # noqa: E402
from vlu.config import RECIPE, RUNS_DIR  # noqa: E402
from vlu.nodes import Node  # noqa: E402
from vlu.worker import item_key, load_plan  # noqa: E402

ENV = {**os.environ, "PYTHONPATH": str(ROOT)}
METHODS = main_run.METHODS

# --- The plan --------------------------------------------------------------------------------------
items = main_run.items("goal_s0")
plan = merge([], [dict(i) for i in items])
trains = [i["node"] for i in plan if i["kind"] == "train"]
tails = {m: [f"LA/LB/LC/UB-{m}", f"LA/LB/LC/UB-{m}/LD", f"LA/LB/LC/UB-{m}/LD/LE", f"LA/LB/LC/UB-{m}/LD/LE/UE-{m}"] for m in METHODS}
expected = {"LA", "LA/LB", "LA/LB/LC", "LA/LC", "LA/LC/LD", "LA/LB/LC/LD", "LA/LB/LC/LD/LE", *sum(tails.values(), [])}
assert len(trains) == len(set(trains)) and set(trains) == expected, sorted(set(trains) ^ expected)
assert all(i["save_at"] == [] for i in plan if i["kind"] == "train")
position = {(i["kind"], i["node"]): k for k, i in enumerate(plan)}
assert all(position[("train", Node.from_path("goal_s0", n).parent.path)] < position[("train", n)] for n in trains if "/" in n)
print(f"plan: prefix LA/LB/LC, oracles LA/LC and LA/LC/LD, control LA/LB/LC/LD/LE, tails of {', '.join(METHODS)}; parents first")

# Every node is evaluated (seed 0, final weights, videos) on the tasks it has seen, an oracle on those of the
# nodes it stands in for; purge nodes are not (their weights are the parent's).
evals = {i["node"]: i for i in plan if i["kind"] == "eval"}
assert set(evals) == {n for n in trains if Node.from_path("goal_s0", n).last_op.method != "purge"}, sorted(evals)
assert all((i["weights"], i["seed"], i["videos"]) == ("final", 0, main_run.VIDEOS) for i in evals.values())
assert evals["LA/LC"]["tasks"] == ["A", "B", "C"] and evals["LA/LC/LD"]["tasks"] == ["A", "B", "C", "D", "E"]
for n, i in evals.items():
    if n not in ("LA/LC", "LA/LC/LD"):
        assert i["tasks"] == sorted(Node.from_path("goal_s0", n).seen_tasks()), (n, i["tasks"])
    assert position[("eval", n)] > position[("train", n)]
print("plan: evals of every node but purge's, on its seen tasks (oracles: A-C and A-E), after its training")

# Drift features, optional, only of the nodes scripts/main_drift.py measures drifts between (each extraction
# loads the model on a training GPU); the Fisher of A and B only at LA/LB/LC.
import main_drift  # noqa: E402
feats = {i["node"]: i for i in plan if i["kind"] == "features"}
assert set(feats) == set(main_drift.endpoints()) and all(i.get("optional") for i in feats.values()), sorted(feats)
assert "" not in feats and "LA/LC/LD" not in feats and not any("purge" in n.split("/")[-1] for n in feats)
assert all(i["fisher"] == (["A", "B"] if n == "LA/LB/LC" else []) for n, i in feats.items()), feats
assert not any(i.get("optional") for i in plan if i["kind"] != "features")
assert all(position[("features", n)] > position[("train", n)] for n in feats)
print(f"plan: drift features of the {len(feats)} nodes the drift metrics use (optional), Fisher of A and B at LA/LB/LC")

# Adding methods later (after calibration round 3) keeps the plan's items and their order.
real = main_run.METHODS
main_run.METHODS = METHODS[:1]
first = merge([], main_run.items("goal_s0"))
main_run.METHODS = real
later = merge([dict(i) for i in first], main_run.items("goal_s0"))
assert [item_key(i) for i in later[:len(first)]] == [item_key(i) for i in first] and len(later) == len(plan)
print(f"plan: adding methods to {METHODS[:1]} appends their items, the earlier ones keep their place")

# A method without a learning rate yet is refused when planning (vlu/config.py unlearn_lrs).
uncalibrated = sorted({"ga", "graddiff", "npo", "npo_rt", "salun"} - set(dict(RECIPE.unlearn_lrs)))
assert not set(METHODS) & set(uncalibrated), "main_run.METHODS must only hold calibrated methods"
if uncalibrated:
    script = f"import sys; sys.path.insert(0, 'scripts'); import main_run; main_run.METHODS.append({uncalibrated[0]!r}); main_run.main()"
    out = subprocess.run([sys.executable, "-c", script, "--run", "refused"], cwd=ROOT, env=ENV, text=True, capture_output=True)
    assert out.returncode != 0 and f"{uncalibrated[0]} has no learning rate yet" in out.stderr, out.stderr
    assert not (RUNS_DIR / "refused" / "plan.json").exists()
    print(f"main_run: a method without a learning rate ({uncalibrated[0]}) is refused, nothing is planned")

# main_run.py writes the plan (run seed 0); rerunning it changes nothing.
subprocess.run([sys.executable, str(ROOT / "scripts" / "main_run.py")], env=ENV, check=True, capture_output=True)
written = load_plan("goal_s0")
assert [item_key(i) for i in written] == [item_key(i) for i in plan]
assert json.loads((RUNS_DIR / "goal_s0" / "run.json").read_text())["seed"] == 0
subprocess.run([sys.executable, str(ROOT / "scripts" / "main_run.py")], env=ENV, check=True, capture_output=True)
assert load_plan("goal_s0") == written
print("main_run: writes the plan of run goal_s0 (seed 0); a rerun leaves it as it is")

# --- The report on a fake run, with known probe losses ----------------------------------------------
TASKS = "ABCDE"


def probe_value(path: str, task: str) -> float:
    """A distinct, known final probe loss per node and task."""
    return round(0.01 * (1 + path.count("/")) + 0.001 * TASKS.index(task) + (0.5 if "U" + task in path else 0), 4)


for i in plan:
    node = Node.from_path("goal_s0", i["node"])
    if i["kind"] == "train":
        u = node.last_op.kind == "U"
        node.dir.mkdir(parents=True, exist_ok=True)
        (node.dir / "op.json").write_text(json.dumps({"op": str(node.last_op), "notes": {"kept_fraction": 0.5} if "salun" in node.path else {}}))
        steps, every = (1000, 50) if u else (10000, 500)
        log = [{"step": s, "probe": {t: (probe_value(node.path, t) if s == steps else 9.0) for t in TASKS},
                "steps_per_s": 1.0, "peak_mem_gb": 35.0 if u else 33.4, "time": 1000.0 + 6 * s} for s in range(0, steps + 1, every)]
        (node.dir / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")
        (node.dir / "weights.safetensors").write_text("x")
        (node.dir / "cost.json").write_text(json.dumps({"seconds": 90.0 if u else 6000.0, "complete": node.path != "LA/LB"}))
    elif i["kind"] == "eval":
        d = node.eval_dir("final", 0)
        d.mkdir(parents=True, exist_ok=True)
        for t in i["tasks"]:
            ok = [k < (5 if "U" + t in node.path else 45) for k in range(50)]
            (d / f"{t}.json").write_text(json.dumps({"success_rate": sum(ok) / 50, "episodes": [{"success": s} for s in ok]}))
out = Path(tempfile.mkdtemp()) / "report"
subprocess.run([sys.executable, str(ROOT / "scripts" / "main_report.py"), "--run", "goal_s0", "--out", str(out)], env=ENV, check=True)
summary = (out / "summary.md").read_text()
for needed in ["## Plan status", "## Success rates", "## Stream metrics (scripts/metrics.py --stream S1)", "oracle_gap",
               "## Unlearning ops", "## Speed and memory per op"]:
    assert needed in summary, needed
success = summary.split("## Success rates")[1].split("## Stream metrics")[0]
rows = {line.split(" | ")[0].lstrip("| "): line for line in success.splitlines() if line.startswith("| LA")}
assert rows["LA"].count("| - ") == 4 and "0.90 [" in rows["LA"], rows["LA"]  # A only
assert rows["LA/LC/LD"].count("0.90 [") == 5, rows["LA/LC/LD"]  # the oracle of UE: every task
# UB-ssd: forget B (parent LA/LB/LC, oracle LA/LC), retained A and C (means of the two tasks)
ub = f"LA/LB/LC/UB-{METHODS[1]}"
parent, oracle = "LA/LB/LC", "LA/LC"
mean = lambda path: (probe_value(path, "A") + probe_value(path, "C")) / 2  # noqa: E731
expect = (f"| {ub} | B | {probe_value(parent, 'B'):.4f} | {probe_value(ub, 'B'):.4f} | {probe_value(oracle, 'B'):.4f} | A,C | "
          f"{mean(parent):.4f} | {mean(ub):.4f} | {mean(oracle):.4f} | 35.0 | 1.5 |")
assert expect in summary, (expect, [r for r in summary.splitlines() if ub + " |" in r])
ue = "LA/LB/LC/UB-salun/LD/LE/UE-salun"
assert f"| {ue} | E | {probe_value(ue.rsplit('/', 1)[0], 'E'):.4f} | {probe_value(ue, 'E'):.4f} | {probe_value('LA/LC/LD', 'E'):.4f} | A,C,D |" in summary
assert "| LA/LB | 1.00 | 33.4 | 1000 | 100.0* |" in summary  # speed table: cost, * = a lower bound
assert "## Drift metrics\n\nNot computed (scripts/main_drift.py has not run" in summary
assert '{"kept_fraction": 0.5}' in summary and "UB-purge |" not in summary.split("## Unlearning ops")[1].split("## Speed")[0]
print("report: success rates per node, stream metrics, unlearning ops' probe losses vs. parent and oracle and cost, "
      "speed, drift section")
for method in METHODS:
    line = next(line for line in summary.splitlines() if line.startswith(method + " "))
    assert "nan" not in line.split()[1:4], line  # retain SR, retain AUC, NBT: every eval they need exists
assert (out / "nodes" / "LA" / "LB" / "LC" / "train_log.jsonl").exists() and (out / "plan.json").exists()
assert (out / "failed").is_dir() and not list(out.rglob("*.safetensors")), "no weights in the report"
print("report: stream metrics of every method computed; per-node files packed, no weights")

# Method info, shared by both reports: the method's own numbers, including those of the methods added later
# (ASSD's alpha, the layers VLA-Forget selected).
from calibration_report import method_info  # noqa: E402
vf = Node.from_path("info", "LA/LB/UB-vlaforget")
vf.dir.mkdir(parents=True)
(vf.dir / "op.json").write_text(json.dumps({"notes": {"assd_alpha": 15.0, "assd_percentile": 96.2, "kept_fraction": 0.5, "lr": 1}}))
(vf.dir / "vlaforget.json").write_text(json.dumps({"stages": {"vision": {"selected": ["v3"]}},
                                                   "merge": {"retained_delta_fraction": {"A": 0.0123, "C": 0.002}}}))
assert json.loads(method_info(vf)) == {"kept_fraction": 0.5, "assd_alpha": 15.0, "assd_percentile": 96.2,
                                       "selected": {"vision": ["v3"]}, "min_retained_delta": 0.002}, method_info(vf)
assert method_info(Node.from_path("info", "LA")) == ""
print("report: method info with ASSD's alpha and VLA-Forget's selected layers, as in the calibration report")

# --- run_local.sh runs every test file ---------------------------------------------------------------
listed = (ROOT / "scripts" / "run_local.sh").read_text()
missing = [p.name for p in sorted((ROOT / "tests").glob("test_*.py")) if f"tests/{p.name}" not in listed]
assert not missing, f"tests not run by scripts/run_local.sh: {missing}"
print("run_local.sh: runs every tests/test_*.py")
print("ALL MAIN RUN CHECKS PASSED")
