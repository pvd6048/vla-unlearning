"""Collect a calibration run (scripts/calibrate.py) into one directory to send back: summary.md with
the tables needed to set the hyperparameters, plus every node's configuration, training log, eval
results and the few rollout videos. Model weights are not included. Plain Python (no GPU, no torch).

    python scripts/calibration_report.py --run cal --out DIR
"""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import VARIANTS  # noqa: E402
from vlu.config import PROJECT_DIR, RUNS_DIR  # noqa: E402
from vlu.nodes import Node  # noqa: E402
from vlu.stream_metrics import wilson_interval  # noqa: E402
from vlu.worker import failed_file, file_name, item_key, load_plan, status  # noqa: E402

COPY = ["op.json", "train_log.jsonl", "vlaforget.json", "vlaforget_boundary.json"]


def eval_result(node: Node, task: str, weights: str = "final", seed: int = 0) -> dict | None:
    path = node.eval_dir(weights, seed) / f"{task}.json"
    return json.loads(path.read_text()) if path.exists() else None


def sr(node: Node, task: str, weights: str = "final", seed: int = 0) -> str:
    """'SR [95% CI]' or '-'."""
    r = eval_result(node, task, weights, seed)
    if r is None:
        return "-"
    lo, hi = wilson_interval(sum(e["success"] for e in r["episodes"]), len(r["episodes"]))
    return f"{r['success_rate']:.2f} [{lo:.2f},{hi:.2f}]"


def train_log(node: Node) -> list[dict]:
    """The node's log records; a line still being written (report made during the run) is skipped."""
    path = node.dir / "train_log.jsonl"
    records = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    return records


def probes(log: list[dict]) -> dict[int, dict]:
    """{step: probe losses}; the last record wins (VLA-Forget logs the merged model at its final step)."""
    return {r["step"]: r["probe"] for r in log if "probe" in r}


def perf(log: list[dict]) -> tuple[str, str, str]:
    speeds = [r["steps_per_s"] for r in log if "steps_per_s" in r]
    mem = [r["peak_mem_gb"] for r in log if "peak_mem_gb" in r]
    times = [r["time"] for r in log if "time" in r]
    return (f"{statistics.median(speeds):.2f}" if speeds else "-", f"{max(mem):.1f}" if mem else "-",
            f"{(times[-1] - times[0]) / 60:.0f}" if len(times) > 1 else "-")


def fmt(x) -> str:
    return "-" if x is None else f"{x:.4f}"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="cal")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run, out = args.run, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base, la, lalb = (Node.from_path(run, p) for p in ("", "LA", "LA/LB"))
    md = [f"# Calibration run `{run}`", "", "See CALIBRATION.md for what was run. Files: `environment.txt`, "
          "`tests.txt`, `run_local.log`, `nodes/` (per node: op.json, train_log.jsonl, eval results, videos), "
          "`failed/` (logs of failed items).", ""]

    # --- Status of every planned item ---------------------------------------------------------
    plan = load_plan(run)
    states = [(item_key(i), status(run, i)) for i in plan]
    md += ["## Plan status", "", table(["item", "status"], states), ""]
    (out / "failed").mkdir(exist_ok=True)
    for item in plan:
        if failed_file(run, item).exists():
            info = json.loads(failed_file(run, item).read_text())
            log = Path(info["log"])
            tail = "\n".join(log.read_text().splitlines()[-300:]) if log.exists() else "(log missing)"
            name = file_name(item)
            (out / "failed" / f"{name}.log").write_text(json.dumps(info, indent=1) + "\n\n" + tail + "\n")

    # --- Learning ------------------------------------------------------------------------------
    rows = [["base (zero-shot)", "final", sr(base, "A"), "-", sr(base, "B")]]
    for weights in ("step_002000", "step_005000", "final"):
        rows.append(["LA", weights, sr(la, "A", weights, 0), sr(la, "A", weights, 1), sr(la, "B", weights, 0)])
    rows.append(["LA/LB", "final", sr(lalb, "A", "final", 0), sr(lalb, "A", "final", 1), sr(lalb, "B", "final", 0)])
    md += ["## Learning", "", "Success rate [95% Wilson interval] over 50 rollouts. Pass bar (PLAN.md): A >= 0.80 "
           "after 10k steps. Seed 0 vs seed 1 = eval noise.", "",
           table(["node", "weights", "A (eval seed 0)", "A (eval seed 1)", "B (eval seed 0)"], rows), "",
           table(["node", "LA/LB: B eval seed 1"], [["LA/LB", sr(lalb, "B", "final", 1)]]), ""]

    # --- Speed and memory ------------------------------------------------------------------------
    nodes = [la, lalb] + [Node.from_path(run, f"LA/LB/UB-{v}") for v in VARIANTS]
    md += ["## Speed and memory per op", "", table(["node", "steps/s (median)", "peak GPU mem (GB)", "wall (min)"],
                                                      [[n.path, *perf(train_log(n))] for n in nodes]), ""]

    # --- Unlearning variants -------------------------------------------------------------------
    theta0 = probes(train_log(lalb))
    oracle = probes(train_log(la))
    theta0_final = theta0[max(theta0)] if theta0 else {}
    oracle_final = oracle[max(oracle)] if oracle else {}
    rows = []
    for v in VARIANTS:
        node = Node.from_path(run, f"LA/LB/UB-{v}")
        p = probes(train_log(node))
        final = p[max(p)] if p else {}
        at = lambda step, task: fmt(p.get(step, {}).get(task))  # noqa: E731
        results_node = lalb if v == "purge" else node
        notes = json.loads((node.dir / "op.json").read_text()).get("notes", {}) if (node.dir / "op.json").exists() else {}
        extra = {k: notes[k] for k in ("dampened_fraction", "kept_fraction", "batch_forget_retain") if k in notes}
        if (node.dir / "vlaforget.json").exists():
            vf = json.loads((node.dir / "vlaforget.json").read_text())
            extra["selected"] = {s: d["selected"] for s, d in vf["stages"].items()}
            extra["min_retained_delta"] = round(min(vf["merge"]["retained_delta_fraction"].values()), 3)
        rows.append([v, sr(results_node, "A"), sr(results_node, "B"), at(250, "B"), at(500, "B"), fmt(final.get("B")),
                     fmt(final.get("A")), *perf(train_log(node))[1:], json.dumps(extra) if extra else ""])
    md += ["## Unlearning variants (forget B, retain A; parent LA/LB, retrain oracle LA)", "",
           "Probe loss = flow-matching loss on fixed probe frames (32 per task). Targets: B's probe loss should rise "
           f"to about the oracle's (LA never saw B): oracle B = {fmt(oracle_final.get('B'))}, A = "
           f"{fmt(oracle_final.get('A'))}; before unlearning (LA/LB): B = {fmt(theta0_final.get('B'))}, "
           f"A = {fmt(theta0_final.get('A'))}. SR B should fall to LA's SR on B; SR A should stay at LA/LB's.", "",
           table(["variant", "SR A (retain)", "SR B (forget)", "probe B @250", "probe B @500", "probe B final",
                  "probe A final", "peak mem (GB)", "wall (min)", "method info"], rows), ""]

    # --- Stream metrics (PLAN.md) --------------------------------------------------------------
    metrics = subprocess.run([sys.executable, str(PROJECT_DIR / "scripts" / "metrics.py"), "--stream", "CAL",
                              "--methods", ",".join(VARIANTS), "--runs", run], capture_output=True, text=True)
    md += ["## Stream metrics (scripts/metrics.py --stream CAL)", "", "```", metrics.stdout + metrics.stderr, "```", ""]
    (out / "summary.md").write_text("\n".join(md))

    # --- Raw files of every node (no weights) --------------------------------------------------
    for node_dir in [RUNS_DIR / run] + [d for d in (RUNS_DIR / run).rglob("*") if d.is_dir() and (d / "op.json").exists()]:
        rel = node_dir.relative_to(RUNS_DIR / run)
        dest = out / "nodes" / (str(rel) if str(rel) != "." else "base")
        for name in COPY:
            if (node_dir / name).exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(node_dir / name, dest / name)
        if (node_dir / "eval").exists():
            shutil.copytree(node_dir / "eval", dest / "eval", dirs_exist_ok=True)
    if (RUNS_DIR / run / "_logs").exists():  # output of every training / eval item
        shutil.copytree(RUNS_DIR / run / "_logs", out / "item_logs", dirs_exist_ok=True)
    for name in ("plan.json", "run.json"):
        if (RUNS_DIR / run / name).exists():
            shutil.copy2(RUNS_DIR / run / name, out / name)
    print(f"report written to {out}/summary.md")


if __name__ == "__main__":
    main()
