"""Collect the main run (scripts/main_run.py) into one directory to send back: summary.md with the success
rates of every node, PLAN.md's stream metrics per method, the probe losses and cost of every unlearning op
next to its parent's and its retrain oracle's, speed and memory per op, and the drift summary
(scripts/main_drift.py); plus every node's configuration, training log, eval results, rollout videos and the
drift CSVs. Model weights are not included. Plain Python (no GPU, no torch).

    python scripts/main_report.py --run goal_s0 --out DIR
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_drift import fmt, table  # noqa: E402
from calibration_report import (copy_run_files, cost, drift_section, perf, plan_section, probes, sr,  # noqa: E402
                                stream_metrics_section, train_log)
from main_drift import input_changes  # noqa: E402
from vlu.nodes import Node, oracle_path  # noqa: E402
from vlu.tasks import TASKS  # noqa: E402
from vlu.worker import load_plan  # noqa: E402

NOTES = ("dampened_fraction", "kept_fraction", "batch_forget_retain", "ssd_reference", "forget_share", "retain_coeff")


def final_probe(node: Node) -> dict:
    """Probe losses at the node's last logged step; {} while it is not done (its log may hold only step 0,
    the parent's)."""
    p = probes(train_log(node))
    return p[max(p)] if p and node.done else {}


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="goal_s0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run, out = args.run, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Main run `{run}`", "", "Stream S1 (`LA LB LC UB LD LE UE`, PLAN.md week 1) per unlearning method, its retrain "
          "oracles, and PLAN.md section 5.1's control branch C (`LA/LB/LC/LD/LE`, B still replayed); see "
          "scripts/main_run.py. Files: `environment.txt`, `tests.txt`, `run_local.log`, `nodes/` (per node: "
          "op.json, train_log.jsonl, eval results, videos), `failed/` (logs of failed items).", ""]
    md += plan_section(run, out)
    plan = load_plan(run)
    trained = [Node.from_path(run, i["node"]) for i in plan if i["kind"] == "train"]
    unlearn = [n for n in trained if n.last_op.kind == "U"]
    methods = list(dict.fromkeys(n.last_op.method for n in unlearn))

    # --- Success rates ---------------------------------------------------------------------------
    evaluated = sorted({i["node"] for i in plan if i["kind"] == "eval"})  # path order = the tree, depth first
    rows = [[path or "base", *(sr(Node.from_path(run, path), t) for t in TASKS)] for path in evaluated]
    md += ["## Success rates", "", "Success rate [95% Wilson interval] over 50 rollouts (eval seed 0); `-` = not "
           "evaluated (a task the node has not seen). Purge nodes are not evaluated: their weights are their parent's.", "",
           *table(["node", *TASKS], rows), ""]

    # --- Stream metrics (PLAN.md) --------------------------------------------------------------
    md += stream_metrics_section(run, "S1", methods)

    # --- Unlearning ops ------------------------------------------------------------------------
    rows = []
    for node in unlearn:
        if node.last_op.method == "purge":
            continue
        forget, kept = node.last_op.task, node.retained_tasks()
        before, after = final_probe(node.parent), final_probe(node)
        oracle = final_probe(Node.from_path(run, oracle_path(run, node.path)))
        retained = lambda p: mean([p.get(t) for t in kept]) if p else None  # noqa: E731
        notes = json.loads((node.dir / "op.json").read_text()).get("notes", {}) if (node.dir / "op.json").exists() else {}
        extra = {k: notes[k] for k in NOTES if k in notes}
        rows.append([node.path, forget, fmt(before.get(forget), ".4f"), fmt(after.get(forget), ".4f"), fmt(oracle.get(forget), ".4f"),
                     ",".join(kept), fmt(retained(before), ".4f"), fmt(retained(after), ".4f"), fmt(retained(oracle), ".4f"),
                     perf(train_log(node))[1], cost(node), json.dumps(extra) if extra else ""])
    md += ["## Unlearning ops", "", "Probe loss = flow-matching loss on fixed probe frames (32 per task) at the op's "
           "end. Targets: the forget task's probe loss rises from the parent's to about the retrain oracle's (which "
           "never saw it); the retained tasks' stays at the parent's. Cost = GPU-minutes of the op (PLAN.md): the "
           "method's preparation (all of SSD, SalUn's mask) and training, without loading the model; * = resumed, "
           "the last session only.", "",
           *table(["node", "forget", "probe forget: parent", "probe forget: after", "probe forget: oracle", "retained",
                   "probe retained: parent", "probe retained: after", "probe retained: oracle", "peak mem (GB)",
                   "cost (GPU-min)", "method info"], rows), ""]

    # --- Speed and memory ----------------------------------------------------------------------
    md += ["## Speed and memory per op", "",
           *table(["node", "steps/s (median)", "peak GPU mem (GB)", "wall (min)", "cost (GPU-min)"],
                  [[n.path, *perf(train_log(n)), cost(n)] for n in trained if n.last_op.method != "purge"]), ""]

    # --- Drift metrics (scripts/main_drift.py, run by run_local.sh --main after the workers) -----------
    md += drift_section(run, out, input_changes(run), "scripts/main_drift.py")
    (out / "summary.md").write_text("\n".join(md))
    copy_run_files(run, out)
    print(f"report written to {out}/summary.md")


if __name__ == "__main__":
    main()
