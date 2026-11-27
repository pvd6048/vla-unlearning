"""Collect a calibration run (scripts/calibrate.py) into one directory to send back: summary.md with
the tables needed to set the hyperparameters and the drift summary (scripts/calibration_drift.py), plus
every node's configuration, training log, eval results, drift CSVs and the few rollout videos. Model weights are not included. Plain Python (no GPU, no torch).

    python scripts/calibration_report.py --run cal --out DIR
"""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import VARIANTS  # noqa: E402
from calibration_drift import fmt, input_changes, report_dir, table  # noqa: E402
from submit import warn_recipe_changes  # noqa: E402
from vlu.config import PROJECT_DIR, RUNS_DIR  # noqa: E402
from vlu.nodes import Node  # noqa: E402
from vlu.stream_metrics import wilson_interval  # noqa: E402
from vlu.worker import failed_file, file_name, item_key, load_plan, status  # noqa: E402

COPY = ["op.json", "train_log.jsonl", "cost.json", "cost_sessions.jsonl", "vlaforget.json", "vlaforget_boundary.json"]


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


# What an op's method reports about itself (op.json notes), shown in the reports' tables.
NOTES = ("dampened_fraction", "kept_fraction", "batch_forget_retain", "ssd_reference", "forget_share", "assd_alpha",
         "assd_percentile", "retain_coeff", "implementation_variant", "actual_forget_fraction",
         "task_arithmetic_scale", "auxiliary_fit_steps", "rmu_layer", "rmu_alpha", "rmu_steering")


def method_info(node: Node) -> str:
    """JSON of the method's own numbers: SSD's dampened fraction, SalUn's kept fraction, ASSD's alpha, the
    layers VLA-Forget selected, ...; '' if none."""
    notes = json.loads((node.dir / "op.json").read_text()).get("notes", {}) if (node.dir / "op.json").exists() else {}
    extra = {k: notes[k] for k in NOTES if k in notes}
    for record in reversed(train_log(node)):
        if record.get("stage") == "finalize":
            extra.update({k: record[k] for k in ("task_vector_norm", "task_edit_norm") if k in record})
            break
    if (node.dir / "vlaforget.json").exists():
        vf = json.loads((node.dir / "vlaforget.json").read_text())
        extra["selected"] = {s: d["selected"] for s, d in vf["stages"].items()}
        extra["min_retained_delta"] = round(min(vf["merge"]["retained_delta_fraction"].values()), 3)
    return json.dumps(extra) if extra else ""


def cost(node: Node) -> str:
    """Method GPU-minutes over all sessions (legacy session time if detailed costs are absent); '*' = a
    lower bound (a session was killed before it could record its time)."""
    path = node.dir / "cost.json"
    if not path.exists():
        return "-"
    c = json.loads(path.read_text())
    efficiency = c.get("efficiency", {})
    method = efficiency.get("method")
    seconds = method["seconds"] if method else c["seconds"]
    complete = c["complete"] and not efficiency.get("unmeasured_sessions")
    return f"{seconds / 60:.1f}" + ("" if complete else "*")


EFFICIENCY_NOTE = (
    "The cost of each op, measured by vlu/cost.py in every session of the op (its docstring has the details). "
    "**Method** = its preparation (all of SSD; SalUn's mask; NPO's reference copy) plus its training (VLA-Forget: "
    "also its boundary set, layer selection and merge), without our probe losses, checkpoints and final save, and "
    "without loading the model; **data wait** = the part of it spent waiting for batches (SSD's and SalUn's "
    "importance batches and VLA-Forget's selection batches are built on the main process; training batches by "
    "DataLoader workers), a cost of the CPUs rather than of the method. **Compute** = FLOPs counted by torch's FLOP counter (matrix multiplications, "
    "convolutions, PyTorch attention; forward and backward) on the first call of each repeated unit (a step, an "
    "importance or selection batch) times its calls: a lower bound (flash-attn's LLM attention and elementwise "
    "work are not counted), the same for every method, so comparable across methods and GPUs. **Energy** = the "
    "GPU's only (no CPU, no DRAM); source: zeus = ML.energy's Zeus, nvml = the same NVML energy counter, both "
    "measured on a full GPU; board = a MIG slice: the board's counter x the slice's share (1 when the slice is the "
    "whole board, e.g. 4g.96gb); power_limit = a pseudo-estimate (power limit x time x share), an upper bound. "
    "**Above idle** = energy - idle power x time, with the idle power measured after loading (the part the method's "
    "work adds; a lower bound if the board was not yet in its lowest power state). **Peak** = PyTorch's allocated / "
    "reserved GPU memory in preparation and training (the CUDA context, ~0.5 GiB, comes on top). * = a lower "
    "bound (a session was killed before it recorded). Times and energies compare across methods on the same GPU only.")
STEPS_NOTE = (
    "Steps and batches per op: train = training steps (VLA-Forget: train:<stage>), importance = SSD's / SalUn's "
    "importance batches, selection:<stage> = VLA-Forget's layer-selection batches. Step = its compute once its "
    "batch has arrived (CUDA synchronized), batch wait = how long it waited for that batch (a wait near the step "
    "time means the CPUs limit the op, not the method); both over the steps after the first 5 of each kind and "
    "session, without the one whose FLOPs were counted. TFLOP / step: as Compute above; achieved = TFLOP per step "
    "/ median step time.")


def efficiency_of(node: Node) -> tuple[dict, bool] | None:
    """(the op's cost.json "efficiency", whether its sums are complete) or None if not recorded."""
    path = node.dir / "cost.json"
    c = json.loads(path.read_text()) if path.exists() else {}
    e = c.get("efficiency") or {}
    if not e.get("method"):
        return None
    return e, c.get("complete", True) and not e.get("unmeasured_sessions")


def num(value, spec: str, scale: float = 1.0, mark: str = "") -> str:
    return "-" if value is None else f"{value / scale:{spec}}{mark}"


def efficiency_tables(nodes: list[Node], out: Path) -> list[str]:
    """Markdown: the cost of every op and of its training steps, the energy notes; writes out/efficiency.csv,
    out/efficiency_steps.csv and out/efficiency_phases.csv (every number, for the paper's own tables)."""
    rows, step_rows, notes = [], [], set()
    csv_ops, csv_steps, csv_phases = [], [], []
    for node in nodes:
        if node.last_op.shares_parent_weights:  # purge: no op to measure
            continue
        found = efficiency_of(node)
        if found is None:
            rows.append([node.path, *["-"] * 11])
            continue
        e, complete = found
        m, mark = e["method"], "" if complete else "*"
        prepare = e["phases"].get("prepare", {}).get("seconds", 0.0)
        rows.append([node.path, ", ".join(e["gpus"]), num(m["seconds"], ".1f", 60, mark), num(prepare, ".1f", 60),
                     num(m.get("data_wait_s"), ".1f", 60),
                     num(m["flops"], ".1f", 1e15, mark), num(m["energy_j"], ".1f", 3600, mark),
                     num(m["energy_above_idle_j"], ".1f", 3600, mark), num(m["mean_power_w"], ".0f"),
                     f"{m['peak_allocated_gb']:.1f} / {m['peak_reserved_gb']:.1f}", e["energy_source"],
                     "; ".join(e["flags"]) or "-"])
        notes.update(f"{', '.join(e['gpus'])}: {n}" for n in e["energy_notes"])
        notes.update(f"{node.path}: FLOPs not counted: {n}" for n in e.get("flops_notes", []))
        csv_ops.append({"node": node.path, "gpus": " + ".join(e["gpus"]), "complete": complete, "method_s": m["seconds"],
                        "prepare_s": prepare, "data_wait_s": m.get("data_wait_s"), "flops": m["flops"], "energy_j": m["energy_j"],
                        "energy_above_idle_j": m["energy_above_idle_j"], "mean_power_w": m["mean_power_w"],
                        "idle_power_w": e["idle_watts"], "power_limit_w": e["limit_watts"],
                        "peak_allocated_gib": m["peak_allocated_gb"], "peak_reserved_gib": m["peak_reserved_gb"],
                        "energy_source": e["energy_source"], "max_gpu_processes": e["max_processes"], "flags": "; ".join(e["flags"])})
        for group, g in e["steps"].items():
            st, wt = g["step_ms"] or {}, g["wait_ms"] or {}
            step_rows.append([node.path, group, str(st.get("n", 0)), "-" if g["frames"] is None else str(g["frames"]),
                              f"{st['median']:.0f} / {st['p90']:.0f}" if st else "-",
                              f"{wt['median']:.0f} / {wt['p90']:.0f}" if wt else "-",
                              num(g["flops_per_step"], ".2f", 1e12), num(g["achieved_tflops"], ".1f")])
            csv_steps.append({"node": node.path, "group": group, "steps_timed": st.get("n", 0), "frames_per_step": g["frames"],
                              "step_median_ms": st.get("median"), "step_p90_ms": st.get("p90"), "step_mean_ms": st.get("mean"),
                              "wait_median_ms": wt.get("median"), "wait_p90_ms": wt.get("p90"),
                              "flops_per_step": g["flops_per_step"], "achieved_tflops": g["achieved_tflops"]})
        for name, ph in e["phases"].items():
            csv_phases.append({"node": node.path, "phase": name, "count": ph["count"], "seconds": ph["seconds"],
                               "energy_j": ph["energy_j"], "peak_allocated_gib": ph["peak_allocated_gb"],
                               "peak_reserved_gib": ph["peak_reserved_gb"]})
    for name, records in (("efficiency.csv", csv_ops), ("efficiency_steps.csv", csv_steps), ("efficiency_phases.csv", csv_phases)):
        (out / name).unlink(missing_ok=True)  # not an earlier pack's numbers
        if records:
            with open(out / name, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    md = ["## Efficiency per op", "", EFFICIENCY_NOTE, "",
          *table(["node", "GPU", "method (min)", "of which prepare (min)", "of which data wait (min)", "compute (PFLOP)", "energy (Wh)",
                  "above idle (Wh)", "mean power (W)", "peak alloc / reserved (GiB)", "energy source", "flags"], rows), ""]
    if notes:
        md += ["Measurement notes:", "", *[f"- {n}" for n in sorted(notes)], ""]
    md += ["### Steps and batches", "", STEPS_NOTE, "",
           *table(["node", "steps", "timed", "frames / step", "step median / p90 (ms)", "batch wait median / p90 (ms)",
                   "TFLOP / step", "achieved TFLOP/s"], step_rows),
           "", "Every number, and the time, energy and peak memory of every phase: `efficiency.csv`, "
           "`efficiency_steps.csv`, `efficiency_phases.csv`.", ""]
    return md


def plan_section(run: str, out: Path) -> list[str]:
    """Markdown: the status of every planned item, and the nodes trained with another recipe than the
    current code's. Writes out/failed/ (the log tail of every failed item)."""
    plan = load_plan(run)
    states = [(item_key(i) + (" (optional)" if i.get("optional") else ""), status(run, i)) for i in plan]
    md = ["## Plan status", "", *table(["item", "status"], states), ""]
    with contextlib.redirect_stdout(io.StringIO()):
        changed = warn_recipe_changes(run, plan)
    if changed:
        md += ["## Nodes trained with another recipe than the current code's", "",
               "Their results (and those of nodes trained on top of them) are not what the code now describes.", "",
               *[f"- {w.removeprefix('WARNING: ')}" for w in changed], ""]
    shutil.rmtree(out / "failed", ignore_errors=True)  # no log of an earlier pack's failures that have been rerun since
    (out / "failed").mkdir()
    for item in plan:
        if failed_file(run, item).exists():
            info = json.loads(failed_file(run, item).read_text())
            log = Path(info["log"])
            tail = "\n".join(log.read_text().splitlines()[-300:]) if log.exists() else "(log missing)"
            name = file_name(item)
            (out / "failed" / f"{name}.log").write_text(json.dumps(info, indent=1) + "\n\n" + tail + "\n")
    return md


def stream_metrics_section(run: str, stream: str, methods: list[str]) -> list[str]:
    """Markdown: scripts/metrics.py's table of the methods (PLAN.md's metrics)."""
    metrics = subprocess.run([sys.executable, str(PROJECT_DIR / "scripts" / "metrics.py"), "--stream", stream,
                              "--methods", ",".join(methods), "--runs", run], capture_output=True, text=True)
    return [f"## Stream metrics (scripts/metrics.py --stream {stream})", "", "```", metrics.stdout + metrics.stderr, "```", ""]


def drift_section(run: str, out: Path, changes: tuple[list[str], list[str]], script: str) -> list[str]:
    """Markdown: the drift script's summary, marked stale if its inputs changed since (changes: its
    input_changes); copies the script's report directory (CSVs, logs) to out/drift/."""
    drift = report_dir(run)
    shutil.rmtree(out / "drift", ignore_errors=True)  # only the current drift report's files
    if drift.exists():  # its logs too when it stopped before the summary
        shutil.copytree(drift, out / "drift")
    if not (drift / "summary.md").exists():
        return ["## Drift metrics", "", f"Not computed ({script} has not run, or stopped early: see "
                "run_local.log and `drift/`).", ""]
    heading, _, body = (drift / "summary.md").read_text().partition("\n")
    changed, new = changes
    if changed or new:  # e.g. packed while a rerun retrains nodes, before its drift step
        heading += ("\n\n**Stale**: computed from other files than the run has now"
                    + (f"; changed or removed since: {', '.join(changed)}" if changed else "")
                    + (f"; new since: {', '.join(new)}" if new else "") + ". The numbers of the other nodes are still valid.")
    return [heading + "\n" + body, ""]


def copy_run_files(run: str, out: Path) -> None:
    """Every node's configuration, training log and eval results (with videos), the output of every item,
    and the run's plan: everything but the weights."""
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
          "`drift/` (drift CSVs and logs), `failed/` (logs of failed items).", ""]
    md += plan_section(run, out)

    # --- Learning ------------------------------------------------------------------------------
    rows = [["base (zero-shot)", "final", sr(base, "A"), "-", sr(base, "B")]]
    for weights in ("step_002000", "step_005000", "final"):
        rows.append(["LA", weights, sr(la, "A", weights, 0), sr(la, "A", weights, 1), sr(la, "B", weights, 0)])
    rows.append(["LA/LB", "final", sr(lalb, "A", "final", 0), sr(lalb, "A", "final", 1), sr(lalb, "B", "final", 0)])
    md += ["## Learning", "", "Success rate [95% Wilson interval] over 50 rollouts. Pass bar (PLAN.md): A >= 0.80 "
           "after 10k steps. Seed 0 vs seed 1 = eval noise.", "",
           *table(["node", "weights", "A (eval seed 0)", "A (eval seed 1)", "B (eval seed 0)"], rows), "",
           *table(["node", "LA/LB: B eval seed 1"], [["LA/LB", sr(lalb, "B", "final", 1)]]), ""]

    # --- Speed and memory ------------------------------------------------------------------------
    nodes = [la, lalb] + [Node.from_path(run, f"LA/LB/UB-{v}") for v in VARIANTS]
    md += ["## Speed and memory per op", "", *table(["node", "steps/s (median)", "peak GPU mem (GB)", "wall (min)"],
                                                       [[n.path, *perf(train_log(n))] for n in nodes]), ""]

    md += efficiency_tables(nodes, out)

    # --- Unlearning variants -------------------------------------------------------------------
    theta0 = probes(train_log(lalb))
    oracle = probes(train_log(la))
    theta0_final = theta0[max(theta0)] if theta0 else {}
    oracle_final = oracle[max(oracle)] if oracle else {}
    rows = []
    for v in VARIANTS:
        node = Node.from_path(run, f"LA/LB/UB-{v}")
        p = probes(train_log(node))
        final = p[max(p)] if p and node.done else {}  # a failed or unfinished op has no final probe (its log may hold only step 0, the parent's)
        at = lambda step, task: fmt(p.get(step, {}).get(task), ".4f")  # noqa: E731
        results_node = lalb if node.last_op.shares_parent_weights else node
        rows.append([v, sr(results_node, "A"), sr(results_node, "B"), at(250, "B"), at(500, "B"), fmt(final.get("B"), ".4f"),
                     fmt(final.get("A"), ".4f"), *perf(train_log(node))[1:], method_info(node)])
    md += ["## Unlearning variants (forget B, retain A; parent LA/LB, retrain oracle LA)", "",
           "Probe loss = flow-matching loss on fixed probe frames (32 per task). Targets: B's probe loss should rise "
           f"to about the oracle's (LA never saw B): oracle B = {fmt(oracle_final.get('B'), '.4f')}, A = "
           f"{fmt(oracle_final.get('A'), '.4f')}; before unlearning (LA/LB): B = {fmt(theta0_final.get('B'), '.4f')}, "
           f"A = {fmt(theta0_final.get('A'), '.4f')}. SR B should fall to LA's SR on B; SR A should stay at LA/LB's.", "",
           *table(["variant", "SR A (retain)", "SR B (forget)", "probe B @250", "probe B @500", "probe B final",
                   "probe A final", "peak mem (GB)", "wall (min)", "method info"], rows), ""]

    # --- Stream metrics (PLAN.md) --------------------------------------------------------------
    md += stream_metrics_section(run, "CAL", VARIANTS)

    # --- Drift metrics (scripts/calibration_drift.py, run by run_local.sh after the workers) ---------
    md += drift_section(run, out, input_changes(run), "scripts/calibration_drift.py")
    (out / "summary.md").write_text("\n".join(md))

    copy_run_files(run, out)
    print(f"report written to {out}/summary.md")


if __name__ == "__main__":
    main()
