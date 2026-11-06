"""Drift metrics of the calibration run (vlu/drift, PLAN.md section 4): base-level numbers on how each
op changed the model, from the checkpoints the calibration trains anyway. Run by scripts/run_local.sh
after the workers; scripts/calibration_report.py adds the summary to summary.md and the CSVs to the
tarball.

    python scripts/calibration_drift.py [--run cal]

Drifts (theta_Y - theta_X; checkpoint references as in vlu/drift/weights.py):
    tauA    base..LA            learning A
    tauB    LA..LA/LB           learning B with A replayed: the task vector unlearning B should remove
    U:<v>   LA/LB..LA/LB/UB-v   each unlearning variant v
    O:<v>   LA..LA/LB/UB-v      what still separates variant v from its retrain oracle LA (never saw B)
("=" in v is written "~": drift names cannot hold "="). vlu.drift.params on the weights (with a random
direction and the Fisher of A and of B at LA/LB) writes params.csv, vlu.drift.compare on the probe
features writes representation.csv; pairwise numbers only for the pairs with tauA or tauB. Both go to
<run>/drift_report/, emptied first, with the tools' logs, summary.md and inputs.json (the files the
numbers come from: scripts/calibration_report.py flags the report as stale when they change).

The distance to the oracle is measured on O:<v>, not derived from U and tauB: theta_U - theta_LA =
U + tauB, but ||U||^2 + 2 <U, tauB> + ||tauB||^2 cancels to rounding noise near the oracle.

Only nodes whose weights / features exist are used; the others are listed. The calibration has one
training seed, so there is no seed-twin noise floor: the random direction is the only control.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import VARIANTS  # noqa: E402
from vlu.config import PROJECT_DIR, RUNS_DIR  # noqa: E402
from vlu.drift.paths import features_path, fisher_path  # noqa: E402
from vlu.nodes import Node  # noqa: E402

PARENT, ORACLE = "LA/LB", "LA"
UNLEARNED = [v for v in VARIANTS if v != "purge"]  # purge does not change the weights


def report_dir(run: str) -> Path:
    return RUNS_DIR / run / "drift_report"


def drift_name(variant: str, kind: str = "U") -> str:
    return f"{kind}:{variant.replace('=', '~')}"


def oracle_drift(d: str) -> str | None:
    """The drift from the oracle LA to the end of drift d: O:<v> for U:<v>; tauB for tauB (LA/LB, before
    unlearning); None for the others."""
    if d.startswith("U:"):
        return "O:" + d.removeprefix("U:")
    return d if d == "tauB" else None


def drifts(run: str, has) -> tuple[list[tuple[str, str, str]], list[str]]:
    """[(name, X, Y)] whose endpoints all have has(node) true, and the variants left out."""
    ok = lambda path: has(Node.from_path(run, path))  # noqa: E731
    kept = [d for d in [("tauA", "", ORACLE), ("tauB", ORACLE, PARENT)] if ok(d[1]) and ok(d[2])]
    missing = []
    for v in UNLEARNED:
        node = f"{PARENT}/UB-{v}"
        if ok(ORACLE) and ok(PARENT) and ok(node):
            kept += [(drift_name(v), PARENT, node), (drift_name(v, "O"), ORACLE, node)]
        else:
            missing.append(v)
    return kept, missing


def inputs(run: str) -> dict[str, list[int]]:
    """{path in the run: [size, mtime_ns]} of the files the drift metrics are computed from that exist
    now: the weights of LA, LA/LB and the variants (the base model's do not change), the probe features,
    the Fisher of A and B."""
    nodes = [Node.from_path(run, p) for p in ("", ORACLE, PARENT, *[f"{PARENT}/UB-{v}" for v in UNLEARNED])]
    files = [n.weights_path for n in nodes if not n.is_base] + [features_path(n) for n in nodes]
    files += [fisher_path(Node.from_path(run, PARENT), t) for t in ("A", "B")]
    found = {}
    for f in files:
        try:
            st = f.stat()
        except FileNotFoundError:
            continue
        found[str(f.relative_to(RUNS_DIR / run))] = [st.st_size, st.st_mtime_ns]
    return found


def input_changes(run: str) -> tuple[list[str], list[str]]:
    """Inputs changed or removed since the drift report was computed, and inputs that exist now but were
    not used (e.g. a variant trained since). Both empty: the report is up to date."""
    path = report_dir(run) / "inputs.json"
    used = json.loads(path.read_text()) if path.exists() else {}
    now = inputs(run)
    return sorted(f for f in used if now.get(f) != used[f]), sorted(set(now) - set(used))


def spec(name: str, x: str, y: str) -> str:
    return f"{name}={x or 'base'}..{y}"


def run_tool(module: str, args: list[str], log: Path, out: Path) -> str:
    """'ok', or 'FAILED ...' (a partly written output is deleted: no number of a failed run is reported)."""
    env = {**os.environ, "PYTHONPATH": str(PROJECT_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    with open(log, "w") as f:
        code = subprocess.run([sys.executable, "-m", module, *args, "--out", str(out)], stdout=f,
                              stderr=subprocess.STDOUT, env=env).returncode
    if code == 0:
        return "ok"
    out.unlink(missing_ok=True)
    print(f"WARNING: {module} failed (exit code {code}); see {log}", flush=True)
    return f"FAILED (exit code {code}; log `drift/{log.name}`)"


def read(path: Path) -> dict[tuple, float]:
    """{(level|feature, group|task, metric, drift_i, drift_j): value}"""
    if not path.exists():
        return {}
    with open(path) as f:
        return {tuple(r[:5]): float(r[5]) for r in list(csv.reader(f))[1:]}


def pair(rows: dict, a: str, b: str, metric: str, where: tuple[str, str]):
    """A symmetric pairwise metric, whichever order the tool wrote it in."""
    return rows.get((*where, metric, a, b), rows.get((*where, metric, b, a)))


def ratio(a, b):
    return None if a is None or b is None or b == 0 or math.isnan(a) or math.isnan(b) else a / b


# Markdown helpers, also used by scripts/calibration_report.py.
def fmt(x, spec: str = ".3g") -> str:
    """A number for a table: '-' if missing or NaN."""
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:{spec}}"


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *["| " + " | ".join(str(c) for c in row) + " |" for row in rows]]


def label(d: str) -> str:
    return d.removeprefix("U:").replace("~", "=") if d.startswith("U:") else f"*{d}*"


def weight_metrics(p: dict, d: str) -> dict | None:
    """The weights table's numbers for drift d (None if params.csv has no row for it)."""
    g = ("global", "all")
    norm = lambda name, m="norm": p.get((*g, m, name, ""))  # noqa: E731
    n = norm(d)
    if n is None:
        return None
    dit = p.get(("module", "dit", "norm", d, ""))
    oracle = oracle_drift(d)
    return {"norm": n, "size": ratio(n, norm("tauB")), "proj": p.get((*g, "proj", d, "tauB")),
            "cos": pair(p, d, "tauB", "cos", g),
            # ||theta_U - theta_LA|| / ||tauB||, measured on O:<v>
            "to_oracle": ratio(norm(oracle), norm("tauB")) if oracle else None,
            "selectivity": ratio(norm(d, "FB_norm"), norm(d, "FA_norm")),
            "dit_share": ratio(dit * dit, n * n) if dit is not None else None}


def dit_cka(r: dict) -> dict[tuple[str, str], float]:
    """{(drift, task): CKA distance of the DiT blocks' features, mean over blocks}"""
    values = defaultdict(list)
    for (feature, task, metric, drift, _), v in r.items():
        if metric == "cka_distance" and feature.startswith("dit.") and not math.isnan(v):
            values[(drift, task)].append(v)
    return {k: statistics.mean(v) for k, v in values.items()}


def policy_metrics(r: dict, cka: dict, d: str) -> dict | None:
    """The policy table's numbers for drift d (None if representation.csv has no row for it); cka = dit_cka(r)."""
    get = lambda name, task, metric: r.get(("policy", task, metric, name, ""))  # noqa: E731
    if get(d, "all", "flow_divergence") is None:
        return None
    oracle = oracle_drift(d)
    return {"fd_A": get(d, "A", "flow_divergence"), "fd_B": get(d, "B", "flow_divergence"),
            # fd(theta_U, theta_LA) / fd(theta_LA/LB, theta_LA) on B's frames, measured on O:<v>
            "B_to_oracle": ratio(get(oracle, "B", "flow_divergence"), get("tauB", "B", "flow_divergence")) if oracle else None,
            "dfm_A": get(d, "A", "fm_loss_change"), "dfm_B": get(d, "B", "fm_loss_change"),
            "lang_start": get(d, "all", "language_sensitivity_start"), "lang_end": get(d, "all", "language_sensitivity_end"),
            "cka_dit_A": cka.get((d, "A")), "cka_dit_B": cka.get((d, "B")), "cka_oracle_B": cka.get((oracle, "B"))}


def weight_rows(p: dict, names: list[str]) -> list[list]:
    rows = []
    for d in names + ["random"]:
        m = weight_metrics(p, d)
        if m is None:
            continue
        # by definition (tauB with itself) / arbitrary scale (random)
        m.update({"tauB": {"proj": 1, "cos": 1}, "random": {"norm": None, "size": None, "proj": None}}.get(d, {}))
        rows.append([label(d), *[fmt(m[k]) for k in ("norm", "size", "proj", "cos", "to_oracle", "selectivity", "dit_share")]])
    return rows


def policy_rows(r: dict, names: list[str]) -> list[list]:
    rows, cka = [], dit_cka(r)
    for d in names:
        m = policy_metrics(r, cka, d)
        if m is not None:
            rows.append([label(d), *[fmt(m[k]) for k in ("fd_A", "fd_B", "B_to_oracle", "dfm_A", "dfm_B")],
                         f"{fmt(m['lang_start'])} -> {fmt(m['lang_end'])}",
                         *[fmt(m[k]) for k in ("cka_dit_A", "cka_dit_B", "cka_oracle_B")]])
    return rows


def summary(out: Path, weight_drifts, feature_drifts, missing_w, missing_f, states: dict[str, str]) -> str:
    p, r = read(out / "params.csv"), read(out / "representation.csv")
    md = ["## Drift metrics (scripts/calibration_drift.py)", "",
          "How each op changed the model, from the checkpoints (PLAN.md section 4). tauA = base..LA (learning A), "
          "tauB = LA..LA/LB (learning B with A replayed), one row per unlearning variant U = LA/LB..LA/LB/UB-v. LA, "
          "which never saw B, is every variant's retrain oracle; distances to it are measured directly "
          "(LA..LA/LB/UB-v). Raw numbers: `drift/params.csv` (weights, per module and layer) and "
          "`drift/representation.csv` (probe features: CKA, PCA, policy, vision-language alignment, attention "
          "ratios, per task); pairwise numbers for the pairs with tauA or tauB. One training seed: no noise "
          "floor; the random direction is the only control.", ""]
    if any(s != "ok" for s in states.values()):
        md += ["**Incomplete**: " + "; ".join(f"{part}: {s}" for part, s in states.items()) + ".", ""]
    if missing_w or missing_f:
        md += [f"Left out (checkpoint or features missing): weights: {', '.join(missing_w) or 'none'}; "
               f"features: {', '.join(missing_f) or 'none'}.", ""]
    names = [n for n, _, _ in weight_drifts if not n.startswith("O:")]
    md += ["### Weights", "",
           "Trainable weights plus any frozen weight an op edited (VLA-Forget's vision layers). ‖D‖ = L2 norm; "
           "size and distance to the oracle in units of ‖tauB‖. proj on tauB = <D, tauB>/‖tauB‖²: -1 = removes "
           "exactly B's task vector. Distance to oracle = ‖theta_U - theta_LA‖ / ‖tauB‖ (1 = as far as before "
           "unlearning, the tauB row; 0 = at the oracle). F_B / F_A = ‖D‖ in the Fisher metric of B over that of "
           "A, both at LA/LB: how much more the change moves B-sensitive than A-sensitive weights (compare with "
           "the random row; the random direction's cosine with tauB is the chance level). DiT share = share of "
           "‖D‖² in the action head's DiT.", "",
           *table(["drift", "‖D‖", "‖D‖ / ‖tauB‖", "proj on tauB", "cos with tauB", "dist. to oracle", "F_B / F_A",
                   "DiT share"], weight_rows(p, names)), ""]
    names = [n for n, _, _ in feature_drifts if not n.startswith("O:")]
    md += ["### Policy and representations (probe set: 64 frames of each task A-E, fixed noise)", "",
           "fd = flow divergence: mean squared change of the predicted velocity (normalized action units). B dist. "
           "to oracle = fd(theta_U, theta_LA) / fd(theta_LA/LB, theta_LA) on B's frames (1 = B's behaviour as "
           "before unlearning, 0 = the oracle's). dL_FM = change of the flow-matching loss on the probe frames. "
           "Language sensitivity = how much the predicted actions change when the instruction is replaced by "
           "another task's (0 = the policy ignores the instruction), before -> after. CKA dist. = 1 - linear CKA "
           "of the DiT blocks' action-token features, mean over blocks: between the drift's ends on A's and on "
           "B's frames, and between theta_U and the oracle on B's frames (tauB row: LA/LB, before unlearning).", "",
           *table(["drift", "fd A", "fd B", "B dist. to oracle", "dL_FM A", "dL_FM B", "language sensitivity",
                   "CKA dist. DiT A", "CKA dist. DiT B", "CKA dist. to oracle DiT B"], policy_rows(r, names)), ""]
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="cal")
    args = parser.parse_args()
    run = args.run
    out = report_dir(run)
    shutil.rmtree(out, ignore_errors=True)  # no file of an earlier computation may be read or packed as current
    out.mkdir(parents=True)
    (out / "inputs.json").write_text(json.dumps(inputs(run), indent=1) + "\n")

    weight_drifts, missing_w = drifts(run, lambda node: node.done)
    feature_drifts, missing_f = drifts(run, lambda node: features_path(node).exists())
    fisher = []
    for t in ("A", "B"):
        if fisher_path(Node.from_path(run, PARENT), t).exists():
            fisher += ["--fisher", f"F{t}={PARENT}:{t}"]
    tools = {"weights": ("vlu.drift.params", weight_drifts, ["--random", *fisher], "params"),
             "features": ("vlu.drift.compare", feature_drifts, [], "representation")}
    states = {}
    for part, (module, found, extra, stem) in tools.items():
        names = [n for n, _, _ in found]
        if "tauB" not in names:
            states[part] = f"not computed (LA or LA/LB has no {part})"
            print(f"WARNING: LA or LA/LB has no {part}: no {part} drift", flush=True)
            continue
        tool_args = ["--run", run, *extra]
        for d in found:
            tool_args += ["--drift", spec(*d)]
        for t in ("tauA", "tauB"):
            if t in names:
                tool_args += ["--pairs-with", t]
        states[part] = run_tool(module, tool_args, out / f"{stem}.log", out / f"{stem}.csv")
    (out / "summary.md").write_text(summary(out, weight_drifts, feature_drifts, missing_w, missing_f, states) + "\n")
    print(f"drift report written to {out}")
    if any(s != "ok" for s in states.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
