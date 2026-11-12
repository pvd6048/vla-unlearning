"""Drift metrics of the main run (PLAN.md section 5: do forgetting and unlearning drift alike?), from the
checkpoints and probe features scripts/main_run.py plans. Run by scripts/run_local.sh --main after the
workers; scripts/main_report.py adds the summary to summary.md and the CSVs to the tarball.

    python scripts/main_drift.py [--run goal_s0]

theta0 = LA/LB/LC; target task B. Drifts (theta_Y - theta_X; checkpoint references as in vlu/drift/weights.py):
    tauB      LA..LA/LB                                 B's task vector
    tauD      LA/LB/LC..LA/LB/LC/LD                      learning D (B replayed): an unrelated op, the
                                                         generic-drift control of PLAN.md section 4
    F_B       LA/LB/LC/LD/LE..LA/LB/LC/UB-purge/LD/LE    incidental forgetting of B (section 5.1: branch F
                                                         minus control branch C, which still replays B)
    U:<m>     LA/LB/LC..LA/LB/LC/UB-m                    unlearning B with method m
    UF:<m>    LA/LB/LC/LD/LE..LA/LB/LC/UB-m/LD/LE        unlearning B, then learning D and E, against C
    O:<m>     LA/LC..LA/LB/LC/UB-m                       what still separates m from the retrain oracle
    O:theta0  LA/LC..LA/LB/LC                            the same before unlearning (the unit of distance)
vlu.drift.params on the weights (with a random direction and the Fisher of A and of B at theta0) writes
params.csv, vlu.drift.compare on the probe features writes representation.csv; pairwise numbers only for the
pairs with tauB or F_B. Both go to <run>/drift_report/, emptied first, with the tools' logs, summary.md and
inputs.json (scripts/main_report.py flags the report as stale when its inputs change).

PLAN.md section 5.3's questions: same direction (cos of U and F_B with each other and with -tauB), same
location (DiT share; per-layer numbers in the CSVs), same representational and behavioural effect (CKA,
flow divergence, the cosine between the changes of the predicted velocity). One training seed: no seed-twin
noise floor; the controls are the random direction and tauD.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration_drift as cd  # noqa: E402
import main_run  # noqa: E402
from vlu.drift.paths import fisher_path  # noqa: E402
from vlu.nodes import Node, Op, oracle_path  # noqa: E402

THETA0 = "LA/LB/LC"
C = main_run.CONTROL[-1]  # LA/LB/LC/LD/LE
F = f"{THETA0}/UB-purge/LD/LE"
ORACLE = oracle_path("", f"{THETA0}/UB-purge")  # LA/LC
FISHER = [(THETA0, t) for t in ("A", "B")]  # scripts/main_run.py extracts these
ANCHORS = ("tauB", "F_B")  # the pairwise numbers are for the pairs with these


def unlearned(methods: list[str]) -> list[str]:
    """The methods that change the weights (not purge)."""
    return [m for m in methods if not Op.parse(f"UB-{m}").shares_parent_weights]


def wanted(methods: list[str] | None = None) -> list[tuple[str, str, str]]:
    """[(name, X, Y)] for the methods (default: scripts/main_run.py's METHODS)."""
    out = [("tauB", "LA", "LA/LB"), ("tauD", THETA0, f"{THETA0}/LD"), ("F_B", C, F), ("O:theta0", ORACLE, THETA0)]
    for m in unlearned(main_run.METHODS if methods is None else methods):
        out += [(f"U:{m}", THETA0, f"{THETA0}/UB-{m}"), (f"UF:{m}", C, f"{THETA0}/UB-{m}/LD/LE"),
                (f"O:{m}", ORACLE, f"{THETA0}/UB-{m}")]
    return out


def drifts(run: str, has) -> tuple[list[tuple[str, str, str]], list[str]]:
    """[(name, X, Y)] whose ends both have has(node) true, and the names of the others."""
    kept, missing = [], []
    for name, x, y in wanted():
        (kept if has(Node.from_path(run, x)) and has(Node.from_path(run, y)) else missing).append((name, x, y))
    return kept, [name for name, _, _ in missing]


def endpoints(methods: list[str] | None = None) -> list[str]:
    """The nodes the drifts are measured between: those whose weights and probe features are used."""
    return list(dict.fromkeys(p for _, x, y in wanted(methods) for p in (x, y)))


def inputs(run: str) -> dict[str, list[int]]:
    """{path in the run: [size, mtime_ns]} of the weights, features and Fisher files the metrics use."""
    return cd.existing(run, [Node.from_path(run, p) for p in endpoints()],
                       [fisher_path(Node.from_path(run, path), t) for path, t in FISHER])


def input_changes(run: str) -> tuple[list[str], list[str]]:
    return cd.input_changes(run, inputs(run))


def label(d: str) -> str:
    return d if ":" in d else f"*{d}*"


def oracle_of(d: str) -> str | None:
    return "O:" + d.removeprefix("U:") if d.startswith("U:") else None


def weight_rows(p: dict, names: list[str]) -> list[list]:
    g = ("global", "all")
    norm = lambda d, m="norm": p.get((*g, m, d, ""))  # noqa: E731
    rows = []
    for d in names + ["random"]:
        n = norm(d)
        if n is None:
            continue
        dit = p.get(("module", "dit", "norm", d, ""))
        o = oracle_of(d)
        cells = {"cos_tauB": 1 if d == "tauB" else cd.pair(p, d, "tauB", "cos", g),  # random: arbitrary scale, no proj
                 "proj_tauB": 1 if d == "tauB" else None if d == "random" else p.get((*g, "proj", d, "tauB")),
                 "cos_F": 1 if d == "F_B" else cd.pair(p, d, "F_B", "cos", g),
                 "fcos_F": 1 if d == "F_B" else cd.pair(p, d, "F_B", "FB_cos", g),
                 "to_oracle": cd.ratio(norm(o), norm("O:theta0")) if o else None,
                 "selectivity": cd.ratio(norm(d, "FB_norm"), norm(d, "FA_norm")),
                 "dit_share": cd.ratio(dit * dit, n * n) if dit is not None else None}
        rows.append([label(d), cd.fmt(None if d == "random" else n), *[cd.fmt(cells[k]) for k in cells]])
    return rows


def policy_rows(r: dict, names: list[str]) -> list[list]:
    cka = cd.dit_cka(r)
    rows = []
    for d in names:
        get = lambda name, task, metric: r.get(("policy", task, metric, name, ""))  # noqa: E731
        if get(d, "all", "flow_divergence") is None:
            continue
        o = oracle_of(d)
        rows.append([label(d), *[cd.fmt(get(d, t, "flow_divergence")) for t in ("B", "A", "C")],
                     cd.fmt(cd.ratio(get(o, "B", "flow_divergence"), get("O:theta0", "B", "flow_divergence")) if o else None),
                     cd.fmt(get(d, "B", "fm_loss_change")),
                     f"{cd.fmt(get(d, 'all', 'language_sensitivity_start'))} -> {cd.fmt(get(d, 'all', 'language_sensitivity_end'))}",
                     cd.fmt(cka.get((d, "B"))), cd.fmt(cka.get((o, "B")) if o else None),
                     cd.fmt(1 if d == "F_B" else cd.pair(r, d, "F_B", "velocity_shift_cos", ("policy", "B")))])
    return rows


def summary(out: Path, weight_drifts, feature_drifts, missing_w, missing_f, states: dict[str, str]) -> str:
    p, r = cd.read(out / "params.csv"), cd.read(out / "representation.csv")
    md = ["## Drift metrics (scripts/main_drift.py)", "",
          "Do forgetting and unlearning move the model alike (PLAN.md section 5)? theta0 = LA/LB/LC. *tauB* = LA..LA/LB "
          "(B's task vector); *tauD* = learning D from theta0 (an unrelated op: the generic-drift control); *F_B* = "
          "incidental forgetting of B = branch F (purge, then LD LE) minus branch C (LD LE with B replayed); U:m = "
          "unlearning B from theta0; UF:m = unlearning B, then LD LE, minus C. Raw numbers per module and layer: "
          "`drift/params.csv`, per probe task and feature: `drift/representation.csv` (pairwise numbers for the pairs "
          "with tauB or F_B). One training seed: no noise floor; the controls are the random direction and tauD. "
          + cd.BF16_CAVEAT, ""]
    if any(s != "ok" for s in states.values()):
        md += ["**Incomplete**: " + "; ".join(f"{part}: {s}" for part, s in states.items()) + ".", ""]
    if missing_w or missing_f:
        md += [f"Left out (checkpoint or features missing): weights: {', '.join(missing_w) or 'none'}; "
               f"features: {', '.join(missing_f) or 'none'}.", ""]
    md += cd.nonfinite_note(r, feature_drifts)
    names = [n for n, _, _ in weight_drifts if not n.startswith("O:")]
    md += ["### Weights", "",
           "‖D‖ = L2 norm of the change of the trainable weights (plus frozen weights an op edited). cos / proj on "
           "tauB: negative = against learning B (proj -1 = removes exactly B's task vector). cos with F_B: does the "
           "drift point where forgetting B does (F_B-weighted: in the Fisher metric of B at theta0, i.e. on the "
           "weights B depends on). Dist. to oracle = ‖theta_U - theta_LA/LC‖ / ‖theta0 - theta_LA/LC‖ (1 = as far "
           "as before unlearning, 0 = at the retrain oracle). F_B / F_A = ‖D‖ in the Fisher metric of B over that "
           "of A at theta0 (compare with the random row). DiT share = share of ‖D‖² in the action head's DiT. "
           "Raw cosines in 1.6B dimensions are near 0 by chance: compare with the random row.", "",
           *cd.table(["drift", "‖D‖", "cos with tauB", "proj on tauB", "cos with F_B", "F_B-weighted cos with F_B",
                      "dist. to oracle", "F_B / F_A", "DiT share"], weight_rows(p, names)), ""]
    names = [n for n, _, _ in feature_drifts if not n.startswith("O:")]
    md += ["### Policy and representations (probe set: 64 frames of each task A-E, fixed noise)", "",
           "fd = flow divergence: mean squared change of the predicted velocity (normalized action units) on a "
           "task's frames. B dist. to oracle = fd(theta_U, LA/LC) / fd(theta0, LA/LC) on B's frames. dL_FM B = "
           "change of B's flow-matching loss. Language sensitivity before -> after (0 = the policy ignores the "
           "instruction). CKA dist. = 1 - linear CKA of the DiT blocks' action-token features on B's frames, mean "
           "over blocks: between the drift's ends, and between theta_U and the oracle. Velocity-shift cos with F_B "
           "= cosine between the drift's change of the predicted velocity and F_B's, on B's frames: do the "
           "unlearned and the forgotten policy fail the same way?", "",
           *cd.table(["drift", "fd B", "fd A", "fd C", "B dist. to oracle", "dL_FM B", "language sensitivity",
                      "CKA dist. DiT B", "CKA dist. to oracle DiT B", "velocity-shift cos with F_B (B)"],
                     policy_rows(r, names)), ""]
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="goal_s0")
    args = parser.parse_args()
    # Without B's task vector (LA, LA/LB) the drifts have nothing to be compared against.
    sys.exit(cd.compute(args.run, lambda has: drifts(args.run, has), inputs(args.run), FISHER, ANCHORS, "tauB", summary))


if __name__ == "__main__":
    main()
