"""Known-answer checks of vlu/stream_metrics.py on hand-made success matrices (no torch, no GPU).
    python tests/test_stream_metrics.py
"""
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ["VLU_RUNS_DIR"] = tempfile.mkdtemp(prefix="vlu_metrics_")  # before importing vlu
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlu.nodes import Node  # noqa: E402
from vlu.stream_metrics import success_rates, summarize  # noqa: E402

# Stream S1 with made-up success rates; expected values worked out by hand.
ops = ["LA", "LB", "LC", "UB-ga", "LD", "LE", "UE-ga"]
sr = [
    {"A": .9},
    {"A": .8, "B": .9},
    {"A": .8, "B": .9, "C": 1.},
    {"A": .6, "B": .1, "C": .9},
    {"A": .6, "B": .3, "C": .9, "D": .8},
    {"A": .6, "B": .2, "C": .9, "D": .8, "E": .9},
    {"A": .5, "B": .2, "C": .8, "D": .7, "E": 0.},
]
oracle = {3: {"A": .7, "B": .1, "C": .9}, 6: {"A": .6, "B": .1, "C": .9, "D": .8, "E": 0.}}
expected = {
    "retain_sr": (.5 + .8 + .7) / 3,       # A, C, D kept at the end
    "nbt": (1.5 / 6 + 0 + .5 / 4 + .1 / 2) / 4,  # A, B (until UB), C, D; E has no later op
    "max_forgetting": .4,                  # A: .9 -> .5
    "forget_sr": (.1 + 0.) / 2,            # B after UB, E after UE
    "collateral": (.2 + .1 + .1 + .1 + .1) / 5,  # UB hurts A, C; UE hurts A, C, D
    "relapse": .2,                         # B climbs back to .3 after LD; E (last op) has no later op: left out
    "oracle_gap": (0. + .1 + 0.) / 3,      # UB: B; UE: B (.2 vs .1), E
}
a, b = 1 - expected["oracle_gap"], expected["retain_sr"]
expected["score"] = 2 * a * b / (a + b)
got = summarize(ops, sr, oracle)
for name, value in expected.items():
    assert abs(got[name] - value) < 1e-9, (name, got[name], value)
got = summarize(ops, sr[:-1] + [{}], oracle)  # last eval missing
assert got["score"] != got["score"], got  # NaN, not a misleading number

got = summarize(ops, sr, oracle)
assert abs(got["nbt_from_l_ops"] + got["nbt_from_u_ops"] - got["nbt"]) < 1e-12, got  # the split adds up to NBT
assert summarize(ops, sr, {3: oracle[3]})["oracle_gap"] != summarize(ops, sr, {3: oracle[3]})["oracle_gap"]  # NaN, no KeyError

# NBT split by cause, by hand: A drops .1 at LB, .2 at UB-ga, .1 at LC; measured after ops 1, 2, 3.
got = summarize(["LA", "LB", "UB-ga", "LC"], [{"A": 1.}, {"A": .9, "B": 1.}, {"A": .7, "B": 0.}, {"A": .6, "B": 0., "C": 1.}])
assert abs(got["nbt"] - (.1 + .3 + .4) / 3) < 1e-12, got
assert abs(got["nbt_from_l_ops"] - (.1 + .1 + .2) / 3) < 1e-12 and abs(got["nbt_from_u_ops"] - (0 + .2 + .2) / 3) < 1e-12, got

# Relearning after unlearning (S5-like): B is unlearned, then learned again.
got = summarize(["LA", "LB", "UB-ga", "LB"], [{"A": 1.}, {"A": 1., "B": 1.}, {"A": 1., "B": 0.}, {"A": 1., "B": .9}])
assert math.isnan(got["relapse"]), got  # relearning B on purpose is not relapse; no op in between: undefined
assert got["retain_sr"] == (1. + .9) / 2, got  # B is kept again at the end

# A missing eval gives NaN instead of a wrong number.
got = summarize(["LA", "LB"], [{"A": 1.}, {"B": 1.}])
assert got["nbt"] != got["nbt"], got

# Immediate unlearning (S3-like): nothing is kept right after U_A. That is not a missing eval.
got = summarize(["LA", "UA-ga", "LB", "UB-ga", "LC"], [{"A": .9}, {"A": .1}, {"A": .1, "B": .8}, {"A": .1, "B": .2}, {"A": .1, "B": .2, "C": .9}])
assert abs(got["retain_auc"] - (.9 + .8 + .9) / 3) < 1e-12 and got["retain_sr"] == .9, got
assert got["retain_stability"] != got["retain_stability"], got  # no two consecutive ops both keep something
got = summarize(["LA", "UA-ga"], [{"A": .9}, {"A": .1}])
assert math.isnan(got["retain_sr"]) and got["forget_sr"] == .1, got  # nothing kept at the end: undefined

# Reading evals: averaged over eval seeds; a task needs every seed; purge nodes use their parent's evals.
for seed, rates in [(0, {"A": .8, "B": .4}), (1, {"A": .6})]:
    d = Node.from_path("r", "LA/LB").eval_dir("final", seed)
    d.mkdir(parents=True)
    for t, v in rates.items():
        (d / f"{t}.json").write_text(json.dumps({"success_rate": v}))
assert success_rates(Node.from_path("r", "LA/LB"), "final", (0,)) == {"A": .8, "B": .4}
assert success_rates(Node.from_path("r", "LA/LB"), "final", (0, 1)) == {"A": .7}
assert success_rates(Node.from_path("r", "LA/LB/UB-purge"), "final", (0,)) == {"A": .8, "B": .4}

# Wilson 95% interval: textbook value for 40/50, and the edges stay inside [0, 1].
from vlu.stream_metrics import wilson_interval  # noqa: E402
lo, hi = wilson_interval(40, 50)
assert abs(lo - 0.6696) < 1e-3 and abs(hi - 0.8876) < 1e-3, (lo, hi)
assert wilson_interval(0, 50)[0] == 0.0 and wilson_interval(50, 50)[1] <= 1.0

print("OK: stream metrics match hand-computed values")
