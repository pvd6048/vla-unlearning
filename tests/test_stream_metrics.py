"""Known-answer checks of vlu/stream_metrics.py on hand-made success matrices (no torch, no GPU).
    python tests/test_stream_metrics.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlu.stream_metrics import summarize  # noqa: E402

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
    "relapse": (.2 + 0.) / 2,              # B climbs back to .3 after LD
    "oracle_gap": (0. + .1 + 0.) / 3,      # UB: B; UE: B (.2 vs .1), E
}
a, b = 1 - expected["oracle_gap"], expected["retain_sr"]
expected["score"] = 2 * a * b / (a + b)
got = summarize(ops, sr, oracle)
for name, value in expected.items():
    assert abs(got[name] - value) < 1e-9, (name, got[name], value)
got = summarize(ops, sr[:-1] + [{}], oracle)  # last eval missing
assert got["score"] != got["score"], got  # NaN, not a misleading number

# Relearning after unlearning (S5-like): B is unlearned, then learned again.
got = summarize(["LA", "LB", "UB-ga", "LB"], [{"A": 1.}, {"A": 1., "B": 1.}, {"A": 1., "B": 0.}, {"A": 1., "B": .9}])
assert got["relapse"] == 0.0, got  # relearning B on purpose is not relapse
assert got["retain_sr"] == (1. + .9) / 2, got  # B is kept again at the end

# A missing eval gives NaN instead of a wrong number.
got = summarize(["LA", "LB"], [{"A": 1.}, {"B": 1.}])
assert got["nbt"] != got["nbt"], got

print("OK: stream metrics match hand-computed values")
