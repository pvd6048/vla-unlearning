"""CPU checks of S1 benchmark planning, variant identity, reports and cost exports.

Values below are synthetic fixtures for checking reporting, never model benchmark results.
On Windows an empty fcntl import stand-in allows importing the Linux worker's pure planning
helpers. No locking, worker execution or GPU measurement is substituted or tested here.
"""
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import stubs  # noqa: F401

if sys.platform == "win32":
    # S1 repeats long override tokens; use Windows' long-path form for the fixture directory.
    os.environ["VLU_RUNS_DIR"] = "\\\\?\\" + os.environ["VLU_RUNS_DIR"]

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
if sys.platform == "win32":
    sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))

import main_drift
import main_report
import main_run
from calibration_report import cost as reported_cost, efficiency_tables
from submit import merge
from vlu.config import RUNS_DIR
from vlu.nodes import Node, Op
from vlu.stream_metrics import stream_matrix, summarize, success_rates


METHODS = ["purge", "task_arithmetic@unlearn_lr=1e-5+task_arithmetic_scale=0.5",
           "task_arithmetic@unlearn_lr=1e-5+task_arithmetic_scale=1",
           "rmu@unlearn_lr=1e-5+rmu_steering=2"]


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.run = "bench_" + self.id().split(".")[-1]
        self.plan = merge([], main_run.items(self.run, METHODS))
        directory = RUNS_DIR / self.run
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "plan.json").write_text(json.dumps(self.plan))
        (directory / "run.json").write_text(json.dumps({"seed": 0}))
        success_rates.cache_clear()

    def test_full_stream_oracles_features_and_variants_are_planned(self):
        self.assertCountEqual(main_drift.planned_methods(self.run), METHODS)
        trained = {i["node"] for i in self.plan if i["kind"] == "train"}
        for method in METHODS:
            self.assertIn(f"LA/LB/LC/UB-{method}/LD/LE/UE-{method}", trained)
            self.assertEqual(Op.parse("UB-" + method).method_spec, method)
        self.assertTrue({"LA/LC", "LA/LC/LD"} <= trained)
        features = {i["node"] for i in self.plan if i["kind"] == "features"}
        self.assertEqual(features, set(main_drift.endpoints(METHODS)))
        available = set()
        main_drift.drifts(self.run, lambda node: available.add(node.path) or True)
        self.assertTrue(set(main_drift.endpoints(METHODS)) <= available)
        for item in self.plan:
            if item["kind"] == "eval":
                self.assertTrue(item["tasks"])
                self.assertEqual(item["seed"], 0)

    def populate_evals(self):
        for item in self.plan:
            node = Node.from_path(self.run, item["node"])
            if item["kind"] == "train":
                node.dir.mkdir(parents=True, exist_ok=True)
                node.weights_path.write_bytes(b"fixture: never load as a model")
                (node.dir / "op.json").write_text(json.dumps({"op": str(node.last_op), "notes": {}}))
            if item["kind"] == "eval":
                directory = node.eval_dir("final", 0)
                directory.mkdir(parents=True, exist_ok=True)
                for task in item["tasks"]:
                    (directory / f"{task}.json").write_text(json.dumps({
                        "success_rate": .5, "episodes": [{"success": True}, {"success": False}]}))

    def test_every_shared_performance_metric_and_report_uses_variant_paths(self):
        self.populate_evals()
        for method in METHODS:
            metrics = summarize(*stream_matrix(self.run, "S1", method))
            self.assertTrue(all(math.isfinite(x) for x in metrics.values()), metrics)
            self.assertEqual(metrics["oracle_gap"], 0.)
            self.assertEqual(metrics["retain_sr"], .5)
            self.assertAlmostEqual(metrics["score"], 2/3)
            self.assertIn("oracle_gap_u2", metrics)
            self.assertIn("nbt_from_u_ops", metrics)
        with tempfile.TemporaryDirectory(dir=RUNS_DIR.parent) as tmp:
            out = Path(tmp)
            with patch.object(sys, "argv", ["main_report.py", "--run", self.run, "--out", str(out)]):
                main_report.main()
            report = (out / "summary.md").read_text()
            for method in METHODS:
                self.assertIn(method, report)
            self.assertIn("oracle_gap_u2", report)
            self.assertIn("Efficiency per op", report)
            self.assertIn("Drift metrics", report)

    def test_cost_exports_include_all_existing_measurements_for_both_methods(self):
        nodes = [Node.from_path(self.run, "LA/LB/UB-" + method) for method in METHODS[1:]]
        for node in nodes:
            node.dir.mkdir(parents=True, exist_ok=True)
            # Exercise exports using known fixture values, without claiming GPU measurements.
            efficiency = {
                "method": {"seconds": 2., "data_wait_s": .1, "flops": 1e9, "energy_j": 20.,
                           "energy_above_idle_j": 10., "mean_power_w": 10.,
                           "peak_allocated_gb": 1., "peak_reserved_gb": 2.},
                "phases": {"train": {"count": 1, "seconds": 2., "energy_j": 20.,
                                     "peak_allocated_gb": 1., "peak_reserved_gb": 2.}},
                "steps": {"train": {"step_ms": {"n": 6, "median": 20., "p90": 30., "mean": 22.},
                                    "wait_ms": {"median": 1., "p90": 2.}, "frames": 16,
                                    "flops_per_step": 1e8, "achieved_tflops": .005}},
                "gpus": ["synthetic fixture"], "energy_source": "fixture", "energy_notes": [],
                "flags": [], "idle_watts": 5., "limit_watts": 100., "max_processes": 1,
                "unmeasured_sessions": 0}
            (node.dir / "cost.json").write_text(json.dumps({"seconds": 999., "complete": True, "efficiency": efficiency}))
            self.assertEqual(reported_cost(node), "0.0")  # method seconds, not session/instrumentation time
        with tempfile.TemporaryDirectory() as tmp:
            efficiency_tables(nodes, Path(tmp))
            for name in ("efficiency", "efficiency_steps", "efficiency_phases"):
                with (Path(tmp) / f"{name}.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), len(nodes))
            self.assertEqual(float(rows[0]["energy_j"]), 20.)


if __name__ == "__main__":
    unittest.main()
