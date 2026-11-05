"""CPU checks for worker and launcher completion/failure propagation."""
import argparse
import ast
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

# Load the orchestration functions without importing Linux-only fcntl on Windows.
ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "vlu" / "worker.py").read_text())
names = {"item_key", "require_complete", "main"}
worker = {"argparse": argparse}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                        type_ignores=[]), str(ROOT / "vlu" / "worker.py"), "exec"), worker)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.items = [{"kind": "train", "node": "LA"},
                      {"kind": "eval", "node": "LA", "weights": "final", "seed": 0}]
        worker.update(load_plan=lambda run: self.items, is_done=lambda run, item: item.get("done", False))

    def test_failed_waiting_and_claimed_items_are_not_complete(self):
        for state in ("failed", "waiting", "claimed"):
            self.items[0]["done"] = True
            self.items[1]["status"] = state
            with self.subTest(state=state), self.assertRaisesRegex(SystemExit, "incomplete"):
                worker["require_complete"]("test")

    def test_success_and_restricted_worker_scope(self):
        self.items[0]["done"] = True
        worker["require_complete"]("test", {"train"})
        with self.assertRaises(SystemExit):
            worker["require_complete"]("test")
        self.items[1]["done"] = True
        worker["require_complete"]("test")

    def test_empty_plan_is_not_a_completed_experiment(self):
        self.items.clear()
        with self.assertRaises(SystemExit):
            worker["require_complete"]("test")

    def test_worker_does_not_report_success_when_only_failures_remain(self):
        worker["status"] = lambda run, item: "failed"
        with patch("sys.argv", ["worker", "--run", "test"]), self.assertRaises(SystemExit):
            worker["main"]()

    def test_check_complete_never_starts_work(self):
        self.items[0]["done"] = self.items[1]["done"] = True
        with patch("sys.argv", ["worker", "--run", "test", "--check-complete"]):
            worker["main"]()

    def test_launcher_propagates_planning_worker_and_completion_failures(self):
        bash = os.environ.get("VLU_TEST_BASH") or shutil.which("bash")
        if not bash:
            self.skipTest("bash is not available")
        source = (ROOT / "scripts" / "run_local.sh").read_text()
        planning = source[source.index('if ! "$MAIN_PY" scripts/calibrate.py'):source.index("# 5. One worker")]
        completion = source[source.index("WORKERS_OK=1"):]
        for failure in ("none", "planning", "submission", "worker", "incomplete"):
            # Execute the real launcher blocks with fake processes; no training or Slurm calls.
            prefix = '''set -uo pipefail
MAIN_PY=main; RUN=test
package() { :; }
main() {
    [[ "$1" = scripts/calibrate.py && "$FAILURE" = planning ]] && return 2
    [[ "$1" = scripts/submit.py && "$FAILURE" = submission ]] && return 3
    [[ "$1" = -m && "$FAILURE" = incomplete ]] && return 4
    return 0
}
PIDS=()
(exit 0) & PIDS+=($!)
(if [[ "$FAILURE" = worker ]]; then exit 5; fi) & PIDS+=($!)
'''
            result = subprocess.run([bash], input=prefix + planning + completion, text=True,
                                    capture_output=True, env={**os.environ, "FAILURE": failure})
            with self.subTest(failure=failure):
                self.assertEqual(result.returncode, 0 if failure == "none" else 1,
                                 result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
