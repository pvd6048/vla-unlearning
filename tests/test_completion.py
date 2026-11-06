"""CPU checks for worker and launcher completion/failure propagation."""
import argparse
import ast
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Load the orchestration functions without importing Linux-only fcntl on Windows.
ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "vlu" / "worker.py").read_text())
names = {"item_key", "require_complete", "main", "_job_alive", "ENDED_STATES"}
worker = {"argparse": argparse, "os": os, "socket": socket, "subprocess": subprocess}
exec(compile(ast.Module(body=[n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
                              or (isinstance(n, ast.Assign) and n.targets[0].id in names)],
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

    def test_optional_items_are_reported_not_required(self):
        # The calibration's drift features: a failed one does not make the run incomplete.
        self.items[0]["done"] = True
        self.items.append({"kind": "features", "node": "LA", "weights": "final", "fisher": [], "optional": True})
        with self.assertRaisesRegex(SystemExit, "incomplete: eval"):
            worker["require_complete"]("test")
        self.items[1]["done"] = True
        with patch("builtins.print") as printed:
            worker["require_complete"]("test")
        self.assertIn("1 optional item(s) not done: features:LA:final", printed.call_args[0][0])

    def test_worker_does_not_report_success_when_only_failures_remain(self):
        with patch.dict(worker, status=lambda run, item: "failed"), patch("sys.argv", ["worker", "--run", "test"]), \
                self.assertRaises(SystemExit):
            worker["main"]()

    def test_worker_leaves_work_to_busy_workers_without_failing(self):
        # Without --wait (Slurm), a worker with nothing ready exits; that is a failure only if nobody is
        # still working on the plan (the rest is then stuck), not while another worker is training. A
        # worker whose job cannot be checked (squeue failing, another machine) may be dead: not trusted.
        for others in ("running", "cannot be checked", "none"):
            def someone_working(run, verified=False, others=others):
                return others == "running" or (others == "cannot be checked" and not verified)
            with self.subTest(others=others), patch("sys.argv", ["worker", "--run", "test"]), \
                    patch.dict(worker, status=lambda run, item: "waiting", is_ready=lambda run, item: False,
                               someone_working=someone_working):
                if others == "running":
                    worker["main"]()
                else:
                    with self.assertRaisesRegex(SystemExit, "incomplete"):
                        worker["main"]()

    def test_job_liveness_unknown_when_it_cannot_be_checked(self):
        alive, host = worker["_job_alive"], socket.gethostname()
        self.assertIsNone(alive(f"local1@not-{host}"), "another machine's process")
        if os.name != "nt":  # os.kill(pid, 0) checks a process on POSIX only
            dead = subprocess.Popen([sys.executable, "-c", "pass"])
            dead.wait()
            self.assertIs(alive(f"local{os.getpid()}@{host}"), True)
            self.assertIs(alive(f"local{dead.pid}@{host}"), False)

        def squeue(code, out="", err=""):
            return SimpleNamespace(run=lambda *a, **k: SimpleNamespace(returncode=code, stdout=out, stderr=err))
        for result, expected in [((0, "RUNNING\n"), True), ((0, "PENDING\n"), True), ((0, "COMPLETING\n"), False), ((0, ""), False),
                                 ((1, "", "slurm_load_jobs error: Invalid job id specified"), False),
                                 ((1, "", "slurm_load_jobs error: Socket timed out on send/recv operation"), None)]:
            with self.subTest(squeue=result), patch.dict(worker, subprocess=squeue(*result)):
                self.assertIs(alive("12345"), expected)

        def no_squeue(*a, **k):
            raise FileNotFoundError("squeue")
        with patch.dict(worker, subprocess=SimpleNamespace(run=no_squeue)):
            self.assertIsNone(alive("12345"))

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
        for failure in ("none", "planning", "submission", "worker", "incomplete", "drift"):
            # Execute the real launcher blocks with fake processes; no training or Slurm calls.
            prefix = '''set -uo pipefail
MAIN_PY=main; RUN=test; VLU_CPUS=4; OUT=/nonexistent; VLU_RUNS_DIR=/nonexistent
package() { :; }
nvidia-smi() { echo "0, 40000"; echo "1, 45000"; echo "2, 9000"; }
main() {
    [[ "$1" = scripts/calibration_drift.py ]] && echo "drift on GPU $CUDA_VISIBLE_DEVICES with $OMP_NUM_THREADS threads"
    [[ "$1" = scripts/calibration_drift.py && "$FAILURE" = drift ]] && return 6
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
                # Drift metrics are extra data: their failure is reported, not fatal.
                self.assertEqual(result.returncode, 0 if failure in ("none", "drift") else 1,
                                 result.stdout + result.stderr)
                if failure == "drift":
                    self.assertIn("WARNING: some drift metrics are missing", result.stdout)
                if failure not in ("planning", "submission"):  # the drift step runs once the workers are done
                    self.assertIn("drift on GPU 1 with 4 threads", result.stdout)  # the GPU with the most free memory


if __name__ == "__main__":
    unittest.main()
