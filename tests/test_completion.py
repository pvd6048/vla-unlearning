"""CPU checks for worker and launcher completion/failure propagation."""
import argparse
import ast
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
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

    def test_ready_only_check_requires_every_item_whose_node_exists(self):
        # run_local.sh --drift-only: the features of the trained models must all be done (optional or not);
        # those of models not trained yet are not asked for.
        self.items[:] = [{"kind": "features", "node": n, "weights": "final", "fisher": [], "optional": True}
                         for n in ("LA", "LA/LB", "LA/LB/UB-ga")]
        trained = {"LA", "LA/LB"}
        with patch.dict(worker, is_ready=lambda run, item: item["node"] in trained):
            with self.assertRaisesRegex(SystemExit, "incomplete: features:LA/LB:final"):
                self.items[0]["done"] = True
                worker["require_complete"]("test", {"features"}, ready_only=True)
            self.items[1]["done"] = True
            worker["require_complete"]("test", {"features"}, ready_only=True)  # UB-ga is not trained: not needed
            worker["require_complete"]("test", {"features"})  # optional anyway without ready_only
            trained.clear()
            with self.assertRaisesRegex(SystemExit, "the plan is empty"):
                self.items.clear()
                worker["require_complete"]("test", {"features"}, ready_only=True)

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
        planning = source[source.index('if ! "$MAIN_PY" "$PLAN" --run "$RUN"'):source.index("# 5. One worker")]
        completion = source[source.index("WORKERS_OK=1"):]
        runs = (("calibration", "scripts/calibrate.py", "scripts/calibration_drift.py"),
                ("main", "scripts/main_run.py", "scripts/main_drift.py"))
        for (mode, plan, drift), drift_only in ((r, d) for r in runs for d in (0, 1)):
            for failure in ("none", "planning", "submission", "worker", "incomplete", "drift", "left_out"):
                # Execute the real launcher blocks with fake processes; no training or Slurm calls.
                prefix = f'''set -uo pipefail
MAIN_PY=main; RUN=test; VLU_CPUS=4; OUT=/nonexistent; VLU_RUNS_DIR=/nonexistent; MODE={mode}; PLAN={plan}; DRIFT={drift}; DRIFT_ONLY={drift_only}
''' + '''package() { :; }
nvidia-smi() { echo "0, 40000"; echo "1, 45000"; echo "2, 9000"; }
main() {
    [[ "$1" = scripts/submit.py ]] && echo "submit $*"
    [[ "$1" = "$DRIFT" ]] && echo "$1 on GPU $CUDA_VISIBLE_DEVICES with $OMP_NUM_THREADS threads"
    [[ "$1" = "$DRIFT" && "$FAILURE" = drift ]] && return 6
    [[ "$1" = "$DRIFT" && "$FAILURE" = left_out ]] && return 3
    [[ "$1" = "$PLAN" && "$FAILURE" = planning ]] && return 2
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
                out = result.stdout
                with self.subTest(mode=mode, drift_only=drift_only, failure=failure):
                    # Drift metrics are extra data: their failure is reported, not fatal, except with --drift-only,
                    # whose only result they are. Drifts left out (exit 3: nodes not trained yet) never are.
                    # --drift-only needs the features of the trained models done, not the rest of the plan.
                    ok = ("none", "left_out") if drift_only else ("none", "drift", "left_out")
                    self.assertEqual(result.returncode, 0 if failure in ok else 1, out + result.stderr)
                    if failure == "planning":
                        self.assertIn(f"ERROR: {mode} planning failed", out)
                    else:  # --drift-only forgets only the failures of features items: the others stay reported
                        retry = "features" if drift_only else "train,eval,features"
                        self.assertIn(f"submit scripts/submit.py --run test --retry-failed --retry-kinds {retry} --workers none", out)
                    if failure in ("drift", "left_out") and not drift_only:
                        self.assertIn("WARNING: some drift metrics are missing", out)
                    if drift_only and failure == "drift":
                        self.assertIn("ERROR: the drift metrics failed", out)
                    if drift_only and failure == "incomplete":
                        self.assertIn("ERROR: the drift features of some trained models are missing", out)
                    if failure not in ("planning", "submission"):  # the drift step runs once the workers are done
                        self.assertIn(f"{drift} on GPU 1 with 4 threads", out)  # the GPU with the most free memory

    def test_launcher_checks_every_task_in_libero(self):
        # The real task_preflight, with a fake simulator interpreter whose make_env knows only some tasks.
        bash = os.environ.get("VLU_TEST_BASH") or shutil.which("bash")
        if not bash:
            self.skipTest("bash is not available")
        source = (ROOT / "scripts" / "run_local.sh").read_text()
        start = source.index("task_preflight() {")
        block = source[start:source.index("\n}\n", start) + 3].replace("source slurm/env.sh;", "source slurm/env.sh; SIM_PY=$FAKE_SIM;")
        self.assertIn("SIM_PY=$FAKE_SIM", block)
        tmp = Path(tempfile.mkdtemp())
        fake = tmp / "sim_py"
        fake.write_text(f"""#!/bin/sh
exec {sys.executable} -c '
import os, sys, types
module = types.ModuleType("vlu.eval.libero_client")
class Env:
    def __init__(self):
        self._env = self
    def close(self):
        print("closed", flush=True)
def make_env(letter):
    if letter in os.environ["BAD_TASKS"]:
        raise ValueError("LIBERO instruction differs for " + letter)
    return Env(), list(range(50))
module.make_env = make_env
sys.modules["vlu.eval.libero_client"] = module
exec(sys.stdin.read())
'
""")
        fake.chmod(0o755)
        for mode, bad in (("main", ""), ("main", "D"), ("calibration", "D")):
            out = tmp / f"{mode}{bad}"
            out.mkdir(exist_ok=True)
            result = subprocess.run([bash, "-c", f"set -uo pipefail; OUT={out}; MODE={mode}; VLU_RENDER=mesa\n"
                                     f"{block}task_preflight; echo \"status $?\""], cwd=ROOT, text=True, capture_output=True,
                                    env={**os.environ, "FAKE_SIM": str(fake), "BAD_TASKS": bad or "-"})
            log = (out / "environment.txt").read_text()
            with self.subTest(mode=mode, bad=bad):
                if bad and mode == "main":  # the launcher stops (status 1)
                    self.assertIn("status 1", result.stdout, result.stdout + result.stderr)
                    self.assertIn("ERROR: a task of vlu/tasks.py is not usable in LIBERO", result.stdout)
                    self.assertIn("LIBERO instruction differs for D", log)
                elif bad:  # the calibration uses A and B only: warned, not stopped
                    self.assertIn("WARNING: a task of vlu/tasks.py is not usable in LIBERO", result.stdout)
                    self.assertIn("status 0", result.stdout)
                else:
                    self.assertIn("status 0", result.stdout, result.stdout + result.stderr + log)
                    for letter in "ABCDE":
                        self.assertIn(f"task {letter} OK:", log)
                    self.assertEqual(log.count("closed"), 5)  # every env's render context is closed

    def test_launcher_skips_simulator_checks_for_drift_only(self):
        # Step 2 as written: --drift-only evaluates nothing, so neither rendering nor the tasks are checked.
        bash = os.environ.get("VLU_TEST_BASH") or shutil.which("bash")
        if not bash:
            self.skipTest("bash is not available")
        source = (ROOT / "scripts" / "run_local.sh").read_text()
        start = source.index('if [ "$DRIFT_ONLY" = 0 ]; then\n    if gpu_render_ok')
        block = source[start:source.index("\nfi\n", start) + 4]
        fakes = ('OUT=$(mktemp -d); package() { :; }; gpu_render_ok() { echo "render checked"; }; '
                 'render_test() { :; }; task_preflight() { echo "tasks checked"; }\n')
        for drift_only, checked in ((0, True), (1, False)):
            result = subprocess.run([bash, "-c", f"DRIFT_ONLY={drift_only}; {fakes}{block}"], text=True, capture_output=True)
            with self.subTest(drift_only=drift_only):
                self.assertEqual("render checked" in result.stdout and "tasks checked" in result.stdout, checked, result.stdout)

    def test_launcher_worker_kinds(self):
        # The real worker_kinds of step 5: what each GPU's worker may run, by its free memory.
        bash = os.environ.get("VLU_TEST_BASH") or shutil.which("bash")
        if not bash:
            self.skipTest("bash is not available")
        source = (ROOT / "scripts" / "run_local.sh").read_text()
        start = source.index("worker_kinds() {")
        function = source[start:source.index("\n}\n", start) + 3]
        for drift_only, expected in ((0, "train,eval,features|eval"), (1, "features|")):
            result = subprocess.run([bash, "-c", f"TRAIN_FREE_MIB=44000; DRIFT_ONLY={drift_only}\n{function}"
                                     'echo "$(worker_kinds 45000)|$(worker_kinds 20000)"'], text=True, capture_output=True)
            with self.subTest(drift_only=drift_only):
                self.assertEqual(result.stdout.strip(), expected, result.stderr)

    def test_launcher_options_select_the_run(self):
        bash = os.environ.get("VLU_TEST_BASH") or shutil.which("bash")
        if not bash:
            self.skipTest("bash is not available")
        source = (ROOT / "scripts" / "run_local.sh").read_text()
        block = source[source.index("MODE=calibration PACK=0"):source.index('mkdir -p "$OUT"')]
        show = 'echo "$MODE $PACK $RUN $PLAN $REPORT $DRIFT $(basename "$OUT")$([ "$DRIFT_ONLY" = 1 ] && echo " 1")"'
        host = socket.gethostname()
        cal = "cal scripts/calibrate.py scripts/calibration_report.py scripts/calibration_drift.py"
        main = "goal_s0 scripts/main_run.py scripts/main_report.py scripts/main_drift.py"
        cases = {(): f"calibration 0 {cal} calibration_{host}", ("--pack",): f"calibration 1 {cal} calibration_{host}",
                 ("--main",): f"main 0 {main} main_{host}", ("--main", "--pack"): f"main 1 {main} main_{host}",
                 # its own output directory: the run's tarball and worker logs stay as they are
                 ("--drift-only",): f"calibration 0 {cal} calibration_drift_{host} 1",
                 ("--main", "--drift-only"): f"main 0 {main} main_drift_{host} 1"}
        env = {k: v for k, v in os.environ.items() if k != "RUN"}
        for args, expected in cases.items():
            result = subprocess.run([bash, "-c", "set -uo pipefail; VLU_STORE=/store\n" + block + show, "run_local", *args],
                                    text=True, capture_output=True, env=env)
            with self.subTest(args=args):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)
        result = subprocess.run([bash, "-c", block, "run_local", "--mian"], text=True, capture_output=True, env=env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown option --mian", result.stdout)

if __name__ == "__main__":
    unittest.main()
