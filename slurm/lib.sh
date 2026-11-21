# Shell functions shared by scripts/run_local.sh, slurm/run.sbatch and slurm/worker.sbatch. Source it after
# slurm/env.sh; do not run it.
#   source slurm/lib.sh

# select_mode <calibration|main>: the run and the scripts of an experiment. RUN, if set, wins (run_local.sh).
# REQUIRED_TASKS: the tasks the experiment evaluates, for scripts/check_tasks.py --required (empty = every
# task of vlu/tasks.py); an unusable task it does not evaluate is only warned about.
select_mode() {
    MODE=$1
    if [ "$MODE" = main ]; then
        RUN="${RUN:-goal_s0}" PLAN=scripts/main_run.py REPORT=scripts/main_report.py DRIFT=scripts/main_drift.py REQUIRED_TASKS=
    else  # scripts/calibrate.py evaluates tasks A and B only
        RUN="${RUN:-cal}" PLAN=scripts/calibrate.py REPORT=scripts/calibration_report.py DRIFT=scripts/calibration_drift.py REQUIRED_TASKS=A,B
    fi
}

# wait_for <pid>: wait until a background child of this shell has exited, and return its exit code. A trapped
# signal (Slurm's USR1 before the time limit) interrupts `wait` with 128 + the signal number; if the child exited
# meanwhile, bash has kept its status, which a second `wait` returns.
wait_for() {
    local status
    while true; do
        wait "$1"; status=$?
        kill -0 "$1" 2>/dev/null || break
    done
    wait "$1" 2>/dev/null
    local again=$?
    [ "$again" = 127 ] && return "$status"  # no longer known to bash: the first status is all there is
    return "$again"
}

# package: the report and the tarball $OUT.tar.gz of the output directory $OUT (no model weights).
package() {
    cp -f "$VLU_PROJECT_DIR/setup.log" "$OUT/" 2>/dev/null || true
    "$MAIN_PY" "$REPORT" --run "$RUN" --out "$OUT" || echo "WARNING: report failed"
    tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
    echo "=== $(date)  done: $OUT.tar.gz ($(du -h "$OUT.tar.gz" | cut -f1))"
}

# finish_run <drift only: 0|1> <workers ok: 0|1>: after the workers, the drift metrics from the checkpoints and
# features that exist (CPU threads for their CPU parts; on GPU $DRIFT_GPU if set), the completeness check, the
# report and the tarball. Drift exit code 3 = only drifts left out (nodes without weights or features). A drift
# failure is reported, not fatal, except with --drift-only, whose only result they are. Returns 1 if the
# experiment is not complete or a worker failed.
finish_run() {
    local drift_only=$1 workers_ok=$2 drift_status logs
    "$MAIN_PY" scripts/submit.py --run "$RUN" --workers none
    (
        if [ -n "${DRIFT_GPU:-}" ]; then export CUDA_VISIBLE_DEVICES="$DRIFT_GPU"; fi
        OMP_NUM_THREADS="$VLU_CPUS" "$MAIN_PY" "$DRIFT" --run "$RUN"
    )
    drift_status=$?
    logs="see summary.md; logs in $VLU_RUNS_DIR/$RUN/drift_report/, packed as drift/"
    if [ "$drift_only" = 1 ]; then
        if [ "$drift_status" != 0 ] && [ "$drift_status" != 3 ]; then
            echo "ERROR: the drift metrics failed ($logs)"; package; return 1
        fi
        # Left out may only be models not trained yet: the features of every trained one must exist.
        if ! "$MAIN_PY" -m vlu.worker --run "$RUN" --kinds features --check-complete --ready-only; then
            echo "ERROR: the drift features of some trained models are missing (logs in failed/)"; package; return 1
        fi
        echo "drift only: training and evals were not run; models not trained yet are left out of the drift metrics"
    else
        if [ "$drift_status" != 0 ]; then echo "WARNING: some drift metrics are missing ($logs)"; fi
        if ! "$MAIN_PY" -m vlu.worker --run "$RUN" --check-complete; then
            echo "ERROR: the experiment did not complete every planned item"; package; return 1
        fi
    fi
    if [ "$workers_ok" != 1 ]; then echo "ERROR: a worker exited unsuccessfully"; package; return 1; fi
    package
}
