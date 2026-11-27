# Calibration run (on a machine without Slurm)

Thank you for running this. It fine-tunes the GR00T N1.6 robot policy on two LIBERO tasks, tries 37
settings of 10 unlearning methods, and evaluates everything in the LIBERO simulator. It also measures what
each method costs: time, memory, compute and the GPU's energy. The results tell
us how to set our hyperparameters. You don't need to interpret anything: run it and send back one file.

## What the machine needs

- Linux (x86_64) with NVIDIA GPUs. At least one GPU with 48 GB or more (e.g. an RTX A6000) for
  training: it needs 44 GB free on that GPU (a 40 GB GPU is not enough). Smaller GPUs (e.g. an RTX
  A5000) are used for evaluation. All GPUs are used, so please don't run other jobs on them at the
  same time. The run checks the free memory at the start, warns about GPUs already in use, and stops
  if no GPU has 44 GB free.
- NVIDIA driver 570 or newer (CUDA 12.8); 525 or newer usually works too.
- About 400 GB of free disk (it fills to about 305 GB: 25 GB of software, model and data, 260 GB of
  trained model weights, 17 GB of drift data; up to 50 GB more for checkpoints while training), 64 GB of
  RAM or more, internet access during setup (about 15 GB is downloaded).
- `git`, `curl`, `cmake` and a C compiler (`build-essential` on Ubuntu). No root access or Slurm is needed.

## Steps

Everything is installed and written inside the folder you clone into: Python environments, dataset,
model, results. Nothing goes elsewhere (except `uv`, installed to `~/.local/bin` if missing). So first
`cd` to any disk with about 250 GB free (`df -h .` shows the free space), then:

```bash
git clone https://github.com/pvd6048/vla-unlearning.git
cd vla-unlearning
tmux new -s cal                                  # so it keeps running if the terminal closes
bash scripts/setup_cluster.sh 2>&1 | tee setup.log   # one time, roughly 30-60 minutes
bash scripts/run_local.sh                        # the whole calibration; expect about a day
```

To leave tmux running, press Ctrl-b then d; `tmux attach -t cal` brings it back.

When it finishes, it prints the file to send back, `calibration_<hostname>.tar.gz`, in the
`vla-unlearning` folder (a few MB; logs, tables and a few short videos, no model weights).

If the disk with space is not where you want the code, you can instead put the large files elsewhere:
`echo 'export VLU_STORE=/big/disk/vla-unlearning' > slurm/env.local.sh` before the setup step. The
`.tar.gz` is then written to that directory.

## Round 3 (on the machine that ran rounds 1 and 2)

This round keeps the training of tasks A and B from before and trains 36 unlearning variants: 24 new ones
(among them four newly added methods: GU on NPO-RT, Sharp MinMax, task arithmetic and RMU) and 12 trained before (in round 2 or on
24 November) again, now with the measurement of their time, memory, compute and energy. The 12 earlier
ones are moved to a separate folder first (`$VLU_RUNS_DIR/_archive/`; nothing is deleted). In the
`vla-unlearning` folder:

```bash
mv calibration_test.tar.gz calibration_test_before_round3.tar.gz   # keep the last file; round 3 writes this name again
git pull
git log -1 --format='%h %ad' --date=short               # must be dated 2026-11-27 or later (see below)
tmux new -s cal3
bash scripts/setup_cluster.sh 2>&1 | tee -a setup.log   # adds the energy-measurement package (a few minutes)
df -h .                                                   # needs about 290 GB free
bash scripts/run_local.sh                                 # about 13-17 hours
```

- **Is it the new code?** If `git pull` prints an error, or the `git log` line shows an earlier date,
  please stop and send us `git pull`'s message and the output of `git status`. Once it runs, the self-tests
  include `tests/test_task_arithmetic_rmu.py`, `tests/check_groot_rmu.py` and `tests/test_cuda_round3.py`, the plan line says
  `37 unlearning variants` (the 36 above plus "purge", which trains nothing; the run of 24 November said 38,
  the code of 25 November 17, earlier code of 27 November 27), and it prints
  `archived (trained before the cost meter; ...)` for 12 variants. If the plan line shows another number,
  please stop and send us that line.
- **Not enough disk?** Before `run_local.sh`, this deletes the model weights of the earlier rounds'
  variants that are no longer used (and of the 12 moved ones), up to about 180 GB; their results stay:
  `source slurm/env.sh && $MAIN_PY scripts/calibrate.py --prune-weights`
- **Please keep other programs off the GPUs** during this round: a GPU's energy reading includes
  everything running on it. (If something else runs there, the report flags that GPU's numbers.)
- **What to send back:** as before, `calibration_<hostname>.tar.gz` (the same name as round 2's file;
  this one replaces it).
- Everything in "If something goes wrong" below applies.

## If something goes wrong

In every case below, `bash scripts/run_local.sh --pack` (run in the `vla-unlearning` folder) writes
the same `.tar.gz` from whatever exists so far: all logs, including `setup.log`. Sending us that file is
always enough for us to see what happened.

- **Setup failed:** run `bash scripts/run_local.sh --pack` and send the `.tar.gz`. Rerunning the setup
  is safe; finished steps are skipped. If `setup.log` ends with `429 Client Error: Too Many Requests`
  (Hugging Face limits downloads per IP address), wait an hour or log in with a free Hugging Face
  account token (`export HF_TOKEN=hf_...`) and rerun the setup.
- **It reports an error:** it packs the `.tar.gz` itself; please send it.
- **It stopped (crash, reboot, Ctrl-C):** run `bash scripts/run_local.sh` again. Finished work is kept
  and interrupted work resumes. If it stops again at the same point, send the `--pack` file.
- **It seems stuck** (no progress for more than 2 hours): leave it running, run
  `bash scripts/run_local.sh --pack` in a second terminal and send that file.
- **Progress while it runs:** `source slurm/env.sh && $MAIN_PY scripts/submit.py --run cal --workers none`
  lists every step as done / ready / waiting / claimed (running) / failed.
- **The disk fills up:** stop it (Ctrl-C), free space, and run `bash scripts/run_local.sh` again.

## What it does (for reference)

1. Records the machine (GPUs, driver, library versions) and picks GPU rendering for the simulator if a
   test render works on every GPU, else software rendering.
2. Runs the code's self-tests (a few minutes). If one fails it stops here and packs the logs.
3. Trains task A (with checkpoints at 2k, 5k, 10k steps), then task B, then 37 unlearning variants of
   task B, and evaluates each in the simulator. Each variant's time, memory, compute and GPU energy are
   measured as it runs. One worker runs per GPU. After each training it also
   records how the model's internals respond to a fixed set of test frames (a few minutes per model).
4. Compares the trained models with each other (drift metrics, `scripts/calibration_drift.py`). These
   are extra data: if a recording or the comparison fails, the run says so (WARNING) but still counts
   as complete.
5. Writes `summary.md` and packs everything into the `.tar.gz`.

## Drift metrics of the models trained so far (only when we ask for it)

This trains and evaluates nothing. It records how each model trained so far responds to a fixed set of
test frames, compares the models with each other, and packs `calibration_drift_<hostname>.tar.gz` (a
separate file: the calibration's own `.tar.gz` stays as it is). Don't run it while the calibration or
the main run is running. In the `vla-unlearning` folder:

```bash
git pull
bash scripts/run_local.sh --drift-only           # about 1-2 hours; needs about 17 GB of free disk
```

It needs one GPU with 32 GB free while a Fisher (the two large items) is still to compute; GPUs with
16 GB free extract the other models' features meanwhile, and once both Fishers exist 16 GB is enough
(training needs 44 GB; this does not train). If it reports an error, it packs the `.tar.gz` anyway; please
send it. If it stops, run the same command again: finished parts are kept. Afterwards,
`bash scripts/run_local.sh` continues the calibration as normal.

If you ran it before 12 November 2026: running it again extracts every model's features again (about
20-60 minutes, depending on how many GPUs have 16 GB free; they are now stored in float32, which does not overflow). The old files, `drift/features.pt` in
each model's folder, are no longer used; you can delete them (about 1.5 GB) with

```bash
source slurm/env.sh && find "$VLU_RUNS_DIR/cal" -path '*/drift/features.pt' -delete
```

## Main run (only when we ask for it, after the calibration)

The main run trains the policy on five LIBERO tasks one after another, unlearns two of them with
three methods, evaluates every model in the simulator, and at the end compares the trained models
(drift metrics; as in the calibration, if that comparison fails it says so with a WARNING but the run
still counts as complete). It uses the GPUs like the calibration
does, so start it only after the calibration has finished. In the same `vla-unlearning` folder:

```bash
git pull
tmux new -s main
bash scripts/run_local.sh --main                 # expect about a day and a half
```

- **Disk:** about 160 GB free on top of what the calibration uses (`df -h .`). When we later add the
  remaining methods, it grows to about 300 GB.
- **What to send back:** when it finishes, it prints `main_<hostname>.tar.gz` (logs, tables and short
  videos, a few tens of MB, no model weights).
- **If something goes wrong:** everything in "If something goes wrong" above applies with `--main`
  added, e.g. `bash scripts/run_local.sh --main --pack`. Progress:
  `source slurm/env.sh && $MAIN_PY scripts/submit.py --run goal_s0 --workers none`.
- **Adding methods later:** we will ask you to `git pull` and run `bash scripts/run_local.sh --main`
  again. It adds the new methods and keeps everything already done.
