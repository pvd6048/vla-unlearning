# Calibration run (on a machine without Slurm)

Thank you for running this. It fine-tunes the GR00T N1.6 robot policy on two LIBERO tasks, tries 38
settings of 8 unlearning methods, and evaluates everything in the LIBERO simulator. The results tell
us how to set our hyperparameters. You don't need to interpret anything: run it and send back one file.

## What the machine needs

- Linux (x86_64) with NVIDIA GPUs. At least one GPU with 48 GB or more (e.g. an RTX A6000) for
  training: it needs 44 GB free on that GPU (a 40 GB GPU is not enough). Smaller GPUs (e.g. an RTX
  A5000) are used for evaluation. All GPUs are used, so please don't run other jobs on them at the
  same time. The run checks the free memory at the start, warns about GPUs already in use, and stops
  if no GPU has 44 GB free.
- NVIDIA driver 570 or newer (CUDA 12.8); 525 or newer usually works too.
- About 350 GB of free disk (it fills to about 305 GB: 25 GB of software, model and data, 265 GB of
  trained model weights, 15 GB of drift data), 64 GB of RAM or more, internet access during setup (about 15 GB is downloaded).
- `git`, `curl`, `cmake` and a C compiler (`build-essential` on Ubuntu). No root access or Slurm is needed.

## Steps

Everything is installed and written inside the folder you clone into: Python environments, dataset,
model, results. Nothing goes elsewhere (except `uv`, installed to `~/.local/bin` if missing). So first
`cd` to any disk with about 350 GB free (`df -h .` shows the free space), then:

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
3. Trains task A (with checkpoints at 2k, 5k, 10k steps), then task B, then 38 unlearning variants of
   task B, and evaluates each in the simulator. One worker runs per GPU. After each training it also
   records how the model's internals respond to a fixed set of test frames (a few minutes per model).
4. Compares the trained models with each other (drift metrics, `scripts/calibration_drift.py`). These
   are extra data: if a recording or the comparison fails, the run says so (WARNING) but still counts
   as complete.
5. Writes `summary.md` and packs everything into the `.tar.gz`.
