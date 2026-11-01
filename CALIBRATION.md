# Calibration run (on a machine without Slurm)

Thank you for running this. It fine-tunes the GR00T N1.6 robot policy on two LIBERO tasks, tries 19
settings of 7 unlearning methods, and evaluates everything in the LIBERO simulator. The results tell
us how to set our hyperparameters. You don't need to interpret anything: run it and send back one file.

## What the machine needs

- Linux (x86_64) with NVIDIA GPUs. At least one GPU with 40 GB or more (e.g. an RTX A6000) for
  training; smaller GPUs (e.g. an RTX A5000) are used for evaluation. All GPUs are used, so please
  don't run other jobs on them at the same time.
- NVIDIA driver 570 or newer (CUDA 12.8); 525 or newer usually works too.
- About 100 GB of free disk, 64 GB of RAM or more, internet access during setup (about 12 GB is downloaded).
- `git`, `curl` and a C compiler (`build-essential` on Ubuntu). No root access or Slurm is needed.

## Steps

```bash
git clone https://github.com/pvd6048/vla-unlearning.git && cd vla-unlearning
echo 'export VLU_STORE=/path/with/100GB/free/vla-unlearning' > slurm/env.local.sh
bash scripts/setup_cluster.sh     # one time, roughly 30-60 minutes (installs into VLU_STORE)
bash scripts/run_local.sh         # the whole calibration; expect about a day
```

Use `tmux` or `nohup bash scripts/run_local.sh &` so it survives a closed terminal.

When it finishes, it prints the file to send back:
`<VLU_STORE>/calibration_<hostname>.tar.gz` (a few MB; logs, tables and a few short videos, no model
weights).

## If something goes wrong

- **It stopped (crash, reboot, Ctrl-C):** just run `bash scripts/run_local.sh` again. Finished work is
  kept and interrupted work resumes.
- **It reports an error:** it still writes the `.tar.gz` with the logs. Please send that; it is what
  we need to fix the problem.
- **Setup failed:** send the terminal output of `bash scripts/setup_cluster.sh`.
- **Progress while it runs:** `source slurm/env.sh && $MAIN_PY scripts/submit.py --run cal --workers none`
  lists every step as done / ready / waiting / claimed (running) / failed.

## What it does (for reference)

1. Records the machine (GPUs, driver, library versions) and picks GPU rendering for the simulator if a
   test render works, else software rendering.
2. Runs the code's self-tests (a few minutes). If one fails it stops here and packs the logs.
3. Trains task A (with checkpoints at 2k, 5k, 10k steps), then task B, then 19 unlearning variants of
   task B, and evaluates each in the simulator. One worker runs per GPU.
4. Writes `summary.md` and packs everything into the `.tar.gz`.
