# Base-level tests

Known-answer checks of the code that can be tested without GR00T, its data or a cluster GPU: each
test builds small fake models with GR00T N1.6's module layout (`stubs.py` stands in for the GR00T
packages and makes a temporary runs/base directory; `fakes.py` holds the fake model and data). They need torch with CUDA, safetensors,
transformers and numpy, and take a minute on a laptop GPU. Run them after changing the files they cover:

```bash
python tests/test_drift_params.py    # vlu/drift/{metrics,weights,params}.py
python tests/test_drift_extract.py   # vlu/drift/{extract,compare}.py
python tests/test_ops.py             # NPO/NPO-RT/SSD/SalUn plans (vlu/ops.py), vlu/train.py with a mask, vlu/run_op.py checks
python tests/test_vlaforget.py       # vlu/{vlaforget,lora}.py, frozen overrides in vlu/model.py
python tests/test_stream_metrics.py  # vlu/stream_metrics.py (plain Python, no torch needed)
python tests/test_calibration.py     # local workers (no Slurm), calibration plan and report (plain Python)
python tests/test_completion.py      # workers and run_local.sh fail on failed / unfinished items (vlu/worker.py)
python tests/test_resume.py          # per-step preprocessing randomness, interrupted training continues identically
python tests/test_training_safety.py # non-finite losses, gradients, weights and checkpoints are rejected
python tests/test_rank_subspace.py   # drift subspaces of rank-deficient matrices (vlu/drift/metrics.py)
```

What they cannot check: anything that needs real GR00T weights, the dataset or the simulator. That is
what the cluster smoke test is for.
