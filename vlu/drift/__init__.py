"""Drift metrics between checkpoints (PLAN.md §4).

    params.py   parameter drift: norms, cosines, projections (e.g. onto a task vector), Fisher-weighted
                versions, singular-subspace overlap; per module and layer. Reads weight files only.
    extract.py  GPU job per checkpoint: hidden states and predicted velocities on a fixed probe set,
                and optionally the diagonal Fisher on some tasks.
    compare.py  representation drift (CKA, PCA shift, CKA between shifts) and policy drift (flow
                divergence, loss change, cosine between velocity changes) from extracted features.
    metrics.py  the math; weights.py, paths.py  checkpoint references, grouping, file locations.

Results are only meaningful against the controls of PLAN §4: a seed-twin drift (noise floor), a
random direction (params.py --random) and an unrelated op of similar size, all passed as more drifts.

Not implemented yet (PLAN §4): DiT cross-attention mass on text vs. image tokens, language
sensitivity under instruction swaps, and labelled behavioural failure modes.
"""
