"""Drift metrics between checkpoints (PLAN.md §4).

    params.py   parameter drift: norms, cosines, projections (e.g. onto a task vector), Fisher-weighted
                versions, singular-subspace overlap; per module and layer. Reads weight files only.
    extract.py  GPU job per checkpoint: hidden states and predicted velocities on a fixed probe set,
                and optionally the diagonal Fisher on some tasks.
    compare.py  from extracted features: representation drift (CKA, PCA shift, CKA between shifts),
                policy drift (flow divergence, loss change, cosine between velocity changes) and
                cross-modal alignment (image-text similarity per LLM layer, size of the DiT's text vs.
                image cross-attention updates, language sensitivity under instruction swaps).
    metrics.py  the math; weights.py, paths.py  checkpoint references, grouping, file locations.

Results are only meaningful against the controls of PLAN §4: a seed-twin drift (noise floor), a
random direction (params.py --random) and an unrelated op of similar size, all passed as more drifts.

GR00T N1.6's DiT routes attention by design (x-text blocks see only text tokens, x-image blocks
only image tokens), so PLAN §4's "cross-attention mass on instruction vs. image" is measured as the
size of each block's attention update instead.

Not implemented yet (PLAN §4): labelled behavioural failure modes (wrong object, wrong goal, freeze).
"""
