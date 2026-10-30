"""GPU check that our wrapper agrees with NVIDIA's model. Run once after setup, before experiments:

    python tests/test_groot.py <checkpoint_dir> data/goal/tasks/task_0
"""

import sys
import tempfile

import torch

from vla_unlearning import groot


def main(ckpt, task_dir):
    work = tempfile.mkdtemp()
    model, data = groot.load(ckpt, [task_dir], [], work, batch=4)
    model.train()
    batch = data["forget"]()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        l1, out = groot.loss(model, batch, seed=0)
        l2, _ = groot.loss(model, batch, seed=0)
        l3, _ = groot.loss(model, batch, seed=1)

    # 1. Our per-sample losses recombine to NVIDIA's own batch loss.
    mask_sum = out["action_mask"].sum((1, 2))
    combined = (l1 * mask_sum).sum() / mask_sum.sum()
    assert torch.allclose(combined, out["loss"], rtol=1e-3), (combined, out["loss"])
    # 2. A fixed seed fixes the noise; a different seed changes it.
    assert torch.equal(l1, l2), "same seed should give identical losses"
    assert not torch.equal(l1, l3), "different seeds should give different losses"
    # 3. Saving and reloading gives the same model.
    groot.save(model, ckpt, f"{work}/saved")
    reloaded, _ = groot.load(f"{work}/saved", [task_dir], [], f"{work}/w2", batch=4)
    reloaded.train()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        l4, _ = groot.loss(reloaded, batch, seed=0)
    assert torch.allclose(l1, l4, rtol=1e-3), (l1, l4)
    print("OK: wrapper matches NVIDIA's loss; seeding and save/reload work")


if __name__ == "__main__":
    main(*sys.argv[1:3])
