"""Exact global saliency ranks without a model-sized GPU argsort."""
import math

import numpy as np
import torch

from vlu.drift.weights import live_part


def ranked_masks(saliency, fraction, select, embodiment):
    """Select floor(N*fraction) live entries. Break cutoff ties in parameter/flat order.

    Algorithm 1 / generate_mask.py rank the absolute accumulated gradient globally.
    A CPU float32 partition uses O(N) host memory and avoids GPU int64 sort indices.
    Unused embodiment rows are outside the ranking universe in this VLA setting.
    """
    live = {n: live_part(n, s, embodiment) for n, s in saliency.items()}
    total = sum(s.numel() for s in live.values())
    count = math.floor(total * fraction)
    if select not in ("smallest", "largest") or not 0 < count < total:
        raise ValueError("Sharp saliency requires two nonempty partitions and a valid selection direction")
    pool = np.empty(total, dtype=np.float32)
    offset = 0
    for s in live.values():
        values = s.detach().float().cpu().reshape(-1).numpy()
        pool[offset:offset + values.size] = values
        offset += values.size
    if not np.isfinite(pool).all():
        raise ValueError("nonfinite Sharp saliency")
    rank = count - 1 if select == "smallest" else total - count
    pool.partition(rank)
    cutoff = float(pool[rank])
    del pool
    masks, selected = {}, 0
    for n, s in saliency.items():
        mask = torch.zeros(s.shape, dtype=torch.bool, device=s.device)
        view = live_part(n, mask, embodiment)
        view.copy_(live[n] < cutoff if select == "smallest" else live[n] > cutoff)
        selected += int(view.sum())
        masks[n] = mask
    remaining = count - selected
    for n, s in live.items():
        flat = s.reshape(-1)
        mask = live_part(n, masks[n], embodiment).view(-1)
        for start in range(0, flat.numel(), 1_000_000):
            if not remaining:
                break
            ties = flat[start:start + 1_000_000] == cutoff
            take = min(remaining, int(ties.sum()))
            if take:
                mask[start:start + ties.numel()] |= ties & (ties.cumsum(0) <= take)
                remaining -= take
    if remaining:
        raise RuntimeError("Sharp rank selection failed to fill the requested count")
    return masks
