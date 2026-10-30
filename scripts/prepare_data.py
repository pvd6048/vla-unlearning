"""Prepare and describe the LIBERO-Goal dataset (run once, from scripts/setup_cluster.sh; CPU only).

1. Writes meta/stats.json if missing (GR00T's own generate_stats, as its DatasetFactory does).
2. Checks that all ten LIBERO-Goal tasks are present and that A–E map onto them.
3. Writes $VLU_DATA_DIR/goal_report.json (per-task demo counts and lengths, state/action
   dimensions, action statistics) and the first camera frames of task A as PNGs, to compare by
   eye with simulator frames (image orientation is a classic silent bug).
"""

import json
from collections import defaultdict

import numpy as np
from PIL import Image

from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.data.types import EmbodimentTag

from vlu.config import DATA_DIR, EMBODIMENT, GOAL_DATASET
from vlu.data import load_demos
from vlu.model import goal_statistics, gr00t_config, modality_configs
from vlu.tasks import GOAL_TASK_NAMES, TASKS, instruction


def main():
    generate_stats(GOAL_DATASET)
    generate_rel_stats(GOAL_DATASET, EmbodimentTag(EMBODIMENT))

    episodes = [json.loads(line) for line in open(GOAL_DATASET / "meta" / "episodes.jsonl")]
    lengths = defaultdict(list)
    for ep in episodes:
        lengths[ep["tasks"][0].strip().lower()].append(ep["length"])
    expected = {name.replace("_", " ") for name in GOAL_TASK_NAMES}
    if set(lengths) != expected:
        raise ValueError(f"dataset tasks differ from LIBERO-Goal: extra {set(lengths) - expected}, "
                         f"missing {expected - set(lengths)}")

    config = gr00t_config()
    mcfg = modality_configs(config)
    demo_a = load_demos(mcfg, letters=("A",))["A"]
    first = demo_a.episodes[0]
    report = {
        "dataset": str(GOAL_DATASET),
        "total_demos": len(episodes),
        "tasks": {
            letter: {
                "name": name,
                "demos": len(lengths[instruction(letter)]),
                "mean_length": float(np.mean(lengths[instruction(letter)])),
                "min_length": int(np.min(lengths[instruction(letter)])),
            }
            for letter, name in TASKS.items()
        },
        "columns_of_one_episode": {
            col: (list(np.asarray(first[col].iloc[0]).shape) if col.startswith(("state.", "action.", "video.")) else "text")
            for col in first.columns
        },
        "instruction_of_one_episode": first[[c for c in first.columns if c.startswith("language.")][0]].iloc[0],
        "action_statistics": goal_statistics(config)["action"],
    }
    out = DATA_DIR / "goal_report.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report["tasks"], indent=2))
    print(f"full report: {out}")

    for col in [c for c in first.columns if c.startswith("video.")]:
        path = DATA_DIR / f"taskA_demo0_frame0_{col.split('.', 1)[1]}.png"
        Image.fromarray(np.asarray(first[col].iloc[0])).save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
