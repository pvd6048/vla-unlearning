"""Split a LeRobot LIBERO suite into one dataset folder per task, without copying any data.

Each folder gets symlinks to the original data/ and videos/, a filtered meta/episodes.jsonl,
and copies of the other meta files. The *full-suite* stats.json / relative_stats.json are
copied as well, so every task uses identical action/state normalization (required for CL).

    python scripts/split_tasks.py data/libero_goal data/goal --replay-episodes 10

Writes:  <out>/tasks/task_<k>/   all demos of task k
         <out>/replay/task_<k>/  first N demos of task k (the ER buffer)
         <out>/tasks.json        task index -> instruction
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil


def write_subset(src, dst, episodes):
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for f in (src / "meta").iterdir():
        if f.is_file() and f.name != "episodes.jsonl":
            shutil.copy(f, dst / "meta" / f.name)
    with open(dst / "meta" / "episodes.jsonl", "w") as fh:
        for ep in episodes:
            fh.write(json.dumps(ep) + "\n")
    for d in ("data", "videos"):
        if not (dst / d).exists():
            (dst / d).symlink_to(src / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--replay-episodes", type=int, default=10)
    args = ap.parse_args()
    src = args.src.resolve()

    for f in ("modality.json", "stats.json", "relative_stats.json"):
        assert (src / "meta" / f).exists(), f"meta/{f} missing: run scripts/setup_server.sh first"

    with open(src / "meta" / "tasks.jsonl") as fh:
        tasks = {t["task_index"]: t["task"] for t in map(json.loads, fh)}
    index_of = {text: k for k, text in tasks.items()}

    by_task = defaultdict(list)
    with open(src / "meta" / "episodes.jsonl") as fh:
        for ep in map(json.loads, fh):
            by_task[index_of[ep["tasks"][0]]].append(ep)

    for k, eps in sorted(by_task.items()):
        write_subset(src, args.out / "tasks" / f"task_{k}", eps)
        write_subset(src, args.out / "replay" / f"task_{k}", eps[: args.replay_episodes])
        print(f"task {k}: {len(eps)} demos  {tasks[k]}")

    with open(args.out / "tasks.json", "w") as fh:
        json.dump(tasks, fh, indent=2)


if __name__ == "__main__":
    main()
