"""LIBERO-Goal rollouts against a running policy server (vlu/eval/serve.py).

    python vlu/eval/libero_client.py --port 5555 --run goal_s0 --node LA --tasks A,B

Runs in the LIBERO simulator environment (Isaac-GR00T's libero_uv venv), CPU only. Protocol:
episode i of a task starts from LIBERO's fixed initial state i (i = 0..episodes-1), waits
`wait_steps` no-op steps for objects to settle, then runs the policy for at most `max_steps` env
steps, executing the first `n_action_steps` actions of every predicted chunk. Success = LIBERO's
check_success at any step. Writes <node dir>/eval/<weights>_seed<seed>/<task>.json per task and
skips tasks already written, so a killed eval job can simply be rerun.

Observation/action conversion is NVIDIA's LiberoEnv (180° image flip, axis-angle state, gripper
sign), so it matches how the training data was converted.
"""

import argparse
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root, for vlu.*
from vlu.config import RECIPE  # noqa: E402
from vlu.nodes import Node  # noqa: E402
from vlu.tasks import TASKS, instruction  # noqa: E402

LIBERO_NOOP = [0.0] * 6 + [-1.0]  # raw LIBERO action: no motion, gripper open


def make_env(letter: str):
    """NVIDIA's LiberoEnv for a LIBERO-Goal task, plus that task's fixed initial states."""
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
    from libero.libero import benchmark
    from libero.libero.utils import get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task_id = suite.get_task_names().index(TASKS[letter])
    task = suite.get_task(task_id)
    if task.language != instruction(letter):
        raise ValueError(f"LIBERO instruction {task.language!r} != training instruction {instruction(letter)!r}")
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return LiberoEnv(task_bddl_file=bddl, task_description=task.language), suite.get_task_init_states(task_id)


def to_policy_obs(obs: dict) -> dict:
    """One env observation -> the batched (B=1, T=1) format of Gr00tSimPolicyWrapper."""
    out = {}
    for key, value in obs.items():
        if key.startswith("video."):
            out[key] = np.asarray(value, dtype=np.uint8)[None, None]
        elif key.startswith("state."):
            out[key] = np.asarray(value, dtype=np.float32)[None, None]
        else:  # language
            out[key] = (value,)
    return out


_worker = {}  # per-process cache: policy client and one env per task


def run_episode(job: tuple[str, int, int, bool, str | None]) -> dict:
    letter, init_index, port, record, video_dir = job
    from gr00t.policy.server_client import PolicyClient

    if "client" not in _worker:
        _worker["client"] = PolicyClient(host="127.0.0.1", port=port)
    if letter not in _worker:
        _worker[letter] = make_env(letter)
    client, (env, init_states) = _worker["client"], _worker[letter]

    env.reset()
    env._env.set_init_state(init_states[init_index])
    for _ in range(RECIPE.eval_wait_steps):
        raw_obs, _, _, _ = env._env.step(LIBERO_NOOP)
    obs = env._process_observation(raw_obs)

    frames, success, steps = [], False, 0
    while steps < RECIPE.eval_max_steps and not success:
        action, _ = client.get_action(to_policy_obs(obs))
        for k in range(RECIPE.n_action_steps):
            obs, _, _, _, info = env.step({key: value[0, k] for key, value in action.items()})
            steps += 1
            if record:
                frames.append(obs["video.image"])
            if info["success"]:
                success = True
                break
            if steps >= RECIPE.eval_max_steps:
                break

    if record and video_dir:
        write_video(Path(video_dir) / f"{letter}_init{init_index:02d}_{'ok' if success else 'fail'}.mp4", frames)
    return {"task": letter, "init_index": init_index, "success": success, "steps": steps}


def write_video(path: Path, frames: list, fps: int = 20) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")))
        container.mux(stream.encode())


def wait_for_server(port: int) -> None:
    """Block until the server (which first loads a 3B model) answers. GR00T's client has no receive
    timeout, so this waits indefinitely; the eval job script kills us if the server dies."""
    from gr00t.policy.server_client import PolicyClient

    PolicyClient(host="127.0.0.1", port=port).ping()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--node", required=True, help='op path; "" = base model')
    parser.add_argument("--weights", default="final", help="only names the output directory")
    parser.add_argument("--seed", type=int, default=0, help="the server's seed; only names the output directory")
    parser.add_argument("--tasks", required=True, help="task letters, e.g. A,B,C")
    parser.add_argument("--episodes", type=int, default=RECIPE.eval_episodes)
    parser.add_argument("--workers", type=int, default=4, help="parallel simulator processes")
    parser.add_argument("--videos", type=int, default=0, help="record the first N episodes of each task")
    args = parser.parse_args()

    out = Node.from_path(args.run, args.node).eval_dir(args.weights, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in args.tasks.split(",") if not (out / f"{t}.json").exists()]
    if not todo:
        print(f"all tasks already evaluated in {out}")
        return
    wait_for_server(args.port)

    jobs = [(t, i, args.port, i < args.videos, str(out / "videos")) for t in todo for i in range(args.episodes)]
    results = {t: [] for t in todo}
    start = time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_episode, jobs):
            results[r["task"]].append(r)
            done = results[r["task"]]
            print(f"{r['task']} init {r['init_index']:2d}: {'success' if r['success'] else 'fail'} "
                  f"({len(done)}/{args.episodes}, {time.time() - start:.0f} s)", flush=True)
            if len(done) == args.episodes:
                done.sort(key=lambda e: e["init_index"])
                sr = float(np.mean([e["success"] for e in done]))
                summary = {"task": r["task"], "instruction": instruction(r["task"]), "success_rate": sr,
                           "episodes": done, "wall_s_since_start": time.time() - start}
                (out / f"{r['task']}.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(f"== {r['task']}: SR {sr:.2%}", flush=True)


if __name__ == "__main__":
    main()
