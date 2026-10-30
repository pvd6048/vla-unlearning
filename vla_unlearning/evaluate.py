"""Success rate with NVIDIA's own LIBERO evaluation (policy server + rollout client), unchanged.

The server loads the checkpoint once; the client runs `episodes` rollouts per task in the
LIBERO simulator (its own Python env, created by NVIDIA's setup_libero.sh).
"""

import re
import shutil
import socket
import subprocess
import sys
import time

from vla_unlearning.paths import GROOT, LIBERO_PYTHON


def wait_for_port(port, server, timeout=900):
    start = time.time()
    while time.time() - start < timeout:
        if server.poll() is not None:
            raise RuntimeError("policy server exited; see its log")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(5)
    raise TimeoutError("policy server did not start")


def rollout(env_name, port, cfg):
    cmd = [
        str(LIBERO_PYTHON), "gr00t/eval/rollout_policy.py",
        "--env_name", env_name,
        "--n_episodes", str(cfg["episodes"]),
        "--n_envs", str(cfg["envs"]),
        "--max_episode_steps", str(cfg["max_steps"]),
        "--n_action_steps", str(cfg["action_steps"]),
        "--policy_client_host", "127.0.0.1",
        "--policy_client_port", str(port),
    ]  # fmt: skip
    out = subprocess.run(cmd, cwd=GROOT, capture_output=True, text=True, check=True).stdout
    video_dir = re.search(r"Video saved to:\s+(\S+)", out)
    if video_dir and not cfg.get("keep_videos", False):
        shutil.rmtree(video_dir.group(1), ignore_errors=True)  # the client always records videos
    return float(re.findall(r"success rate:\s+([\d.]+)", out)[-1])


def evaluate(ckpt, env_names, cfg, log, port=5555):
    """{env_name: success rate in [0, 1]} for one checkpoint."""
    server_cmd = [
        sys.executable, "gr00t/eval/run_gr00t_server.py",
        "--model-path", str(ckpt),
        "--embodiment-tag", "LIBERO_PANDA",
        "--use-sim-policy-wrapper",
        "--port", str(port),
    ]  # fmt: skip
    server = subprocess.Popen(server_cmd, cwd=GROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for_port(port, server)
        return {env: rollout(env, port, cfg) for env in env_names}
    finally:
        server.terminate()
        server.wait()
