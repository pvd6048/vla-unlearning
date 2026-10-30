"""Training batches: LIBERO-Goal demos grouped by task, and the batch stream of one op.

All demos of the tasks A–E are decoded into RAM once when a job starts (roughly 2 GB per task).
A batch is described by a list of "slots", one per sample; each slot lists the tasks that sample
may be drawn from. For example an L_D op with A and C retained uses

    {"train": [("D",)] * 8 + [("A", "C")] * 8}

i.e. 8 frames of D plus 8 replay frames (task chosen uniformly, then a uniform frame of it).
Batch number t of an op depends only on (seed, t), so a resumed job sees the same batches it
would have seen without the interruption.
"""

import av
import numpy as np
from torch.utils.data import Dataset

from gr00t.data.dataset import lerobot_episode_loader
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, MessageType

from vlu.config import EMBODIMENT, GOAL_DATASET, VIDEO_BACKEND
from vlu.tasks import TASKS, instruction

Slots = list[tuple[str, ...]]


def _frames_pyav(video_path: str, indices, video_backend=None, video_backend_kwargs=None) -> np.ndarray:
    """Frames `indices` of a video as RGB uint8 (N, H, W, 3), decoded with PyAV. The av wheel bundles
    FFmpeg with the AV1 decoder (dav1d) the LIBERO-Goal videos need, so no system FFmpeg is required.
    Same signature as GR00T's get_frames_by_indices, which it replaces (see episode_loader)."""
    with av.open(video_path) as container:
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    return frames[np.asarray(indices)]


def episode_loader(modality_configs: dict) -> LeRobotEpisodeLoader:
    """GR00T's episode loader for LIBERO-Goal, decoding videos with VIDEO_BACKEND (vlu/config.py).
    GR00T has no PyAV frame reader, so for "pyav" we swap its frame reader for ours."""
    if VIDEO_BACKEND == "pyav":
        lerobot_episode_loader.get_frames_by_indices = _frames_pyav
    return LeRobotEpisodeLoader(GOAL_DATASET, modality_configs, video_backend=VIDEO_BACKEND)


class TaskDemos:
    """Every demo of one task, as GR00T episode DataFrames, plus the list of usable
    (episode, step) pairs: steps with a full action chunk ahead of them (no padding, as in GR00T)."""

    def __init__(self, letter: str, episodes: list, action_horizon: int):
        self.letter = letter
        self.episodes = episodes
        self.index = [(e, s) for e, df in enumerate(episodes) for s in range(len(df) - action_horizon + 1)]

    def __repr__(self) -> str:
        return f"TaskDemos({self.letter}: {len(self.episodes)} demos, {len(self.index)} frames)"


def load_demos(modality_configs: dict, letters=tuple(TASKS)) -> dict[str, TaskDemos]:
    loader = episode_loader(modality_configs)
    action_horizon = len(modality_configs["action"].delta_indices)
    demos = {}
    for letter in letters:
        positions = [
            i
            for i, meta in enumerate(loader.episodes_metadata)
            if meta["tasks"][0].strip().lower() == instruction(letter)
        ]
        if not positions:
            raise ValueError(f"No demos of task {letter} ({instruction(letter)!r}) in {GOAL_DATASET}")
        demos[letter] = TaskDemos(letter, [loader[i] for i in positions], action_horizon)
        print(f"loaded {demos[letter]}", flush=True)
    return demos


def make_sample(processor, modality_configs: dict, demos: TaskDemos, episode: int, step: int) -> dict:
    """One processed training sample, exactly as GR00T's ShardedSingleStepDataset builds it."""
    step_data = extract_step_data(demos.episodes[episode], step, modality_configs, EmbodimentTag(EMBODIMENT))
    return processor([{"type": MessageType.EPISODE_STEP.value, "content": step_data}])


class OpBatches(Dataset):
    """Item t is batch number t of the op: {part name: model inputs}. Used with a DataLoader with
    batch_size=None, so worker processes build (and augment) whole batches in parallel."""

    def __init__(self, demos, parts: dict[str, Slots], processor, modality_configs, seed: int, num_steps: int):
        self.demos, self.parts, self.processor = demos, parts, processor
        self.modality_configs, self.seed, self.num_steps = modality_configs, seed, num_steps

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, step: int) -> dict[str, dict]:
        rng = np.random.default_rng([self.seed, step])
        batch = {}
        for name, slots in self.parts.items():
            samples = []
            for tasks in slots:
                task = self.demos[tasks[rng.integers(len(tasks))]]
                episode, t = task.index[rng.integers(len(task.index))]
                samples.append(make_sample(self.processor, self.modality_configs, task, episode, t))
            batch[name] = dict(self.processor.collator(samples)["inputs"])
        return batch


def fixed_batch(demos: TaskDemos, n: int, processor, modality_configs) -> dict:
    """n evenly spread frames of one task, processed without augmentation: a fixed probe batch for
    tracking that task's flow-matching loss during training."""
    was_training = processor.training
    processor.eval()
    picks = np.linspace(0, len(demos.index) - 1, n).round().astype(int)
    samples = [make_sample(processor, modality_configs, demos, *demos.index[i]) for i in picks]
    if was_training:
        processor.train()
    return dict(processor.collator(samples)["inputs"])
