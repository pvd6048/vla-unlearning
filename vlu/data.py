"""Training batches: LIBERO-Goal demos grouped by task, and the batch stream of one op.

All demos of the tasks A–E are decoded into RAM once when a job starts (roughly 2 GB per task).
A batch is described by a list of "slots", one per sample. A slot is either a tuple of tasks the
sample may come from (task chosen uniformly, then a uniform frame of it) or a ReplayBuffer (a
uniform frame of a fixed replay buffer). For example an L_D op with A and C retained uses

    {"train": [("D",)] * 16 + [ReplayBuffer(("A", "C"), 1000)] * 16}

i.e. 16 frames of D plus 16 frames from the replay buffer of A and C (Liu et al. 2026's ER).
Batch number t of an op depends only on (seed, t), so a resumed job sees the same batches it
would have seen without the interruption.
"""

import dataclasses
from dataclasses import dataclass

import av
import numpy as np
from torch.utils.data import Dataset

from gr00t.data.dataset import lerobot_episode_loader
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, MessageType

from vlu.config import EMBODIMENT, GOAL_DATASET, VIDEO_BACKEND
from vlu.tasks import TASKS, instruction

Slots = list  # of tuple[str, ...] (tasks) or ReplayBuffer


@dataclass(frozen=True)
class ReplayBuffer:
    """A slot drawing one frame from the replay buffer of `tasks`: per task, `per_task` frames chosen
    once from (run seed, task) and reused by every op of the run (as Liu et al.'s continual-GR00T:
    n_memories random transitions per previous task, fixed). per_task=None: all frames."""

    tasks: tuple[str, ...]
    per_task: int | None

    def frames(self, demos: dict, run_seed: int) -> list[tuple[str, int, int]]:
        out = []
        for letter in self.tasks:
            index = demos[letter].index
            n = len(index) if self.per_task is None else min(self.per_task, len(index))
            rng = np.random.default_rng([run_seed, ord(letter)])
            out += [(letter, *index[i]) for i in sorted(rng.choice(len(index), n, replace=False))]
        return out


class Frames(list):
    """A batch part drawn from an explicit list of frames (task letter, episode, step) instead of
    from tasks: `per_batch` frames per batch, uniformly with replacement (VLA-Forget's boundary set)."""

    def __init__(self, frames: list[tuple[str, int, int]], per_batch: int):
        super().__init__(frames)
        self.per_batch = per_batch


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


def make_sample(processor, modality_configs: dict, demos: TaskDemos, episode: int, step: int, text: str | None = None) -> dict:
    """One processed training sample, exactly as GR00T's ShardedSingleStepDataset builds it.
    `text` replaces the instruction (used to measure language sensitivity)."""
    step_data = extract_step_data(demos.episodes[episode], step, modality_configs, EmbodimentTag(EMBODIMENT))
    if text is not None:
        step_data = dataclasses.replace(step_data, text=text)
    return processor([{"type": MessageType.EPISODE_STEP.value, "content": step_data}])


class OpBatches(Dataset):
    """Item t is batch number t of the op: {part name: model inputs}. Used with a DataLoader with
    batch_size=None, so worker processes build (and augment) whole batches in parallel."""

    def __init__(self, demos, parts: dict[str, Slots], processor, modality_configs, seed: int, num_steps: int,
                 run_seed: int | None = None):
        """seed: of this op's batches; run_seed: of the run, fixes the replay buffers (needed if a part uses one)."""
        self.demos, self.parts, self.processor = demos, parts, processor
        self.modality_configs, self.seed, self.num_steps = modality_configs, seed, num_steps
        buffers = {slot for slots in parts.values() if not isinstance(slots, Frames) for slot in slots
                   if isinstance(slot, ReplayBuffer)}
        if buffers and run_seed is None:
            raise ValueError("a replay buffer needs the run seed")
        self.buffers = {b: b.frames(demos, run_seed) for b in buffers}

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, step: int) -> dict[str, dict]:
        rng = np.random.default_rng([self.seed, step])
        batch = {}
        for name, slots in self.parts.items():
            if isinstance(slots, Frames):
                picks = [slots[i] for i in rng.integers(len(slots), size=slots.per_batch)]
            else:
                picks = []
                for slot in slots:
                    if isinstance(slot, ReplayBuffer):
                        frames = self.buffers[slot]
                        picks.append(frames[rng.integers(len(frames))])
                        continue
                    letter = slot[rng.integers(len(slot))]
                    episode, t = self.demos[letter].index[rng.integers(len(self.demos[letter].index))]
                    picks.append((letter, episode, t))
            samples = [make_sample(self.processor, self.modality_configs, self.demos[letter], episode, t)
                       for letter, episode, t in picks]
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
