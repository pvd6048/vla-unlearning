"""Reading the trainable weights of any checkpoint tensor by tensor, and grouping them by module/layer.

A checkpoint reference is written  [run:]path[@weights]
    LA/LB/LC                 node of the default run, final weights
    goal_s1:LA/LB@step_002000  node of another run, a snapshot
    base  (or an empty path) the base GR00T-N1.6-3B checkpoint
A node file holds only the trainable parameters; for the base model they are read from the HF
checkpoint shards under the same names, so no model has to be built.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re

from safetensors import safe_open
import torch

from vlu.config import BASE_MODEL, EMBODIMENT
from vlu.nodes import Node


@dataclass(frozen=True)
class Ref:
    run: str
    path: str  # "" = base model
    weights: str = "final"

    @classmethod
    def parse(cls, text: str, default_run: str) -> "Ref":
        run, _, rest = text.rpartition(":")
        path, _, weights = rest.partition("@")
        path = "" if path in ("", "base") else path
        return cls(run or default_run, path, weights or "final")

    def __str__(self) -> str:
        return f"{self.run}:{self.path or 'base'}" + (f"@{self.weights}" if self.weights != "final" else "")

    @property
    def node(self) -> Node:
        return Node.from_path(self.run, self.path)

    @property
    def is_base(self) -> bool:
        return not self.path


def base_checkpoint_dir() -> Path:
    if Path(BASE_MODEL).is_dir():
        return Path(BASE_MODEL)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(BASE_MODEL, local_files_only=True))


def embodiment_index() -> int:
    """Row of LIBERO's copy in GR00T's embodiment-specific projector weights."""
    return json.loads((base_checkpoint_dir() / "embodiment_id.json").read_text())[EMBODIMENT]


# Projector weights are stacked per embodiment (dim 0 = 32 robots); only LIBERO's row ever changes.
_PER_EMBODIMENT = re.compile(r"^action_head\.(state_encoder|action_encoder|action_decoder)\.")
# Module paths of GR00T N1.6 (shared with vlu/vlaforget.py).
VISION_LAYER = re.compile(r"^backbone\.model\.vision_model\.vision_model\.encoder\.layers\.(\d+)\.")
LLM_LAYER = re.compile(r"^backbone\.model\.language_model\.model\.layers\.(\d+)\.")


def live_part(key: str, t: torch.Tensor, embodiment: int) -> torch.Tensor:
    """The part of a parameter (or of a per-parameter tensor such as a gradient or Fisher) that can
    change when training LIBERO: LIBERO's row of the per-embodiment projector stacks, else all of it."""
    if _PER_EMBODIMENT.match(key) and t.ndim >= 2 and t.shape[0] == 32:
        return t[embodiment]
    return t


class WeightReader:
    """get(ref, key) -> float32 tensor; per-embodiment projector weights are reduced to LIBERO's row."""

    def __init__(self, device: str):
        self.device = device
        self._files: dict[Path, object] = {}
        self._base_map: dict[str, Path] | None = None
        self._embodiment = embodiment_index()

    def _open(self, path: Path):
        if path not in self._files:
            self._files[path] = safe_open(str(path), framework="pt", device="cpu")
        return self._files[path]

    def _file_of(self, ref: Ref, key: str) -> Path:
        if not ref.is_base:
            path = ref.node.weights_file(ref.weights)
            if key in self._open(path).keys():
                return path
            # A frozen weight this node does not override (vlu/model.py: frozen overrides) = base value.
        if self._base_map is None:
            root = base_checkpoint_dir()
            index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
            self._base_map = {k: root / f for k, f in index.items()}
        return self._base_map[key]

    def keys(self, refs: list[Ref]) -> list[str]:
        """Parameter names stored in the nodes' weights files (trainable ones and frozen overrides);
        the base model has no such list."""
        return sorted(set().union(*(self._open(r.node.weights_file(r.weights)).keys() for r in refs if not r.is_base)))

    def get(self, ref: Ref, key: str) -> torch.Tensor:
        return self.reduce(key, self._open(self._file_of(ref, key)).get_tensor(key))

    def reduce(self, key: str, t: torch.Tensor) -> torch.Tensor:
        """To float32 on the device; per-embodiment projector weights reduced to LIBERO's row. Also
        for other per-parameter files with the same layout (e.g. a Fisher file)."""
        return live_part(key, t.to(self.device, torch.float32), self._embodiment)


# --- Grouping ------------------------------------------------------------------------------------

_DIT_BLOCK = re.compile(r"^action_head\.model\.transformer_blocks\.(\d+)\.")
# GR00T N1.6's AlternateVLDiT (attend_text_every_n_blocks=2): blocks cycle cross-attention to the
# text tokens, self-attention, cross-attention to the image tokens, self-attention.
DIT_BLOCK_KIND = {0: "x-text", 1: "self", 2: "x-image", 3: "self"}


def module_of(key: str) -> str:
    """vision (vision tower), vl_projector (vision -> LLM projector), vlm (LLM layers), projector
    (state/action encoders and decoder, position embedding, VL layer norm) or dit. The vision tower and
    vl_projector are frozen in learning; only VLA-Forget changes them (frozen overrides)."""
    if key.startswith("backbone.model.vision_model."):
        return "vision"
    if key.startswith("backbone.model.mlp1."):
        return "vl_projector"
    if key.startswith("backbone."):
        return "vlm"
    if key.startswith("action_head.model."):
        return "dit"
    return "projector"


def layer_of(key: str) -> str:
    if m := VISION_LAYER.match(key):
        return f"vision.layer{int(m[1]):02d}"
    if key.startswith("backbone.model.mlp1."):
        return "vl_projector"
    if key.startswith("backbone.model.vision_model."):
        return "vision.other"
    if m := LLM_LAYER.match(key):
        return f"vlm.layer{int(m[1]):02d}"
    if m := _DIT_BLOCK.match(key):
        i = int(m[1])
        return f"dit.block{i:02d}.{DIT_BLOCK_KIND[i % 4]}"
    parts = key.split(".")
    if key.startswith("action_head.model."):
        return f"dit.{parts[2]}"  # timestep_encoder, proj_out_1, proj_out_2
    if key.startswith("action_head."):
        return f"projector.{parts[1]}"
    return "vlm." + ".".join(parts[3:-1])  # e.g. the final norm of the language model


def groups_of(key: str) -> list[tuple[str, str]]:
    """(level, group) pairs a parameter tensor counts towards."""
    return [("global", "all"), ("module", module_of(key)), ("layer", layer_of(key))]
