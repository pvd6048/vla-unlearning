"""Where the drift files of a node live (light module: no torch, no GR00T)."""

from pathlib import Path

from vlu.nodes import Node

# Layout of the features file written by vlu.drift.extract (3: float32 features; 2 stored float16, which
# overflowed to inf on diverged checkpoints).
FEATURES_VERSION = 3


def _suffix(weights: str) -> str:
    return "" if weights == "final" else f"_{weights}"


def features_path(node: Node, weights: str = "final") -> Path:
    """Probe-set features written by vlu.drift.extract (the base model's are in the run directory). The
    layout version is part of the name: a file of an older layout is not found, so its features item is not
    done (vlu.worker) and is extracted again, and no tool reads it."""
    return node.dir / "drift" / f"features_v{FEATURES_VERSION}{_suffix(weights)}.pt"


def fisher_path(node: Node, task: str, weights: str = "final") -> Path:
    """Diagonal Fisher of the flow-matching loss on one task, same tensors as a weights file."""
    return node.dir / "drift" / f"fisher_{task}{_suffix(weights)}.safetensors"
