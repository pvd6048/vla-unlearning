"""Where the drift files of a node live (light module: no torch, no GR00T)."""

from pathlib import Path

from vlu.nodes import Node


def _suffix(weights: str) -> str:
    return "" if weights == "final" else f"_{weights}"


def features_path(node: Node, weights: str = "final") -> Path:
    """Probe-set features written by vlu.drift.extract (the base model's are in the run directory)."""
    return node.dir / "drift" / f"features{_suffix(weights)}.pt"


def fisher_path(node: Node, task: str, weights: str = "final") -> Path:
    """Diagonal Fisher of the flow-matching loss on one task, same tensors as a weights file."""
    return node.dir / "drift" / f"fisher_{task}{_suffix(weights)}.safetensors"
