"""Where things live. Everything is relative to the repo root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROOT = ROOT / "third_party" / "Isaac-GR00T"
LIBERO_PYTHON = GROOT / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
