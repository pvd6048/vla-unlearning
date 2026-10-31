"""Shared setup of the base-level tests: they run without GR00T (and without its data or checkpoint).

Importing this module (first!) points VLU_RUNS_DIR / VLU_BASE_MODEL at a fresh temporary directory
with a tiny fake base checkpoint, and replaces the GR00T modules vlu imports with empty stand-ins.
The tests then drive vlu's code with small fake models that have GR00T N1.6's module layout.
Needs: torch (with CUDA: autocast("cuda") is used throughout), safetensors, transformers, numpy.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import types

import transformers  # the real package, imported before the stand-ins below

for _name in ("AutoModel", "AutoProcessor"):  # unused by the tests; some transformers builds cannot import them
    try:
        getattr(transformers, _name)
    except Exception:
        setattr(transformers, _name, None)

TMP = Path(tempfile.mkdtemp(prefix="vlu_tests_"))
os.environ["VLU_RUNS_DIR"] = str(TMP / "runs")
os.environ["VLU_BASE_MODEL"] = str(TMP / "base")
(TMP / "base").mkdir()
(TMP / "base" / "embodiment_id.json").write_text(json.dumps({"libero_panda": 2}))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for name in ["gr00t", "gr00t.data", "gr00t.data.dataset", "gr00t.data.dataset.lerobot_episode_loader",
             "gr00t.data.dataset.sharded_single_step_dataset", "gr00t.data.types", "gr00t.configs",
             "gr00t.configs.base_config", "gr00t.model", "av"]:
    sys.modules.setdefault(name, types.ModuleType(name))
_m = sys.modules
_m["gr00t.data.dataset.lerobot_episode_loader"].LeRobotEpisodeLoader = object
_m["gr00t.data.dataset"].lerobot_episode_loader = _m["gr00t.data.dataset.lerobot_episode_loader"]
_m["gr00t.data.dataset.sharded_single_step_dataset"].extract_step_data = None
_m["gr00t.data.types"].EmbodimentTag = _m["gr00t.data.types"].MessageType = None
_m["gr00t.configs.base_config"].Config = _m["gr00t.configs.base_config"].get_default_config = None
