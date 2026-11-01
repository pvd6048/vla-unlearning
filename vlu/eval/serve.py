"""Serve a node's policy to the LIBERO client over ZMQ (GR00T's PolicyServer protocol).

    python -m vlu.eval.serve --run goal_s0 --node LA --weights final --port 5555 --seed 0

Runs in the main (GR00T) environment on the GPU; the simulator runs in a separate environment
(vlu/eval/libero_client.py), as in NVIDIA's LIBERO evaluation.
"""

import argparse

import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.policy import BasePolicy
from gr00t.policy.server_client import PolicyServer

from vlu.config import EMBODIMENT
from vlu.model import build_processor, gr00t_config, load_model
from vlu.nodes import Node


class NodePolicy(Gr00tPolicy):
    """Gr00tPolicy built from an in-memory model and processor instead of a checkpoint directory.
    The attributes set here are exactly the ones Gr00tPolicy.__init__ sets after loading."""

    def __init__(self, model, processor, strict: bool = True):
        BasePolicy.__init__(self, strict=strict)
        model.eval()
        # bf16, as NVIDIA's Gr00tPolicy. Training and the probe losses run under bf16 autocast, which
        # casts every linear layer's weights to bf16 per matmul, so this does not hide weight changes
        # that training-time outputs could see (only LayerNorm weights and biases lose precision).
        self.model = model.to(dtype=torch.bfloat16)
        processor.eval()
        self.processor = processor
        self.embodiment_tag = EmbodimentTag(EMBODIMENT)
        self.modality_configs = processor.get_modality_configs()[EMBODIMENT]
        self.collate_fn = processor.collator
        self.language_key = self.modality_configs["language"].modality_keys[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--node", required=True, help='op path; "" = base model (zero-shot)')
    parser.add_argument("--weights", default="final", help='"final" or a snapshot like step_002000')
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0, help="seed of the flow-matching sampling noise")
    args = parser.parse_args()

    node = Node.from_path(args.run, args.node)
    torch.manual_seed(args.seed)
    config = gr00t_config()
    policy = NodePolicy(load_model(config, node.weights_file(args.weights)), build_processor(config))
    print(f"serving {node} ({args.weights}) on port {args.port}", flush=True)
    PolicyServer(policy=Gr00tSimPolicyWrapper(policy), host="127.0.0.1", port=args.port).run()


if __name__ == "__main__":
    main()
