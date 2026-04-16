#!/usr/bin/env python3
"""
Minimal Megatron actor: build the model, load a checkpoint, then run WeightBridge setup.

No Ray and no rollout dataset; this is a small harness for reproducing the same initialization
sequence as training: ``parse_args`` → ``init`` → ``initialize_model_and_optimizer`` (compare
``slime.backends.megatron_utils.actor.MegatronTrainRayActor``).

**torchrun:** ``WORLD_SIZE`` must match the process layout you configure (e.g. tensor/data/pipeline
parallelism). Example::

    source scripts/models/qwen3-0.6B.sh   # or pass equivalent Megatron model flags
    torchrun --standalone --nproc_per_node=2 scripts/minimal_megatron_load_checkpoint.py \\
      --load /path/to/Qwen3-0.6B_torch_dist \\
      --hf-checkpoint /path/to/Qwen3-0.6B \\
      --tensor-model-parallel-size 2 \\
      --sequence-parallel \\
      "${MODEL_ARGS[@]}"

You may add other slime/Megatron CLI flags (e.g. ``--no-load-optim``). Do not duplicate the
synthetic rollout-related flags this script prepends (``--num-rollout``, ``--rollout-batch-size``,
…).

**WeightBridge / LoadSpec:** After ``dist.barrier()``, this script always constructs
:class:`~wbridge.adapter.megatron_adapter.WBMegatronAdapter`. That runs Megatron-Bridge
``AutoBridge.from_hf_pretrained`` on ``--hf-checkpoint``, tries to load a cached :class:`~wbridge.utils.data.LoadSpec` from ``~/.cache/megatron/loadspec_rank{RANK}.json``, and on
cache miss or verification failure re-infers the spec via ``wbridge.utils.specgen.infer_load_spec``
and ``verify_load_spec`` (under ``torch.inference_mode()``), then writes the cache file. Inference
can take a long time on large models because it probes many weight mappings.

Wrapper shells (e.g. ``scripts/minimal_megatron_load_from_hf.sh``) may export
``WBRIDGE_INFER_LOAD_SPEC`` or ``SLIME_WBRIDGE_INFER_LOAD_SPEC`` for documentation consistency;
this Python entrypoint does not branch on those variables.

**``WeightSender``:** ``WBMegatronAdapter`` forwards trailing constructor arguments to
:class:`~wbridge.frontend.sender.WeightSender`. This script passes placeholder ``None`` values so you
can exercise LoadSpec inference without a live receiver; use real ``world_size``, transfer mode,
receiver URLs, and master address/port when wiring an actual send path.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

from slime.backends.megatron_utils.initialize import init, is_megatron_main_rank
from slime.backends.megatron_utils.model import initialize_model_and_optimizer
from slime.utils.arguments import parse_args
from slime.utils.distributed_utils import init_gloo_group
from slime.utils.logging_utils import configure_logger
from slime.utils.memory_utils import clear_memory
from slime.utils.reloadable_process_group import monkey_patch_torch_dist
from wbridge.adapter.megatron_adapter import WBMegatronAdapter


def _synthetic_slime_argv(world_size: int) -> list[str]:
    """Smallest argv slice that satisfies slime ``parse_args`` / ``slime_validate_args`` without rollout data."""
    return [
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        str(world_size),
        "--num-gpus-per-node",
        str(world_size),
        "--disable-rollout-global-dataset",
        "--num-rollout",
        "1",
        "--rollout-batch-size",
        "1",
        "--n-samples-per-prompt",
        "1",
        "--global-batch-size",
        "1",
        "--micro-batch-size",
        "1",
        "--advantage-estimator",
        "grpo",
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
    ]


def main() -> None:
    prog = sys.argv[0]
    user_argv = sys.argv[1:]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    sys.argv = [prog, *_synthetic_slime_argv(world_size), *user_argv]

    # Megatron validate_args requires this when tensor/context parallelism is enabled.
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    configure_logger()
    args = parse_args()

    monkey_patch_torch_dist()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=args.distributed_backend,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
    init_gloo_group()
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    init(args)
    model, _optimizer, _opt_param_scheduler, iteration = initialize_model_and_optimizer(args, role="actor")

    if is_megatron_main_rank():
        print(f"OK: loaded checkpoint, iteration={iteration}, num_model_chunks={len(model)}")

    dist.barrier()
    # Slime megatron.bridge plugin (optional; only imported when LoadSpec inference runs).
    adapter = WBMegatronAdapter(args.hf_checkpoint, model, args.rank, None, None, None, None, None)
    # dist.barrier()
    clear_memory()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
