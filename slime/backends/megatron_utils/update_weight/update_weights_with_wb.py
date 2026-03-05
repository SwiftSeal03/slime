"""
Update distributed engines via wbridge. Same interface as update_weight_from_distributed.
"""

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from wbridge import WeightSender, WeightData

from slime.utils.distributed_utils import get_gloo_group

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers
from .update_weight_from_distributed import post_process_weights

class UpdateWeightWithWB:
    """
    Update distributed engines via wbridge. Same interface as UpdateWeightFromDistributed.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        
    def connect_rollout_engines(self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self.weight_sender = WeightSender(
            transfer_mode="gpu_direct",
            receiver_urls=[
                f"tcp://{ip}:{port}" for ip, port in zip(
                    self.rollout_engines.get_host_ip_addresses(), 
                    self.rollout_engines.get_host_ports()
                )
            ],
        )

    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        converted_named_tensors = []
        for name, param in named_params_and_buffers(self.args, self.model):
            converted_named_tensors.append(convert_to_hf(self.args, self.model_name, name, param, self.quantization_config))
        weight_data = convert_to_wb_format(converted_named_tensors)
        self.weight_sender.send(weight_data)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())


def _compute_shard_intervals(name: str, param: torch.nn.Parameter) -> list[tuple[int, int]]:
    """
    Compute shard intervals for the local tensor on each dimension of the original tensor.
    Mirrors the partitioning logic from all_gather_param in common.py.
    Returns list of (start, end) per dimension.
    """
    shape = list(param.shape)
    if "expert_bias" in name:
        return [(0, dim, dim) for dim in shape]

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return [(0, dim, dim) for dim in shape]

    if ".experts." in name:
        tp_rank = mpu.get_expert_tensor_parallel_rank()
        tp_size = mpu.get_expert_tensor_parallel_world_size()
    else:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        
    partition_dim = param.partition_dim
    if "linear_fc1.weight" in name:
        partition_dim = 1
        shape = (2, shape[0] // 2, shape[1])
    if "linear_fc2.weight" in name:
        partition_dim = 1

    shard = []
    for d in range(len(shape)):
        s = shape[d]
        if d == partition_dim:
            l = tp_rank * s
            r = l + s
            w = s * tp_size
            shard.append((l, r, w))
        else:
            shard.append((0, s, s))
    return shard


def convert_to_wb_format(converted_named_tensors: list[tuple[str, torch.nn.Parameter]]) -> WeightData:
    state_dict = {}
    for name, param in converted_named_tensors:
        shard = _compute_shard_intervals(name, param)
        state_dict[name] = {
            "metadata": {
                "shard": shard,
                "dtype": param.dtype,
            },
            "data": param.data.view(-1),
        }
    return WeightData(state_dict)