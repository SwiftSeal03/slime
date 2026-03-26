"""
Update distributed engines via wbridge. Same interface as update_weight_from_distributed.
"""

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle

from wbridge import WeightSender, WeightData
from wbridge.utils.megatron_utils import convert_to_wb


from slime.utils.distributed_utils import get_gloo_group

from .common import named_params_and_buffers
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
        """
        This function is called by the Actor after the rollout engines are created.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        server_infos = ray.get([engine.get_server_info.remote() for engine in self.rollout_engines])
        receiver_urls = [f"http://{host}:{port}" for host, port in server_infos]
        self.weight_sender = WeightSender(
            transfer_mode="gpu_direct",
            receiver_urls=receiver_urls,
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

        named_tensors = list(named_params_and_buffers(self.args, self.model))
        weight_data = convert_to_wb(self.args, self.model_name, named_tensors, self.quantization_config)
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
