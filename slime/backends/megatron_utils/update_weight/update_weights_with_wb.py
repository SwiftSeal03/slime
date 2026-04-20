"""
Update distributed engines via wbridge. Same interface as update_weight_from_distributed.
"""

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle

from wbridge.backend.sender import SenderArgs
from wbridge.frontend.megatron_adapter import WBMegatronAdapter
from wbridge.utils.distributed import get_local_ip, get_full_group_port

from slime.utils.distributed_utils import get_gloo_group

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
        self._weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self._wb: WBMegatronAdapter | None = None

    def connect_rollout_engines(self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle) -> None:
        """
        This function is called by the Actor after the rollout engines are created.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        server_infos = ray.get([engine.get_server_info.remote() for engine in self.rollout_engines])
        receiver_urls = [f"http://{host}:{port}" for host, port in server_infos]
        sender_args = SenderArgs(
            world_size=dist.get_world_size(),
            transfer_mode="gpu_direct",
            receiver_urls=receiver_urls,
            master_addr=get_local_ip(),
            master_port=get_full_group_port() + 1,
        )
        self._wb = WBMegatronAdapter(
            self.args.hf_checkpoint,
            list(self.model),
            dist.get_rank(),
            sender_args,
        )
        self._wb.connect()

    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1
        assert self._wb is not None, "connect_rollout_engines must be called before update_weights"

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

        self._wb.send_weights()

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
