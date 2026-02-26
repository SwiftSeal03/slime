import socket
import threading
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from queue import Queue

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.timestamp import timestamp

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers


def _copy_chunk_to_pinned(
    converted_named_tensors: list[tuple[str, torch.Tensor]],
    stream: torch.cuda.Stream,
) -> list[tuple[str, torch.Tensor]]:
    """Copy a chunk of (name, tensor) to pinned CPU memory for faster transfer.
    Uses the given CUDA stream for non-blocking GPU->CPU copy. Caller records one
    event after all chunks are launched.
    """
    pinned: list[tuple[str, torch.Tensor]] = []
    with torch.cuda.stream(stream):
        for name, t in converted_named_tensors:
            src = t.data if hasattr(t, "data") else t
            buf = torch.empty(
                src.shape, dtype=src.dtype, device="cpu", pin_memory=True
            )
            buf.copy_(src, non_blocking=True)
            pinned.append((name, buf))
    return pinned


class UpdateWeightFromHost:
    """
    Update distributed engines via NCCL. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
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
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self._timestamp_lock = threading.Lock()
        self._chunk_queue: Queue | None = None
        self._sender_thread: threading.Thread | None = None
        self._offload_stream: torch.cuda.Stream | None = None

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Create NCCL "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock

        # For TP:
        #   1. AllGather paramters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}-cpu"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_host(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_host(
                self.args, self._group_name, rollout_engines
            )
            if self._chunk_queue is None:
                self._chunk_queue = Queue()
                self._offload_stream = torch.cuda.Stream()
                self._sender_thread = threading.Thread(
                    target=self._sender_thread_loop,
                    daemon=False,
                )
                self._sender_thread.start()

    def _sender_thread_loop(self) -> None:
        """
        Long-lived loop: consume from self._chunk_queue. Chunks (list) extend accumulated;
        sentinel (tuple) means end: sync event, send accumulated if non-empty, update pbar, timestamp, continue.
        Need to send is determined by type only: tuple -> sentinel, list -> chunk.
        """
        accumulated: list[tuple[str, torch.Tensor]] = []
        while True:
            item = self._chunk_queue.get()
            if isinstance(item, tuple):
                pbar, final_event, rollout_id = item
                final_event.synchronize()
                with self._timestamp_lock:
                    timestamp(self.args, f"weight_updates_offload_end {rollout_id}")
                if accumulated:
                    while not ray.get(self.rollout_engine_lock.acquire.remote()):
                        time.sleep(0.1)
                    refs = update_weights_from_host(
                        self._group_name,
                        self._model_update_groups,
                        self.weight_version,
                        self.rollout_engines,
                        accumulated,
                    )
                    ray.get(refs)
                    ray.get(self.rollout_engine_lock.release.remote())
                pbar.update(1)
                with self._timestamp_lock:
                    timestamp(self.args, f"weight_updates_end {rollout_id}")
                # if dist.get_rank() == 0:
                #     ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
                accumulated = []
                continue
            accumulated.extend(item)


    @torch.no_grad()
    def update_weights(self, rollout_id: int | None = None) -> None:
        """
        Pause → flush → non-expert (TP) → expert (EP) → continue. Progress on PP source.
        Reuses instance-level _chunk_queue and long-lived _sender_thread.
        """
        if rollout_id is not None:
            with self._timestamp_lock:
                timestamp(self.args, f"weight_updates_begin {rollout_id}")

        self.weight_version += 1

        # if dist.get_rank() == 0:
        #     ray.get([e.pause_generation.remote() for e in self.rollout_engines])
        #     ray.get([e.flush_cache.remote() for e in self.rollout_engines])
        # dist.barrier(group=get_gloo_group())
        

        buffer_size = 0
        converted_named_tensors = []
        # non expert params
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_host(
                name, param, converted_named_tensors, buffer_size
            )

        if converted_named_tensors:
            self._update_bucket_weights_from_host(converted_named_tensors)

        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        named_tensors = []
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_host(
                name, param, named_tensors, buffer_size
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_host(named_tensors)

        if self._is_pp_src_rank and self._chunk_queue is not None:
            if rollout_id is not None:
                with self._timestamp_lock:
                    timestamp(self.args, f"weight_updates_gather_end {rollout_id}")
            final_event = self._offload_stream.record_event()
            self._chunk_queue.put((pbar, final_event, rollout_id))

    def _update_weight_from_host(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
    ) -> int | None:
        """
        Non-expert: gather TP → rm pad → HF → buffer (flush if full). All gather, PP source buffers.
        Returns updated bytes on source, None on non-source.
        """
        param = all_gather_param(name, param)
        if not self._is_pp_src_rank:
            return

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            self._update_bucket_weights_from_host(converted_named_tensors)
            buffer_size = 0
        converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_host(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
    ) -> int:
        """
        Expert: gather TP → rm pad → buffer. EP gather + HF deferred. Threshold × EP size.
        """
        param = all_gather_param(name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            self._update_expert_bucket_weights_from_host(named_tensors)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_host(
        self, named_tensors: list[tuple[str, torch.Tensor]]
    ) -> None:
        """
        Gather EP → HF → broadcast. Clears buffer.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        self._update_bucket_weights_from_host(converted_hf_tensors)

    def _update_bucket_weights_from_host(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]]
    ) -> None:
        """
        Offload chunk to pinned CPU on the shared _offload_stream and hand to sender thread.
        One event and pbar are pushed only with the sentinel at the end.
        """
        if not self._is_pp_src_rank:
            converted_named_tensors.clear()
            return
        pinned_chunk = _copy_chunk_to_pinned(converted_named_tensors, self._offload_stream)
        self._chunk_queue.put(pinned_chunk)
        converted_named_tensors.clear()


def connect_rollout_engines_from_host(
    args: Namespace, group_name: str, rollout_engines: Sequence[ActorHandle]
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.
    """
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = len(rollout_engines) * args.rollout_num_gpus_per_engine + 1

    refs = [
        engine.init_weights_update_group.remote(
            master_address,
            master_port,
            i * args.rollout_num_gpus_per_engine + 1,
            world_size,
            group_name,
            backend="gloo",
        )
        for i, engine in enumerate(rollout_engines)
    ]
    model_update_groups = init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_host(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def update_weights_from_host(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
        )
        for engine in rollout_engines
    ]

    handles = []
    for _, param in converted_named_tensors:
        assert param.is_cpu, "tensor is not on CPU"
        handles.append(dist.broadcast(param, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs
