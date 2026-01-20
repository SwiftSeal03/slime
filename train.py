import time
import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.tracking_utils import init_tracking


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # always update weight first so that sglang has the loaded weights from training.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def onload_rollout():
        if args.offload_rollout:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0:
            ray.get(rollout_manager.eval.remote(rollout_id))

        iter_start = time.time()
        t_generate = time.time()
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        t_offload_rollout = time.time()
        
        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())
        t_train = time.time()

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            t_actor_train = time.time()
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            t_critic_wait = time.time()
            ray.get(critic_train_handle)
        else:
            t_actor_train = time.time()
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            t_critic_wait = time.time()
        t_save = time.time()

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
                actor_model.save_model(rollout_id)
            if args.use_critic:
                critic_model.save_model(rollout_id)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
        t_offload_train = time.time()

        offload_train()
        t_onload_rollout = time.time()
        onload_rollout()
        t_update_weights = time.time()
        actor_model.update_weights()
        t_onload_additional = time.time()

        if args.offload_rollout:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))
        iter_end = time.time()

        # Compute timing intervals
        timing_dict = {
            'generate_time': t_offload_rollout - t_generate,
            'offload_rollout_time': (t_train - t_offload_rollout) if args.offload_rollout else 0.0,
            'actor_train_time': (t_critic_wait - t_actor_train) if (not args.use_critic or rollout_id >= args.num_critic_only_steps) else 0.0,
            'critic_train_launch_time': (t_actor_train - t_train) if args.use_critic else 0.0,
            'critic_train_wait_time': (t_save - t_critic_wait) if args.use_critic else 0.0,
            'critic_train_time': (t_save - t_train) if args.use_critic else 0.0,
            'save_time': (t_offload_train - t_save),
            'offload_train_time': t_onload_rollout - t_offload_train,
            'onload_rollout_time': t_update_weights - t_onload_rollout,
            'update_weights_time': t_onload_additional - t_update_weights,
            'onload_rollout_additional_time': (iter_end - t_onload_additional) if args.offload_rollout else 0.0,
            'total_iteration_time': iter_end - iter_start,
        }
        print(f"real-perf {rollout_id}: {timing_dict}")

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
