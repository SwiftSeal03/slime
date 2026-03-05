#!/bin/bash
# Lightweight script for debug-train-only. Uses ray job submit so runtime env
# (PYTHONPATH, etc.) reaches Ray workers; direct python would fail with
# ModuleNotFoundError: megatron.training in workers.
#
# Lighter than full script: no pkill cleanup, optional ray start.

set -ex

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
HAS_NVLINK=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "/root/slime/scripts/models/qwen3-0.6B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/data/Qwen3-0.6B
   --ref-load /root/data/Qwen3-0.6B_torch_dist
   --save /root/data/Qwen3-0.6B_slime/
   --save-interval 25
)

ROLLOUT_ARGS=(
   --prompt-data /root/data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 1
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 0.8
   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /root/data/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --use-slime-router
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Ensure Ray is running (optional: start if not)
if ! ray status &>/dev/null; then
   echo "Starting Ray head node..."
   ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

# ray job submit injects runtime env into workers (required for megatron.training import)
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

cd /root/slime
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   --debug-train-only \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
