#!/bin/bash
# torchrun: build Megatron actor weights from a HuggingFace model dir (not Megatron dist ckpt).
#
# Slime picks HF loading when --load is not a Megatron checkpoint (no
# latest_checkpointed_iteration.txt) and --megatron-to-hf-mode bridge (required by checkpoint.py).
#
# Usage:
#   HF_CHECKPOINT=/path/to/Qwen3-0.6B NPROC_PER_NODE=2 ./scripts/minimal_megatron_load_from_hf.sh
#
# Requires: same env as slime Megatron training (Megatron-LM on PYTHONPATH, CUDA, etc.)

set -euo pipefail

export PYTHONBUFFERED="${PYTHONBUFFERED:-16}"
# Required by Megatron when using tensor or context parallelism (see megatron/training/arguments.py validate_args).
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${HOME}/data}"

# HF weights directory (must contain config.json, safetensors/bin, etc.)
HF_CHECKPOINT="${HF_CHECKPOINT:-${DATA_DIR}/Qwen3-0.6B}"
# Process count = actor GPUs on this node; must match tensor/data parallel layout.
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# Megatron-LM on PYTHONPATH (override with MEGATRON_LM_PATH)
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${HOME}/Megatron-LM}"
if [[ -d "${MEGATRON_LM_PATH}" ]]; then
  export PYTHONPATH="${MEGATRON_LM_PATH}:${PYTHONPATH:-}"
fi

# After load: infer + verify wbridge LoadSpec (see scripts/minimal_megatron_load_checkpoint.py).
export SLIME_WBRIDGE_INFER_LOAD_SPEC="${SLIME_WBRIDGE_INFER_LOAD_SPEC:-1}"

source "${SLIME_DIR}/scripts/models/qwen3-0.6B.sh"

# --load: any non-empty directory that is NOT a Megatron iter checkpoint; using the HF tree is fine.
HF_LOAD_ARGS=(
  --load "${HF_CHECKPOINT}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --megatron-to-hf-mode bridge
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${SLIME_DIR}/scripts/minimal_megatron_load_checkpoint.py" \
  "${HF_LOAD_ARGS[@]}" \
  "${MODEL_ARGS[@]}"
