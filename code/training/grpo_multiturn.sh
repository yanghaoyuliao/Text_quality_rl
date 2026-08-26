#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

readonly ENV_DIR="${GENPROVE_ENV_DIR:-${REPO_ROOT}/.venv}"
readonly SWIFT_BIN="${ENV_DIR}/bin/swift"
readonly MODEL_PATH="${GENPROVE_MODEL_PATH:-${REPO_ROOT}/models/merge_model/GenProve_sft_200}"
readonly REWARD_PLUGIN="${GENPROVE_REWARD_PLUGIN:-${REPO_ROOT}/code/training/Text_quality_only_annotation.py}"
readonly DATASET_PATH="${GENPROVE_DATASET_PATH:-${REPO_ROOT}/data/train_data/en_rl_annotation_with_refs.jsonl}"
readonly OUTPUT_DIR="${GENPROVE_OUTPUT_DIR:-${REPO_ROOT}/models/rl_train_model}"

for required_path in "${SWIFT_BIN}" "${MODEL_PATH}" "${REWARD_PLUGIN}" "${DATASET_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        printf 'Missing required path: %s\n' "${required_path}" >&2
        exit 1
    fi
done

if [[ ! -x "${SWIFT_BIN}" ]]; then
    printf 'Swift launcher is not executable: %s\n' "${SWIFT_BIN}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=23456 \
NPROC_PER_NODE=8 \
"${SWIFT_BIN}" rlhf \
    --rlhf_type grpo \
    --model "${MODEL_PATH}" \
    --model_type qwen3_5 \
    --external_plugins "${REWARD_PLUGIN}" \
    --reward_funcs text_quality_reward \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --torch_dtype bfloat16 \
    --dataset "${DATASET_PATH}" \
    --max_length 8192 \
    --max_completion_length 4096 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 16 \
    --save_steps 50 \
    --logging_steps 1 \
    --output_dir "${OUTPUT_DIR}" \
    --dataloader_num_workers 4 \
    --num_generations 4 \
    --generation_batch_size 8 \
    --temperature 1.0 \
    --log_completions true \
    --beta 0.02 \
    --num_iterations 4 \
    --deepspeed zero3 \
    --gradient_checkpointing True \
    --warmup_ratio 0.01 \
    --enable_thinking false \
    --report_to swanlab \
    --swanlab_project GenProve_2
