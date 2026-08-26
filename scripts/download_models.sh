#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

readonly ENV_DIR="${GENPROVE_ENV_DIR:-${REPO_ROOT}/.venv}"
readonly MODELSCOPE_BIN="${ENV_DIR}/bin/modelscope"
readonly MODEL_DIR="${GENPROVE_MODEL_PATH:-${REPO_ROOT}/models/merge_model/GenProve_sft_200}"
readonly RELEVANCE_MODEL_DIR="${TEXT_QUALITY_RELEVANCE_MODEL:-${REPO_ROOT}/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
readonly DATASET_PATH="${GENPROVE_DATASET_PATH:-${REPO_ROOT}/data/train_data/en_rl_annotation_with_refs.jsonl}"
readonly DATASET_DIR="$(dirname -- "${DATASET_PATH}")"
readonly DATASET_FILENAME="en_rl_annotation_with_refs.jsonl"

if [[ ! -x "${MODELSCOPE_BIN}" ]]; then
    printf 'ModelScope CLI not found: %s\nRun %s first.\n' "${MODELSCOPE_BIN}" "${REPO_ROOT}/scripts/setup_grpo_env.sh" >&2
    exit 1
fi
mkdir -p "${MODEL_DIR}" "${RELEVANCE_MODEL_DIR}" "${DATASET_DIR}"

"${MODELSCOPE_BIN}" download \
    --model "LeonYoung/GenProve_sft_200" \
    --revision master \
    --local_dir "${MODEL_DIR}" \
    --max-workers 8

"${MODELSCOPE_BIN}" download \
    --model "LeonYoung/paraphrase-multilingual-MiniLM-L12-v2" \
    --revision master \
    --local_dir "${RELEVANCE_MODEL_DIR}" \
    --max-workers 8

"${MODELSCOPE_BIN}" download \
    --model "LeonYoung/en_rl_annotation_with_refs.jsonl" \
    --revision master \
    --local_dir "${DATASET_DIR}" \
    "${DATASET_FILENAME}" \
    --max-workers 4

DOWNLOADED_DATASET_PATH="${DATASET_DIR}/${DATASET_FILENAME}"
if [[ "${DOWNLOADED_DATASET_PATH}" != "${DATASET_PATH}" ]]; then
    mv -f -- "${DOWNLOADED_DATASET_PATH}" "${DATASET_PATH}"
fi

if [[ ! -s "${DATASET_PATH}" ]]; then
    printf 'Dataset download did not produce a non-empty file: %s\n' "${DATASET_PATH}" >&2
    exit 1
fi

printf 'Model and dataset downloads completed.\nModel: %s\nSimilarity model: %s\nDataset: %s\n' \
    "${MODEL_DIR}" "${RELEVANCE_MODEL_DIR}" "${DATASET_PATH}"
