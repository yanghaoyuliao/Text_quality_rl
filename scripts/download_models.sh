#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

readonly ENV_DIR="${GENPROVE_ENV_DIR:-${REPO_ROOT}/.venv}"
readonly MODELSCOPE_BIN="${ENV_DIR}/bin/modelscope"
readonly MODEL_DIR="${GENPROVE_MODEL_PATH:-${REPO_ROOT}/models/merge_model/GenProve_sft_200}"
readonly RELEVANCE_MODEL_DIR="${TEXT_QUALITY_RELEVANCE_MODEL:-${REPO_ROOT}/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"

if [[ ! -x "${MODELSCOPE_BIN}" ]]; then
    printf 'ModelScope CLI not found: %s\nRun %s first.\n' "${MODELSCOPE_BIN}" "${REPO_ROOT}/scripts/setup_grpo_env.sh" >&2
    exit 1
fi
if [[ -z "${MODELSCOPE_API_TOKEN:-}" ]]; then
    printf 'Set MODELSCOPE_API_TOKEN before downloading the controlled model repositories.\n' >&2
    exit 1
fi

mkdir -p "${MODEL_DIR}" "${RELEVANCE_MODEL_DIR}"

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

printf 'Model downloads completed.\nModel: %s\nSimilarity model: %s\n' "${MODEL_DIR}" "${RELEVANCE_MODEL_DIR}"
