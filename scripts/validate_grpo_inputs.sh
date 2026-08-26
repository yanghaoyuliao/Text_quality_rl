#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

readonly ENV_DIR="${GENPROVE_ENV_DIR:-${REPO_ROOT}/.venv}"
readonly PYTHON_BIN="${ENV_DIR}/bin/python"
readonly MODEL_DIR="${GENPROVE_MODEL_PATH:-${REPO_ROOT}/models/merge_model/GenProve_sft_200}"
readonly RELEVANCE_MODEL_DIR="${TEXT_QUALITY_RELEVANCE_MODEL:-${REPO_ROOT}/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
readonly DATASET_PATH="${GENPROVE_DATASET_PATH:-${REPO_ROOT}/data/train_data/en_rl_annotation_with_refs.jsonl}"
readonly REWARD_PLUGIN="${GENPROVE_REWARD_PLUGIN:-${REPO_ROOT}/code/training/Text_quality_only_annotation.py}"

for required_path in "${PYTHON_BIN}" "${MODEL_DIR}" "${RELEVANCE_MODEL_DIR}" "${DATASET_PATH}" "${REWARD_PLUGIN}"; do
    if [[ ! -e "${required_path}" ]]; then
        printf 'Missing required path: %s\n' "${required_path}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" - "${MODEL_DIR}" "${RELEVANCE_MODEL_DIR}" "${DATASET_PATH}" <<'PY'
import numpy as np
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

model_path, similarity_path, dataset_path = map(Path, sys.argv[1:4])

for path in (model_path, similarity_path, dataset_path):
    if path.is_file():
        print(path, "bytes", path.stat().st_size)
    else:
        print(path, "present")

similarity_model = SentenceTransformer(str(similarity_path), device="cpu")
embedding = similarity_model.encode(["input validation"], normalize_embeddings=True)
assert embedding.shape == (1, 384), embedding.shape
assert np.isfinite(embedding).all(), "similarity model returned a non-finite embedding"
print("Similarity model check passed; embedding shape", embedding.shape)
print("Dataset lines", sum(1 for _ in dataset_path.open(encoding="utf-8")))
PY

printf 'GRPO inputs are ready.\n'
