#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="/usr/data/wjx/genprove_2"
readonly ENV_DIR="${PROJECT_ROOT}/.venv"
readonly PYTHON_BIN="${ENV_DIR}/bin/python"
readonly MODEL_DIR="/usr/data/wjx/genprove_2/models/merge_model/GenProve_sft_200"
readonly RELEVANCE_MODEL_DIR="/usr/data/wjx/trove/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
readonly DATASET_PATH="/usr/data/wjx/genprove_2/data/train_data/en_rl_annotation_with_refs.jsonl"
readonly REWARD_PLUGIN="/usr/data/wjx/genprove_2/code/training/Text_quality_only_annotation.py"

for required_path in "${PYTHON_BIN}" "${MODEL_DIR}" "${RELEVANCE_MODEL_DIR}" "${DATASET_PATH}" "${REWARD_PLUGIN}"; do
    if [[ ! -e "${required_path}" ]]; then
        printf 'Missing required path: %s\n' "${required_path}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" - <<'PY'
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

model_path = Path("/usr/data/wjx/genprove_2/models/merge_model/GenProve_sft_200")
similarity_path = Path("/usr/data/wjx/trove/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
dataset_path = Path("/usr/data/wjx/genprove_2/data/train_data/en_rl_annotation_with_refs.jsonl")

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
